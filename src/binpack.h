#pragma once
// binpack.h — clean-room implementation of the STOCKFISH-ECOSYSTEM ".binpack"
// training-data BYTE FORMAT (see official-stockfish/Stockfish tools branch,
// docs/binpack.md) for Owen's OWN self-play games.
//
// *** No Stockfish code, weights, or data are used or copied here — only the
// *** documented container format, so Owen's self-play can be stored, merged,
// *** shuffled and inspected with ecosystem-compatible tooling. ***
//
// Contents:
//   BinpackWriter  games -> .binpack  (chunked "BINP" blocks, stem+movetext)
//   BinpackReader  .binpack -> decoded positions (for 71B sdata conversion)
//
// Conventions (match the ecosystem spec):
//   score  : int16 centipawns, side-to-move perspective, per position
//   result : -1/0/+1 from the side-to-move perspective at that position
//   ply    : absolute game halfmove ply
//   rule50 : halfmove clock

#include "types.h"
#include "position.h"
#include "bitboard.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace binpack {

// ── scalar codec (zigzag-ish, bit-exact per spec) ────────────────────────────
inline std::uint16_t signedToUnsigned(std::int16_t a) {
    std::uint16_t r;
    std::memcpy(&r, &a, sizeof(std::uint16_t));
    if (r & 0x8000) r ^= 0x7FFF; // flip value bits if negative
    r = (r << 1) | (r >> 15);    // store sign bit at bit 0
    return r;
}
inline std::int16_t unsignedToSigned(std::uint16_t r) {
    std::uint16_t v = (r >> 1) | ((r & 1) ? 0x8000 : 0);
    if (v & 0x8000) v ^= 0x7FFF;
    std::int16_t a;
    std::memcpy(&a, &v, sizeof(std::int16_t));
    return a;
}

// ── bit helpers ──────────────────────────────────────────────────────────────
inline int usedBits(std::uint64_t x) { return x == 0 ? 0 : (64 - __builtin_clzll(x)); }
inline int usedBitsSafe(std::size_t v) { return v == 0 ? 0 : usedBits((std::uint64_t)v - 1); }
inline owen2::Bitboard belowMask(int sq) { return sq <= 0 ? 0ULL : ((sq >= 64) ? ~0ULL : ((1ULL << sq) - 1)); }

// ── id mappings ─────────────────────────────────────────────────────────────
// binpack piece id = type*2 + color  (P=0 N=1 B=2 R=3 Q=4 K=5; W=0 B=1; none=12)
inline int bpPieceId(owen2::Piece p) {
    if (p == owen2::NO_PIECE) return 12;
    return (int)owen2::type_of(p) * 2 + (int)owen2::color_of(p);
}
inline owen2::Piece bpIdToPiece(int id) {
    if (id < 0 || id > 11) return owen2::NO_PIECE;
    return owen2::make_piece((owen2::Color)(id & 1), (owen2::PieceType)(id >> 1));
}

// Effective ep square: encoded only when a pawn of the side to move can
// LEGALLY capture there (mirrors reference nullifyEpSquareIfNotPossible,
// hot path + pin cold path).
inline bool epCaptureLegal(const owen2::Position& pos, int epsq, int pawnSq) {
    using namespace owen2;
    Color stm = pos.side_to_move();
    int ksq = pos.king_sq(stm);
    // enemy sliders
    Bitboard bishops = pos.pieces(Color(stm ^ 1), BISHOP);
    Bitboard rooks = pos.pieces(Color(stm ^ 1), ROOK);
    Bitboard queens = pos.pieces(Color(stm ^ 1), QUEEN);
    Bitboard sliders = bishops | rooks | queens;
    if (sliders == 0) return true;
    // pseudo queen rays from king, ignoring blockers: any alignment?
    Bitboard rays = bishop_attacks(ksq, 0) | rook_attacks(ksq, 0);
    if ((sliders & rays) == 0) return true;
    // occupancy after the ep capture: mover off, placed on ep, victim off
    int capSq = make_square(file_of(epsq), rank_of(pawnSq));
    Bitboard occ = pos.occupancy() ^ sq_bb(pawnSq);
    occ |= sq_bb(epsq);
    occ ^= sq_bb(capSq);
    if ((bishop_attacks(ksq, occ) & (bishops | queens)) == 0 &&
        (rook_attacks(ksq, occ) & (rooks | queens)) == 0)
        return true;
    return false;
}
inline int effEp(const owen2::Position& pos) {
    int epsq = pos.ep_square();
    if (epsq < 0 || epsq >= 64) return 64;
    owen2::Color stm = pos.side_to_move();
    owen2::Bitboard pawns = pos.pieces(stm, owen2::PAWN);
    while (pawns) {
        int s = owen2::pop_lsb(pawns);
        const auto& tab = (stm == owen2::WHITE) ? owen2::PawnAttacksWhite : owen2::PawnAttacksBlack;
        if (tab[s] & owen2::sq_bb(epsq)) {
            if (epCaptureLegal(pos, epsq, s)) return epsq;
        }
    }
    return 64;
}
// A position is binpack-representable unless an ep square is set while a
// WRONG-COLOR pawn sits on the ep file at the encoding rank (nibble 12 would
// assign it the opposite color on decode — a format limitation the reference
// shares; we skip stemming there instead of emitting corrupt bytes).
inline bool representable(const owen2::Position& pos) {
    int epsq = effEp(pos);
    if (epsq < 0 || epsq >= 64) return true;
    int ef = owen2::file_of(epsq);
    owen2::Color stm = pos.side_to_move();
    owen2::Bitboard b = pos.occupancy();
    while (b) {
        int sq = owen2::pop_lsb(b);
        if (owen2::file_of(sq) != ef) continue;
        int r = owen2::rank_of(sq);
        bool fires = (r == 3 && stm == owen2::BLACK) || (r == 4 && stm == owen2::WHITE);
        if (!fires) continue;
        owen2::Piece pc = pos.piece_on(sq);
        if (owen2::type_of(pc) != owen2::PAWN) continue;
        bool wantWhite = (r == 3); // decode rule: rank4->white, rank5->black
        if ((owen2::color_of(pc) == owen2::WHITE) != wantWhite) return false;
    }
    return true;
}

// ── move mapping ────────────────────────────────────────────────────────────
// binpack move type: 0 normal, 1 promotion, 2 castle (king-captures-rook), 3 ep
struct BpMove { int from = 0, to = 0, type = 0, promo = (int)owen2::QUEEN; };
inline BpMove toBp(owen2::Move m) {
    int flags = owen2::move_flags(m);
    int from = owen2::move_from(m), to = owen2::move_to(m);
    BpMove b; b.from = from; b.to = to;
    if (flags & owen2::MoveFlag::CASTLING) {
        b.type = 2;
        int r = owen2::rank_of(from);
        // king-captures-rook: kingside (king went to g-file) -> h-file rook
        b.to = owen2::make_square(owen2::file_of(to) == 6 ? 7 : 0, r);
    } else if (flags & owen2::MoveFlag::PROMO) {
        b.type = 1;
        int pt = (int)owen2::move_promo(m);
        b.promo = (pt >= 1 && pt <= 4) ? pt : (int)owen2::QUEEN;
    } else if (flags & owen2::MoveFlag::ENPASSANT) {
        b.type = 3;
    }
    return b;
}
inline std::uint16_t packBpMove(const BpMove& b) {
    std::uint16_t v = (std::uint16_t)((b.type & 3) << 14) | (std::uint16_t)((b.from & 63) << 8) |
                      (std::uint16_t)((b.to & 63) << 2);
    if (b.type == 1) v |= (std::uint16_t)((b.promo - 1) & 3);
    return v;
}
// binpack move -> owen2 move (flags derived from the CURRENT board position)
// NOTE: the movetext stream does not store Normal-vs-EnPassant; like the
// reference decoder we return Normal and detect the ep capture by square
// (a diagonal pawn move onto an empty square is only encodable as ep).
inline owen2::Move bpToOwen(const owen2::Position& pos, int from, int to, int type, int promoIdx) {
    using namespace owen2;
    Color us = pos.side_to_move();
    if (type == 2) { // castle: king-captures-rook -> king destination
        int r = rank_of(from);
        int dest = make_square(file_of(to) == 7 ? 6 : 2, r);
        return make_move(from, dest, MoveFlag::CASTLING);
    }
    if (type == 1) { // promotion
        PieceType pt = (PieceType)(promoIdx + 1);
        if (pt < KNIGHT || pt > QUEEN) pt = QUEEN;
        int fl = MoveFlag::PROMO;
        if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) != us) fl |= MoveFlag::CAPTURE;
        return make_move(from, to, fl, pt);
    }
    if (type == 3) return make_move(from, to, MoveFlag::ENPASSANT | MoveFlag::CAPTURE);
    int fl = 0;
    Piece moving = pos.piece_on(from);
    if (type_of(moving) == PAWN) {
        if (abs(rank_of(to) - rank_of(from)) == 2) fl |= MoveFlag::DOUBLE;
        // diagonal onto empty square == en passant capture (only encodable case)
        if (file_of(to) != file_of(from) && pos.piece_on(to) == NO_PIECE)
            fl |= MoveFlag::ENPASSANT | MoveFlag::CAPTURE;
        else if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) != us)
            fl |= MoveFlag::CAPTURE;
    } else if (pos.piece_on(to) != NO_PIECE && color_of(pos.piece_on(to)) != us) {
        fl |= MoveFlag::CAPTURE;
    }
    return make_move(from, to, fl);
}

// ── pseudo-legal destinations (spec: pseudo attacks & ~own, ep included) ────
inline owen2::Bitboard pseudoDests(const owen2::Position& pos, int from, owen2::PieceType pt,
                                   int* numMovesOut = nullptr) {
    using namespace owen2;
    Color stm = pos.side_to_move();
    Bitboard ours = pos.pieces(stm);
    Bitboard theirs = pos.pieces(Color(stm ^ 1));
    Bitboard occ = ours | theirs;
    Bitboard d = 0;
    if (pt == PAWN) {
        Bitboard targets = theirs;
        int epsq = effEp(pos);
        if (epsq >= 0 && epsq < 64) targets |= sq_bb(epsq);
        const auto& ptab = (stm == WHITE) ? PawnAttacksWhite : PawnAttacksBlack;
        d = ptab[from] & targets;
        int dir = (stm == WHITE) ? 8 : -8;
        int f1 = from + dir;
        if (f1 >= 0 && f1 < 64 && !(occ & sq_bb(f1))) {
            d |= sq_bb(f1);
            int startRank = (stm == WHITE) ? 1 : 6;
            int f2 = f1 + dir;
            if (rank_of(from) == startRank && f2 >= 0 && f2 < 64 && !(occ & sq_bb(f2)))
                d |= sq_bb(f2);
        }
    } else if (pt == KNIGHT) {
        d = KnightAttacks[from] & ~ours;
    } else if (pt == KING) {
        d = KingAttacks[from] & ~ours; // castling counted by caller
    } else if (pt == BISHOP) {
        d = bishop_attacks(from, occ) & ~ours;
    } else if (pt == ROOK) {
        d = rook_attacks(from, occ) & ~ours;
    } else { // QUEEN
        d = queen_attacks(from, occ) & ~ours;
    }
    if (numMovesOut) *numMovesOut = popcount(d);
    return d;
}

// move-id for the movetext stream (also returns numMoves for bit width)
inline int moveIdEncode(const owen2::Position& pos, const BpMove& b, int* numMovesOut) {
    using namespace owen2;
    Color stm = pos.side_to_move();
    Piece pc = pos.piece_on(b.from);
    PieceType pt = type_of(pc);
    int numMoves = 0;
    int moveId = 0;
    if (pt == PAWN) {
        Bitboard d = pseudoDests(pos, b.from, PAWN, &numMoves);
        moveId = popcount(d & belowMask(b.to));
        int promoRank = (stm == WHITE) ? 6 : 1;
        if (rank_of(b.from) == promoRank) {
            int pi = b.promo - 1;
            if (pi < 0 || pi > 3) pi = 3;
            moveId = moveId * 4 + pi;
            numMoves *= 4;
        }
    } else if (pt == KING) {
        Bitboard d = pseudoDests(pos, b.from, KING);
        int attacksSize = popcount(d);
        int rights = pos.castling_rights();
        int ourMask = (stm == WHITE) ? 3 : 12;
        int numCastle = popcount((Bitboard)(rights & ourMask));
        numMoves = attacksSize + numCastle;
        if (b.type == 2) { // castle
            int longBit = (stm == WHITE) ? 2 : 8;
            moveId = attacksSize - 1;
            if (rights & longBit) moveId += 1;        // long possible
            if (file_of(b.to) == 7) moveId += 1;      // short played (rook on h-file)
        } else {
            moveId = popcount(d & belowMask(b.to));
        }
    } else {
        Bitboard d = pseudoDests(pos, b.from, pt, &numMoves);
        moveId = popcount(d & belowMask(b.to));
    }
    if (numMovesOut) *numMovesOut = numMoves;
    return moveId;
}

// ── 24-byte compressed position ─────────────────────────────────────────────
inline void writePos24(const owen2::Position& pos, unsigned char out[24]) {
    using namespace owen2;
    Bitboard occ = pos.occupancy();
    Color stm = pos.side_to_move();
    int rights = pos.castling_rights();
    int epsq = effEp(pos);
    int epFile = (epsq >= 0 && epsq < 64) ? file_of(epsq) : -1;
    // occupied, big endian
    for (int i = 0; i < 8; ++i) out[i] = (unsigned char)(occ >> (56 - 8 * i));
    unsigned char packed[16] = {0};
    Bitboard b = occ;
    int i = 0;
    while (b) {
        int sq = pop_lsb(b);
        Piece pc = pos.piece_on(sq);
        int id = bpPieceId(pc);
        PieceType pt = type_of(pc);
        Color c = color_of(pc);
        if (pt == PAWN && epFile >= 0 && file_of(sq) == epFile &&
            ((rank_of(sq) == 3 && stm == BLACK) || (rank_of(sq) == 4 && stm == WHITE))) {
            id = 12;
        } else if (pt == ROOK) {
            if (c == WHITE && ((sq == 0 && (rights & 2)) || (sq == 7 && (rights & 1)))) id = 13;
            if (c == BLACK && ((sq == 56 && (rights & 8)) || (sq == 63 && (rights & 4)))) id = 14;
        } else if (pt == KING && c == BLACK) {
            id = (stm == BLACK) ? 15 : 11;
        } else if (pt == KING) {
            id = 10;
        }
        if (i & 1) packed[i >> 1] |= (unsigned char)(id << 4);
        else packed[i >> 1] |= (unsigned char)(id & 0xF);
        ++i;
    }
    std::memcpy(out + 8, packed, 16);
}

struct DecodedPos {
    int board[64];   // owen2 Piece ids (NO_PIECE=12)
    int stm = 0;     // 0 white 1 black
    int rights = 0;  // K=1 Q=2 k=4 q=8
    int ep = 64;     // 64 none
    int rule50 = 0;
    int ply = 0;
    int score = 0;   // stm perspective cp
    int result = 0;  // -1/0/+1 stm perspective
    // outgoing move (owen form: castle = king destination) for v3 policy labels
    int mv_from = 0, mv_to = 0, mv_promo = 0; // promo: 0 none, 1 N, 2 B, 3 R, 4 Q
    bool mv_ep = false;
    std::uint16_t move16() const {
        return (std::uint16_t)(mv_from | (mv_to << 6) | (mv_promo << 12) | (mv_ep ? (1 << 15) : 0));
    }
};
// bp-form (castle = king-captures-rook) -> owen-form label fields
inline void bpLabel(int from, int to, int type, int promoIdx, int& of, int& ot, int& op, bool& oep) {
    of = from; ot = to; op = 0; oep = false;
    if (type == 2) { // castle -> king destination
        ot = owen2::make_square(owen2::file_of(to) == 7 ? 6 : 2, owen2::rank_of(from));
    } else if (type == 1) {
        op = promoIdx + 1;
        if (op < 1 || op > 4) op = 4;
    } else if (type == 3) {
        oep = true;
    }
}
inline void readPos24(const unsigned char in[24], DecodedPos& dp) {
    std::uint64_t occ = 0;
    for (int i = 0; i < 8; ++i) occ = (occ << 8) | in[i];
    for (int s = 0; s < 64; ++s) dp.board[s] = (int)owen2::NO_PIECE;
    dp.stm = 0; dp.rights = 0; dp.ep = 64;
    int i = 0;
    while (occ) {
        int sq = owen2::lsb(occ); occ &= occ - 1;
        int nib = (i & 1) ? (in[8 + (i >> 1)] >> 4) : (in[8 + (i >> 1)] & 0xF);
        ++i;
        if (nib <= 11) {
            dp.board[sq] = (int)bpIdToPiece(nib);
        } else if (nib == 12) {
            int r = owen2::rank_of(sq), f = owen2::file_of(sq);
            if (r == 3) { dp.board[sq] = 0; dp.ep = owen2::make_square(f, 2); }        // white pawn, ep behind
            else { dp.board[sq] = 6; dp.ep = owen2::make_square(f, 5); }               // black pawn, ep ahead
        } else if (nib == 13) {
            dp.board[sq] = 3;
            dp.rights |= (sq == 0) ? 2 : 1;
        } else if (nib == 14) {
            dp.board[sq] = 9;
            dp.rights |= (sq == 56) ? 8 : 4;
        } else { // 15
            dp.board[sq] = 11;
            dp.stm = 1;
        }
    }
}

// ── bitstream (MSB-first packing, matches spec) ─────────────────────────────
struct BitWriter {
    std::vector<unsigned char> bytes;
    int bitsLeft = 0;
    void add(std::uint8_t bits, int count) {
        if (count == 0) return;
        if (bitsLeft == 0) {
            bytes.push_back((unsigned char)(bits << (8 - count)));
            bitsLeft = 8;
        } else if (count <= bitsLeft) {
            bytes.back() |= (unsigned char)(bits << (bitsLeft - count));
        } else {
            int spill = count - bitsLeft;
            bytes.back() |= (unsigned char)(bits >> spill);
            bytes.push_back((unsigned char)(bits << (8 - spill)));
            bitsLeft += 8;
        }
        bitsLeft -= count;
    }
    void addVle(std::uint16_t v) { // block size 4
        std::uint16_t mask = (1u << 4) - 1;
        for (;;) {
            std::uint8_t block = (std::uint8_t)((v & mask) | ((v > mask) << 4));
            add(block, 5);
            v >>= 4;
            if (v == 0) break;
        }
    }
};
struct BitReader {
    const unsigned char* data = nullptr;
    std::size_t size = 0, off = 0;
    int left = 8;
    BitReader() {}
    BitReader(const unsigned char* d, std::size_t n) : data(d), size(n) {}
    std::uint8_t get(int count) {
        if (count == 0) return 0;
        if (left == 0) { ++off; left = 8; }
        if (off >= size) throw std::runtime_error("binpack: movetext overrun");
        std::uint8_t byte = (std::uint8_t)(data[off] << (8 - left));
        std::uint8_t bits = (std::uint8_t)(byte >> (8 - count));
        if (count > left) {
            int spill = count - left;
            if (off + 1 >= size) throw std::runtime_error("binpack: movetext overrun");
            bits |= (std::uint8_t)(data[off + 1] >> (8 - spill));
            left += 8; ++off;
        }
        left -= count;
        return bits;
    }
    std::uint16_t getVle() { // block size 4
        std::uint16_t mask = (1u << 4) - 1, v = 0;
        int shift = 0;
        for (;;) {
            std::uint16_t block = get(5);
            v |= (std::uint16_t)((block & mask) << shift);
            if (!(block >> 4)) break;
            shift += 4;
            if (shift > 16) throw std::runtime_error("binpack: bad VLE");
        }
        return v;
    }
    std::size_t consumed() const { return off + (left != 8 ? 1 : 0); }
};

// ── chain entry (one recorded position) ─────────────────────────────────────
struct ChainEntry {
    owen2::Position pos;  // position BEFORE the move
    owen2::Move move;     // move played from it (last entry: searched best, unplayed)
    int score = 0;        // stm-perspective cp at this position
    int result = 0;       // -1/0/+1 stm perspective
};

// ── writer ──────────────────────────────────────────────────────────────────
class Writer {
public:
    explicit Writer(const std::string& path) : f_(path, std::ios::binary | std::ios::trunc) {
        if (!f_) throw std::runtime_error("binpack: cannot open " + path);
    }
    // One game = one chain (split automatically if continuity breaks, or at
    // format-unrepresentable positions which are skipped, losing 1 record).
    // Reference layout: stem = entry 0 (pos/move/score); movetext[k] carries
    // the NEXT position's (move, score): (move_{k+1} @ pos_{k+1}, score_{k+1}).
    void addGame(const std::vector<ChainEntry>& g) {
        if (g.empty()) return;
        bool open = false;
        ChainEntry prev{};
        bool havePrev = false;
        auto openStem = [&](const ChainEntry& e) {
            emitStem(e);
            lastScore_ = (std::int16_t)(-std::clamp(e.score, -32000, 32000));
            numPlies_ = 0; bw_.bytes.clear(); bw_.bitsLeft = 0;
        };
        for (std::size_t idx = 0; idx < g.size(); ++idx) {
            const ChainEntry& e = g[idx];
            if (!open) {
                if (!representable(e.pos)) { havePrev = false; continue; }
                openStem(e);
                open = true;
            } else {
                bool cont = havePrev && (prev.result == -e.result) &&
                            (prev.pos.ply() + 1 == e.pos.ply());
                if (!cont) {
                    closeChain();
                    if (!representable(e.pos)) { open = false; havePrev = false; continue; }
                    openStem(e);
                } else {
                    // encode the NEXT entry's move from the NEXT position
                    addMoveScore(e.pos, toBp(e.move),
                                 (std::int16_t)std::clamp(e.score, -32000, 32000));
                }
            }
            prev = e;
            havePrev = true;
        }
        if (open) closeChain();
        if (buf_.size() >= chunkSize_) flushChunk();
    }
    void close() {
        if (openChain_) closeChain();
        flushChunk();
        f_.close();
    }
    ~Writer() { if (f_.is_open()) { if (openChain_) closeChain(); flushChunk(); } }

private:
    static constexpr std::size_t chunkSize_ = (1u << 20); // 1 MiB
    std::ofstream f_;
    std::vector<char> buf_;
    BitWriter bw_;
    std::int16_t lastScore_ = 0;
    std::uint16_t numPlies_ = 0;
    bool openChain_ = false;

    void put(const void* p, std::size_t n) {
        const char* c = (const char*)p;
        buf_.insert(buf_.end(), c, c + n);
    }
    void put16be(std::uint16_t v) {
        unsigned char b[2] = {(unsigned char)(v >> 8), (unsigned char)(v & 0xFF)};
        put(b, 2);
    }
    void emitStem(const ChainEntry& e) {
        unsigned char pos24[24];
        writePos24(e.pos, pos24);
        put(pos24, 24);
        put16be(packBpMove(toBp(e.move)));
        put16be(signedToUnsigned((std::int16_t)std::clamp(e.score, -32000, 32000)));
        std::uint16_t pr = ((std::uint16_t)e.pos.ply() & 0x3FFF) |
                           (std::uint16_t)(signedToUnsigned((std::int16_t)e.result) << 14);
        put16be(pr);
        put16be((std::uint16_t)e.pos.rule50());
        openChain_ = true;
    }
    void addMoveScore(const owen2::Position& pos, const BpMove& b, std::int16_t score) {
        using namespace owen2;
        Color stm = pos.side_to_move();
        Bitboard ours = pos.pieces(stm);
        int pieceId = popcount(ours & belowMask(b.from));
        int numPieces = popcount(ours);
        int numMoves = 0;
        int moveId = moveIdEncode(pos, b, &numMoves);
        bw_.add((std::uint8_t)pieceId, usedBitsSafe((std::size_t)numPieces));
        bw_.add((std::uint8_t)moveId, usedBitsSafe((std::size_t)numMoves));
        std::uint16_t delta = signedToUnsigned((std::int16_t)(score - lastScore_));
        bw_.addVle(delta);
        lastScore_ = (std::int16_t)(-score);
        ++numPlies_;
    }
    void closeChain() {
        put16be(numPlies_);
        if (numPlies_ > 0) put(bw_.bytes.data(), bw_.bytes.size());
        openChain_ = false;
        numPlies_ = 0; bw_.bytes.clear(); bw_.bitsLeft = 0;
    }
    void flushChunk() {
        if (buf_.empty()) return;
        char hdr[8] = {'B', 'I', 'N', 'P', 0, 0, 0, 0};
        std::uint32_t n = (std::uint32_t)buf_.size();
        hdr[4] = (char)(n & 0xFF); hdr[5] = (char)((n >> 8) & 0xFF);
        hdr[6] = (char)((n >> 16) & 0xFF); hdr[7] = (char)((n >> 24) & 0xFF);
        f_.write(hdr, 8);
        f_.write(buf_.data(), buf_.size());
        buf_.clear();
    }
};

// ── reader ──────────────────────────────────────────────────────────────────
inline owen2::Position fenOf(const DecodedPos& dp) {
    static const char* tab = "PNBRQKpnbrqk";
    std::string fen;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int p = dp.board[r * 8 + f];
            if (p < 0 || p > 11) { ++empty; continue; }
            if (empty) { fen += (char)('0' + empty); empty = 0; }
            fen += tab[p];
        }
        if (empty) fen += (char)('0' + empty);
        if (r) fen += '/';
    }
    fen += (dp.stm == 0) ? " w " : " b ";
    std::string cs;
    if (dp.rights & 1) cs += 'K';
    if (dp.rights & 2) cs += 'Q';
    if (dp.rights & 4) cs += 'k';
    if (dp.rights & 8) cs += 'q';
    fen += cs.empty() ? "-" : cs;
    fen += ' ';
    if (dp.ep >= 0 && dp.ep < 64) fen += owen2::sq_to_str(dp.ep);
    else fen += '-';
    fen += ' ' + std::to_string(dp.rule50) + ' ' + std::to_string(dp.ply / 2 + 1);
    owen2::Position pos;
    pos.set_fen(fen);
    return pos;
}

class Reader {
public:
    inline static bool verbose = false;
    struct Sink {
        virtual ~Sink() {}
        virtual void emit(const DecodedPos& dp) = 0;
    };
    struct VectorSink : Sink {
        std::vector<DecodedPos>& out;
        explicit VectorSink(std::vector<DecodedPos>& o) : out(o) {}
        void emit(const DecodedPos& dp) override { out.push_back(dp); }
    };
    // Returns every decoded position in file order (chain after chain).
    // NOTE: loads everything into RAM — fine for small files, use stream()
    // for multi-GB binpacks.
    static std::vector<DecodedPos> readFile(const std::string& path) {
        std::vector<DecodedPos> out;
        VectorSink sink(out);
        stream(path, sink);
        return out;
    }
    // Streaming decode: constant memory, suitable for GB-scale files.
    static void stream(const std::string& path, Sink& sink) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("binpack: cannot open " + path);
        for (;;) {
            char hdr[8];
            f.read(hdr, 8);
            if (f.gcount() == 0) break; // EOF
            if (f.gcount() != 8 || hdr[0] != 'B' || hdr[1] != 'I' || hdr[2] != 'N' || hdr[3] != 'P')
                throw std::runtime_error("binpack: bad chunk magic");
            std::uint32_t n = (std::uint8_t)hdr[4] | ((std::uint32_t)(std::uint8_t)hdr[5] << 8) |
                              ((std::uint32_t)(std::uint8_t)hdr[6] << 16) |
                              ((std::uint32_t)(std::uint8_t)hdr[7] << 24);
            if (n > 100u * (1u << 20)) throw std::runtime_error("binpack: chunk too large");
            std::vector<unsigned char> chunk(n);
            f.read((char*)chunk.data(), n);
            if ((std::uint32_t)f.gcount() != n) throw std::runtime_error("binpack: truncated chunk");
            std::size_t off = 0;
            while (off < chunk.size()) off = readChain(chunk.data(), chunk.size(), off, sink);
        }
    }

private:
    static std::size_t readChain(const unsigned char* d, std::size_t n, std::size_t off,
                                 Sink& sink) {
        using namespace owen2;
        if (off + 32 > n) throw std::runtime_error("binpack: truncated stem");
        DecodedPos dp;
        readPos24(d + off, dp);
        std::uint16_t mv = ((std::uint16_t)d[off + 24] << 8) | d[off + 25];
        dp.score = unsignedToSigned(((std::uint16_t)d[off + 26] << 8) | d[off + 27]);
        std::uint16_t pr = ((std::uint16_t)d[off + 28] << 8) | d[off + 29];
        dp.ply = pr & 0x3FFF;
        dp.result = unsignedToSigned((std::uint16_t)(pr >> 14));
        dp.rule50 = ((std::uint16_t)d[off + 30] << 8) | d[off + 31];
        int bfrom = (mv >> 8) & 63, bto = (mv >> 2) & 63, btype = mv >> 14, bpromo = mv & 3;
        off += 32;
        if (off + 2 > n) throw std::runtime_error("binpack: truncated count");
        std::uint16_t count = ((std::uint16_t)d[off] << 8) | d[off + 1];
        off += 2;
        bpLabel(bfrom, bto, btype, bpromo, dp.mv_from, dp.mv_to, dp.mv_promo, dp.mv_ep);
        sink.emit(dp);
        int prevPly = dp.ply;
        if (count == 0) return off;
        // Reference layout: movetext[k] = (move_{k+1} @ pos_{k+1}, score_{k+1}).
        // Walk: apply current move -> decode next (move, score) from new pos.
        Position pos = fenOf(dp);
        if (verbose) {
            fprintf(stderr, "[chain @%zu] stem ply=%d res=%d score=%d count=%u fen=%s\n",
                    off, dp.ply, dp.result, dp.score, count, pos.fen().c_str());
        }
        BitReader br(d + off, n - off);
        std::int16_t last = (std::int16_t)(-dp.score);
        int result = dp.result;
        // current move starts as the stem move (leads pos_0 -> pos_1)
        int curFrom = bfrom, curTo = bto, curType = btype, curPromo = bpromo;
        for (std::uint16_t k = 0; k < count; ++k) {
            Move om = bpToOwen(pos, curFrom, curTo, curType, curPromo);
            pos.do_move(om); // now at pos_{k+1}
            Color stm = pos.side_to_move();
            Bitboard ours = pos.pieces(stm);
            int numPieces = popcount(ours);
            int pieceId = br.get(usedBitsSafe((std::size_t)numPieces));
            if (pieceId < 0 || pieceId >= numPieces) throw std::runtime_error("binpack: bad pieceId");
            // from = pieceId-th set bit of ours (LSB order)
            Bitboard b = ours;
            int from = -1;
            for (int j = 0; j <= pieceId; ++j) from = pop_lsb(b);
            if (from < 0 || from >= 64) throw std::runtime_error("binpack: bad from");
            Piece pc = pos.piece_on(from);
            PieceType pt = type_of(pc);
            int moveId, numMoves = 0, to = -1, mtype = 0, promo = (int)QUEEN;
            if (pt == PAWN) {
                Bitboard dests = pseudoDests(pos, from, PAWN, &numMoves);
                int promoRank = (stm == WHITE) ? 6 : 1;
                bool isPromo = (rank_of(from) == promoRank);
                int width = isPromo ? numMoves * 4 : numMoves;
                if (width <= 0) throw std::runtime_error("binpack: no pawn moves");
                moveId = br.get(usedBitsSafe((std::size_t)width));
                if (moveId >= width) throw std::runtime_error("binpack: bad pawn moveId");
                int di = isPromo ? moveId / 4 : moveId;
                if (di >= popcount(dests)) throw std::runtime_error("binpack: bad pawn dest");
                Bitboard dd = dests;
                for (int j = 0; j <= di; ++j) to = pop_lsb(dd);
                if (isPromo) { mtype = 1; promo = (moveId % 4) + 1; }
            } else if (pt == KING) {
                Bitboard dests = pseudoDests(pos, from, KING);
                int attacksSize = popcount(dests);
                int rights = pos.castling_rights();
                int ourMask = (stm == WHITE) ? 3 : 12;
                int numCastle = popcount((Bitboard)(rights & ourMask));
                numMoves = attacksSize + numCastle;
                if (numMoves <= 0) throw std::runtime_error("binpack: no king moves");
                moveId = br.get(usedBitsSafe((std::size_t)numMoves));
                if (moveId >= numMoves) throw std::runtime_error("binpack: bad king moveId");
                if (moveId >= attacksSize) {
                    int idx = moveId - attacksSize;
                    int longBit = (stm == WHITE) ? 2 : 8;
                    bool isLong = (idx == 0 && (rights & longBit));
                    mtype = 2;
                    to = make_square(isLong ? 0 : 7, rank_of(from)); // rook square
                } else {
                    if (moveId >= popcount(dests)) throw std::runtime_error("binpack: bad king dest");
                    Bitboard dd = dests;
                    for (int j = 0; j <= moveId; ++j) to = pop_lsb(dd);
                }
            } else {
                Bitboard dests = pseudoDests(pos, from, pt, &numMoves);
                if (numMoves <= 0) throw std::runtime_error("binpack: no piece moves");
                moveId = br.get(usedBitsSafe((std::size_t)numMoves));
                if (moveId >= numMoves) throw std::runtime_error("binpack: bad moveId");
                Bitboard dd = dests;
                for (int j = 0; j <= moveId; ++j) to = pop_lsb(dd);
            }
            if (to < 0 || to >= 64) throw std::runtime_error("binpack: bad to");
            std::uint16_t raw = br.getVle();
            std::int16_t score = (std::int16_t)(last + unsignedToSigned(raw));
            last = (std::int16_t)(-score);
            result = -result;
            if (verbose) {
                fprintf(stderr, "  e%u: stm=%d %s score=%d res=%d fen=%s\n", k,
                        (int)pos.side_to_move(), owen2::move_to_uci(om).c_str(),
                        score, result, pos.fen().c_str());
            }
            // record pos_{k+1} with its decoded (move, score); the move leads
            // forward and is applied at the top of the next iteration
            DecodedPos ndp;
            for (int s = 0; s < 64; ++s) ndp.board[s] = (int)pos.piece_on(s);
            ndp.stm = (int)pos.side_to_move();
            ndp.rights = pos.castling_rights();
            ndp.ep = pos.ep_square() >= 64 ? 64 : pos.ep_square();
            ndp.rule50 = pos.rule50();
            ndp.score = score;
            ndp.result = result;
            ndp.ply = prevPly + 1;
            prevPly = ndp.ply;
            bpLabel(from, to, mtype, (mtype == 1) ? (promo - 1) : 0,
                    ndp.mv_from, ndp.mv_to, ndp.mv_promo, ndp.mv_ep);
            sink.emit(ndp);
            curFrom = from; curTo = to; curType = mtype;
            curPromo = (mtype == 1) ? (promo - 1) : 0;
        }
        return off + br.consumed();
    }
};

} // namespace binpack
