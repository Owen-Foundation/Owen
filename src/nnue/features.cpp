#include "features.h"
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace owen2::nnue {

inline void add_weights_avx(int16_t* dst, const int16_t* src) {
#if defined(__AVX2__)
    // 1024 int16 = 64 x __m256i (16 lanes). Wraps mod 2^16 like scalar.
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i r = _mm256_add_epi16(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
#elif defined(__ARM_NEON)
    // 1024 int16 = 128 x int16x8. Wraps mod 2^16 like scalar.
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        int16x8_t d = vld1q_s16(dst + i);
        int16x8_t s = vld1q_s16(src + i);
        vst1q_s16(dst + i, vaddq_s16(d, s));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i) dst[i] += src[i];
#endif
}

inline void sub_weights_avx(int16_t* dst, const int16_t* src) {
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i d = _mm256_loadu_si256((const __m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256i r = _mm256_sub_epi16(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 8) {
        int16x8_t d = vld1q_s16(dst + i);
        int16x8_t s = vld1q_s16(src + i);
        vst1q_s16(dst + i, vsubq_s16(d, s));
    }
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i) dst[i] -= src[i];
#endif
}

int feature_index(Color perspective, Square kingSq, Piece piece, Square pieceSq){
    auto orient = [&](Square s)->Square { return perspective==BLACK ? Square(s ^ 56) : s; };
    Square k = orient(kingSq);
    Square ps = orient(pieceSq);
    int pc = (int)piece;
    if(pc==W_KING || pc==B_KING) return -1;
    int pc10 = pc < 6 ? pc : (pc-6)+5;
    if(pc10<0||pc10>=10) return -1;
    return pc10*4096 + k*64 + ps; // 0..40959
}
int threat_index(Color perspective, Square kingSq, Piece piece, Square pieceSq){
    int base = feature_index(perspective, kingSq, piece, pieceSq);
    if(base < 0) return -1;
    return HALFKP_SIZE + base; // 40960..81919
}

void refresh_accumulator(const Position& pos, Accumulator& acc, const int16_t* weights){
    acc.white.fill(0); acc.black.fill(0);
    Square wk = pos.king_sq(WHITE), bk = pos.king_sq(BLACK);
    for(int s=0;s<64;++s){
        Piece p = pos.piece_on(Square(s));
        if(p==NO_PIECE) continue;
        if(type_of(p)==KING) continue;
        int idxW = feature_index(WHITE, wk, p, Square(s));
        int idxB = feature_index(BLACK, bk, p, Square(s));
        if(idxW>=0){
            const int16_t* w = weights + idxW * HIDDEN_SIZE;
            add_weights_avx(acc.white.data(), w);
        }
        if(idxB>=0){
            const int16_t* w = weights + idxB * HIDDEN_SIZE;
            add_weights_avx(acc.black.data(), w);
        }
        // Threat inputs: + second plane if piece is threatened
        if(is_threatened(pos, Square(s), WHITE) && idxW>=0){
            int tW = threat_index(WHITE, wk, p, Square(s));
            const int16_t* w = weights + tW * HIDDEN_SIZE;
            add_weights_avx(acc.white.data(), w);
        }
        if(is_threatened(pos, Square(s), BLACK) && idxB>=0){
            int tB = threat_index(BLACK, bk, p, Square(s));
            const int16_t* w = weights + tB * HIDDEN_SIZE;
            add_weights_avx(acc.black.data(), w);
        }
    }
    acc.computed_white=acc.computed_black=true;
}

// ---- Incremental evaluation ----

static uint64_t threat_mask_for(const Position& pos, Color perspective) {
    uint64_t m = 0;
    Color enemy = Color(perspective ^ 1);
    for (int s = 0; s < 64; ++s)
        if (pos.square_attacked(Square(s), enemy)) m |= (1ULL << s);
    return m;
}

static void refresh_side(const Position& pos, Color side,
                         std::array<int16_t, HIDDEN_SIZE>& out,
                         const int16_t* weights) {
    out.fill(0);
    Square k = pos.king_sq(side);
    for (int s = 0; s < 64; ++s) {
        Piece p = pos.piece_on(Square(s));
        if (p == NO_PIECE || type_of(p) == KING) continue;
        int idx = feature_index(side, k, p, Square(s));
        if (idx >= 0) add_weights_avx(out.data(), weights + idx * HIDDEN_SIZE);
        if (is_threatened(pos, Square(s), side)) {
            int t = threat_index(side, k, p, Square(s));
            if (t >= 0) add_weights_avx(out.data(), weights + t * HIDDEN_SIZE);
        }
    }
}

void refresh_acc_state(const Position& pos, AccState& st,
                       const int16_t* weights) {
    refresh_side(pos, WHITE, st.acc.white, weights);
    refresh_side(pos, BLACK, st.acc.black, weights);
    st.acc.computed_white = st.acc.computed_black = true;
    st.threatW = threat_mask_for(pos, WHITE);
    st.threatB = threat_mask_for(pos, BLACK);
}

AccMove capture_acc_move(const Position& pre, Move m) {
    AccMove u;
    u.from = move_from(m);
    u.to = move_to(m);
    int flags = move_flags(m);
    u.mover = pre.piece_on(u.from);
    Color us = color_of(u.mover);
    u.capSq = u.to;
    if (flags & MoveFlag::CAPTURE) {
        if (flags & MoveFlag::ENPASSANT)
            u.capSq = make_square(file_of(u.to), rank_of(u.from));
        u.captured = pre.piece_on(u.capSq);
    }
    u.finalPiece = (flags & MoveFlag::PROMO)
        ? make_piece(us, move_promo(m)) : u.mover;
    if (flags & MoveFlag::CASTLING) {
        u.castle = true;
        int home = (us == WHITE) ? 0 : 7;
        if (file_of(u.to) > file_of(u.from)) {  // kingside
            u.rookFrom = make_square(7, home);
            u.rookTo = make_square(5, home);
        } else {  // queenside
            u.rookFrom = make_square(0, home);
            u.rookTo = make_square(3, home);
        }
        u.rook = pre.piece_on(u.rookFrom);
    }
    if (type_of(u.mover) == KING) {
        if (us == WHITE) u.whiteKingMoved = true;
        else u.blackKingMoved = true;
    }
    return u;
}

static void sub_feature(std::array<int16_t, HIDDEN_SIZE>& out,
                        const int16_t* weights, Color side, Square king,
                        Piece p, Square sq) {
    int idx = feature_index(side, king, p, sq);
    if (idx >= 0) sub_weights_avx(out.data(), weights + idx * HIDDEN_SIZE);
}

static void add_feature(std::array<int16_t, HIDDEN_SIZE>& out,
                        const int16_t* weights, Color side, Square king,
                        Piece p, Square sq) {
    int idx = feature_index(side, king, p, sq);
    if (idx >= 0) add_weights_avx(out.data(), weights + idx * HIDDEN_SIZE);
}

static void sub_threat(std::array<int16_t, HIDDEN_SIZE>& out,
                       const int16_t* weights, Color side, Square king,
                       Piece p, Square sq) {
    int t = threat_index(side, king, p, sq);
    if (t >= 0) sub_weights_avx(out.data(), weights + t * HIDDEN_SIZE);
}

static void add_threat(std::array<int16_t, HIDDEN_SIZE>& out,
                       const int16_t* weights, Color side, Square king,
                       Piece p, Square sq) {
    int t = threat_index(side, king, p, sq);
    if (t >= 0) add_weights_avx(out.data(), weights + t * HIDDEN_SIZE);
}

void apply_acc_move(const Position& post, const AccMove& u,
                    const int16_t* weights,
                    const AccState& parent, AccState& child) {
    child.acc = parent.acc;  // copy both perspectives (2KB)
    Square wk = post.king_sq(WHITE), bk = post.king_sq(BLACK);

    // A king move shifts every index of that perspective: full refresh.
    // (Threat adjustment below is then skipped for the refreshed side —
    // refresh_side already includes the post-move threat rows.)
    bool refreshedW = u.whiteKingMoved;
    bool refreshedB = u.blackKingMoved;

    // White perspective.
    if (u.whiteKingMoved) {
        refresh_side(post, WHITE, child.acc.white, weights);
    } else {
        auto& out = child.acc.white;
        sub_feature(out, weights, WHITE, wk, u.mover, u.from);
        add_feature(out, weights, WHITE, wk, u.finalPiece, u.to);
        if (u.captured != NO_PIECE)
            sub_feature(out, weights, WHITE, wk, u.captured, u.capSq);
        if (u.castle) {
            sub_feature(out, weights, WHITE, wk, u.rook, u.rookFrom);
            add_feature(out, weights, WHITE, wk, u.rook, u.rookTo);
        }
    }
    // Black perspective.
    if (u.blackKingMoved) {
        refresh_side(post, BLACK, child.acc.black, weights);
    } else {
        auto& out = child.acc.black;
        sub_feature(out, weights, BLACK, bk, u.mover, u.from);
        add_feature(out, weights, BLACK, bk, u.finalPiece, u.to);
        if (u.captured != NO_PIECE)
            sub_feature(out, weights, BLACK, bk, u.captured, u.capSq);
        if (u.castle) {
            sub_feature(out, weights, BLACK, bk, u.rook, u.rookFrom);
            add_feature(out, weights, BLACK, bk, u.rook, u.rookTo);
        }
    }
    child.acc.computed_white = child.acc.computed_black = true;

    // Threat plane: recompute masks on the post-move position (cheap
    // attack queries). A threat ROW depends on (threatened?, piece), and
    // the piece can change without the threat bit flipping (e.g. a pawn
    // stepping onto a threatened square), so resolve every square against
    // BOTH boards: subtract the pre-move piece's row when it was
    // threatened, add the post-move piece's row when it is now. Untouched
    // squares (pre == post piece) naturally reduce to flip handling.
    auto prePieceAt = [&](Square s) -> Piece {
        // Only these squares can differ pre/post; everything else reads
        // the post board (identical). Note `to` needs care: pre holds the
        // victim on direct captures, else it was empty (incl. en-passant
        // landing squares and rook destinations).
        if (s == u.from) return u.mover;
        if (s == u.to) return (u.capSq == u.to) ? u.captured : NO_PIECE;
        if (s == u.capSq) return u.captured;
        if (u.castle && s == u.rookFrom) return u.rook;
        if (u.castle && s == u.rookTo) return NO_PIECE;
        return post.piece_on(s);
    };
    auto threatAdj = [&](std::array<int16_t, HIDDEN_SIZE>& out, Color side,
                         Square king, uint64_t parentMask, uint64_t newMask) {
        for (int s = 0; s < 64; ++s) {
            bool was = (parentMask >> s) & 1ULL;
            bool now = (newMask >> s) & 1ULL;
            if (!was && !now) continue;
            if (was) {
                Piece pre = prePieceAt(Square(s));
                if (pre != NO_PIECE && type_of(pre) != KING)
                    sub_threat(out, weights, side, king, pre, Square(s));
            }
            if (now) {
                Piece pst = post.piece_on(Square(s));
                if (pst != NO_PIECE && type_of(pst) != KING)
                    add_threat(out, weights, side, king, pst, Square(s));
            }
        }
    };
    uint64_t newW = threat_mask_for(post, WHITE);
    uint64_t newB = threat_mask_for(post, BLACK);
    if (!refreshedW) threatAdj(child.acc.white, WHITE, wk, parent.threatW, newW);
    if (!refreshedB) threatAdj(child.acc.black, BLACK, bk, parent.threatB, newB);
    child.threatW = newW;
    child.threatB = newB;
}

} // namespace owen2::nnue
