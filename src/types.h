#pragma once
#include <cstdint>
#include <string>
#include <array>
#include <cassert>

namespace owen2 {

using Bitboard = uint64_t;
using Square   = int;   // 0..63  (a1=0, h1=7, a8=56, h8=63)
using Move     = uint32_t;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
enum PieceType : int { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, PIECE_TYPE_NB=6 };
enum Piece : int {
    W_PAWN=0, W_KNIGHT=1, W_BISHOP=2, W_ROOK=3, W_QUEEN=4, W_KING=5,
    B_PAWN=6, B_KNIGHT=7, B_BISHOP=8, B_ROOK=9, B_QUEEN=10, B_KING=11,
    NO_PIECE=12, PIECE_NB=12
};

constexpr int BOARD_SIZE = 64;
constexpr int RANK_NB = 8, FILE_NB = 8;

inline Color  color_of(Piece p)      { return p < 6 ? WHITE : BLACK; }
inline PieceType type_of(Piece p)    { return PieceType(p % 6); }
inline Piece  make_piece(Color c, PieceType pt) { return Piece(c*6 + pt); }

inline int rank_of(Square s) { return s / 8; }
inline int file_of(Square s) { return s % 8; }
inline Square make_square(int file, int rank) { return rank*8 + file; }
inline Color  operator~(Color c) { return Color(c ^ 1); }

// Move encoding: 6 bits from, 6 bits to, 4 bits promo, 4 bits flags
// flags: bit0 capture, bit1 double-push, bit2 en-passant, bit3 castling
namespace MoveFlag {
    constexpr int CAPTURE   = 1<<0;
    constexpr int DOUBLE    = 1<<1;
    constexpr int ENPASSANT = 1<<2;
    constexpr int CASTLING  = 1<<3;
    constexpr int PROMO     = 1<<4;
}
inline Move make_move(Square from, Square to, int flags=0, PieceType promo=QUEEN) {
    return Move(from | (to<<6) | (promo<<12) | (flags<<16));
}
inline Square move_from(Move m) { return Square(m & 0x3F); }
inline Square move_to(Move m)   { return Square((m>>6) & 0x3F); }
inline PieceType move_promo(Move m){ return PieceType((m>>12)&0xF); }
inline int      move_flags(Move m){ return (m>>16)&0x1F; }
inline bool     is_capture(Move m){ return move_flags(m) & MoveFlag::CAPTURE; }
inline bool     is_promo(Move m)  { return move_flags(m) & MoveFlag::PROMO; }

inline std::string sq_to_str(Square s){
    std::string r; r += char('a'+file_of(s)); r += char('1'+rank_of(s)); return r;
}
inline std::string move_to_uci(Move m){
    std::string s = sq_to_str(move_from(m)) + sq_to_str(move_to(m));
    if(is_promo(m)){
        const char pc[]={'?','n','b','r','q','k'};
        s += pc[move_promo(m)];
        // type_of mapping: PNBRQK -> 0..5; promo stored as PieceType
        // 1=N,2=B,3=R,4=Q
    }
    return s;
}
inline std::string move_to_str(Move m){ return move_to_uci(m); }

// Value / Score
using Value = int; // centipawns
constexpr Value VALUE_INFINITE = 32000;
constexpr Value VALUE_MATE     = 30000;
constexpr Value VALUE_DRAW     = 0;
inline Value mate_in(int ply){ return VALUE_MATE - ply; }
inline Value mated_in(int ply){ return -VALUE_MATE + ply; }
inline bool is_mate_score(Value v){ return std::abs(v) > VALUE_MATE - 1000; }

} // namespace owen2
