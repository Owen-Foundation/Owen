#pragma once
#include "types.h"
#include <array>

namespace owen2 {

// Bitboard utilities
inline int popcount(Bitboard b){ return __builtin_popcountll(b); }
inline Square lsb(Bitboard b){ return Square(__builtin_ctzll(b)); }
inline Square msb(Bitboard b){ return Square(63 - __builtin_clzll(b)); }
inline Square pop_lsb(Bitboard &b){
    Square s = lsb(b); b &= b-1; return s;
}
inline Bitboard sq_bb(Square s){ return 1ULL << s; }
inline bool has_sq(Bitboard b, Square s){ return (b >> s) & 1ULL; }

// Attack tables
extern std::array<Bitboard,64> KnightAttacks;
extern std::array<Bitboard,64> KingAttacks;
extern std::array<Bitboard,64> PawnAttacksWhite;
extern std::array<Bitboard,64> PawnAttacksBlack;

// Sliding attacks — magic bitboards (clean-room: magics generated at
// startup, every entry verified against the naive slider; see bitboard.cpp)
Bitboard bishop_attacks(Square sq, Bitboard occ);
Bitboard rook_attacks(Square sq, Bitboard occ);
inline Bitboard queen_attacks(Square sq, Bitboard occ){
    return bishop_attacks(sq,occ) | rook_attacks(sq,occ);
}

void init_attacks();

// File / rank masks
constexpr Bitboard FileA = 0x0101010101010101ULL;
constexpr Bitboard FileH = 0x8080808080808080ULL;
constexpr Bitboard Rank1 = 0x00000000000000FFULL;
constexpr Bitboard Rank8 = 0xFF00000000000000ULL;

inline Bitboard shift_north(Bitboard b){ return b << 8; }
inline Bitboard shift_south(Bitboard b){ return b >> 8; }

} // namespace owen2
