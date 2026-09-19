#pragma once
#include "position.h"
#include <vector>

namespace owen2 {

// Maximum legal moves in any position (~218); fixed buffers use this.
inline constexpr int kMaxMoves = 256;

std::vector<Move> generate_legal(const Position& pos);
std::vector<Move> generate_pseudo_legal(const Position& pos);
// Buffer API (no heap allocation): writes up to cap moves, returns count.
// `scratch` is assigned from pos (reuses heap capacity after warmup); the
// caller must guarantee exclusive use (one scratch per recursion level).
int generate_pseudo_buf(const Position& pos, Move* out, int cap);
int generate_legal_buf(Position& scratch, const Position& pos, Move* out, int cap);
// Captures + all promotions (quiet pushes included), no quiets/castling:
// exactly the quiescence set outside check evasions (which use full gen).
std::vector<Move> generate_captures(const Position& pos);
int generate_captures_buf(Position& scratch, const Position& pos, Move* out, int cap);
uint64_t perft(Position& pos, int depth);
Move parse_uci_move(const Position& pos, const std::string& s);

} // namespace owen2
