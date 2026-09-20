#pragma once
#include "../position.h"

namespace owen2 {

// ── Static Exchange Evaluation ─────────────────────────────────────
// Exact-ish swap-list SEE for move ordering and quiescence delta pruning.
// Returns net material swing in cp from the SIDE-TO-MOVE's perspective,
// assuming the capture on `m` is made and both sides recapture optimally.
// Non-captures score 0 (non-capture promos score the promotion delta).
// Capturing a king scores +20000 (unreachable from legal play, guarded).
//
// Approximations (standard, shared with Stockfish-style SEE):
//  - pins ignored (attackers_to is pin-blind),
//  - en-passant pin edge cases ignored,
//  - kings never recapture: a king capture ends the sequence (capturing a
//    king is game-over, and recapturing a king is meaningless — the reply
//    to a king capture would be check, not a capture).
int see(const Position& pos, Move m);
inline bool see_ge(const Position& pos, Move m, int threshold) {
    return see(pos, m) >= threshold;
}

} // namespace owen2
