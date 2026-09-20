#pragma once
#include "../types.h"
#include "../position.h"

namespace owen2::nnue {

// Owen2 v2 feature set: HalfKP + Threat inputs
// HalfKP + Threat Inputs
// bucketed king: 4 files mirrored -> 4 buckets, but we keep full 64 for accuracy and bucket at runtime
// INPUT = HalfKP (40960) + Threat (40960) = 81920
// H = 1024 (was 256). AVX2/VNNI still fast via int16.
// Threat: for each piece square, add feature if that square is attacked by opponent.
// Threat inputs let the net see hanging pieces / tensions.

constexpr int HIDDEN_SIZE = 1024;
constexpr int HALFKP_SIZE = 40960; // 64*64*10
constexpr int THREAT_SIZE = 40960; // same shape, 1 bit per piece-square = threatened?
constexpr int INPUT_SIZE  = HALFKP_SIZE + THREAT_SIZE; // 81920
constexpr int OUTPUT_BUCKETS = 8;

int feature_index(Color perspective, Square kingSq, Piece piece, Square pieceSq);
int threat_index(Color perspective, Square kingSq, Piece piece, Square pieceSq);

inline bool is_threatened(const Position& pos, Square sq, Color perspective) {
    // threatened = attacked by enemy
    return pos.square_attacked(sq, Color(perspective ^ 1));
}

struct Accumulator {
    std::array<int16_t, HIDDEN_SIZE> white{};
    std::array<int16_t, HIDDEN_SIZE> black{};
    bool computed_white=false, computed_black=false;
};

void refresh_accumulator(const Position& pos, Accumulator& acc,
                         const int16_t* weights);

// ---- Incremental evaluation (path-indexed, undo-free) ----
// A search path never undoes: each descent step recomputes the child slot
// from its parent, so slots only flow parent -> child. AccState carries the
// accumulator plus the threat bitmasks needed to diff the threat plane.
struct AccState {
    Accumulator acc;
    // threatW bit s = square s threatened from WHITE's perspective
    // (attacked by Black); threatB mirrored. Recomputed per step from the
    // post-move position (64 attack queries, ~0.2us) and diffed against
    // the parent masks, so only flipped squares touch weight rows.
    uint64_t threatW = 0;
    uint64_t threatB = 0;
};

// Full refresh into a state (root slots, king moves).
void refresh_acc_state(const Position& pos, AccState& st,
                       const int16_t* weights);

// Pre-move capture: everything apply_acc_move needs, read BEFORE do_move.
struct AccMove {
    Piece mover = NO_PIECE;
    Piece captured = NO_PIECE;
    Piece finalPiece = NO_PIECE;  // mover, or promotion result
    Piece rook = NO_PIECE;        // castle rook (else NO_PIECE)
    Square from = Square(0);
    Square to = Square(0);
    Square capSq = Square(0);     // == to, except en-passant victim square
    bool castle = false;
    Square rookFrom = Square(0);
    Square rookTo = Square(0);
    bool whiteKingMoved = false;
    bool blackKingMoved = false;
};
AccMove capture_acc_move(const Position& pre, Move m);

// Apply captured pre-move data to the POST-move position:
// child = parent + diffs. King moves refresh that perspective fully
// (indices shift with the king square); everything else is exact adds/subs.
void apply_acc_move(const Position& post, const AccMove& u,
                    const int16_t* weights,
                    const AccState& parent, AccState& child);

} // namespace owen2::nnue
