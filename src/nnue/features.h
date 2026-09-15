#pragma once
#include "../types.h"
#include "../position.h"

namespace owen2::nnue {

// SFNNv10-class — Owen2 v2 to match Stockfish 18 Jan 31 2026
// HalfKP + Threat Inputs
// bucketed king: 4 files mirrored -> 4 buckets, but we keep full 64 for accuracy and bucket at runtime
// INPUT = HalfKP (40960) + Threat (40960) = 81920
// H = 1024 (was 256). AVX2/VNNI still fast via int16.
// Threat: for each piece square, add feature if that square is attacked by opponent.
// This mirrors SFNNv10 "Threat Inputs" idea: lets net see hanging pieces / tensions.

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

} // namespace owen2::nnue
