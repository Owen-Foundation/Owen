#pragma once
#include "features.h"
#include "../position.h"
#include <string>
#include <vector>
#include <cstdint>

namespace owen2::nnue {

// Owen2 v2/v3 — SFNNv10-class .o2nn
// magic "O2NN" ver=2, H=1024 (value only) or ver=3 (+policy head)
// Layout: feature_weights [INPUT_SIZE*H] int16
//         feature_bias    [H] int16
//         l1 [H*L1] int8, L1b [L1] int16, l2 [L1*L2] int8, L2b [L2] int16,
//         out [L2] int8, out_bias int16
//         [v3 only] pol [NPOL*L2] int8, pol_bias [NPOL] int16
// v1 (256) cannot be loaded as v2 — trainer will convert if needed.

struct Network {
    static constexpr int H = HIDDEN_SIZE; // 1024
    static constexpr int L1 = 16, L2 = 32; // 1024->16->32->1 : SFNNv10 style bottleneck, faster on AVX2
    static constexpr int NPOL = 4352; // 4096 from-to + 256 promo ((promo-1)*64+to)
    std::vector<int16_t> feature_weights; // INPUT*H
    std::array<int16_t, H> feature_bias{};
    std::array<int8_t, H*L1> l1_weights{};
    std::array<int16_t, L1> l1_bias{};
    std::array<int8_t, L1*L2> l2_weights{};
    std::array<int16_t, L2> l2_bias{};
    std::array<int8_t, L2> out_weights{};
    int16_t out_bias=0;
    std::vector<int8_t> pol_weights;   // v3: NPOL*L2, empty if v2
    std::vector<int16_t> pol_bias;     // v3: NPOL
    bool loaded=false;
    bool has_policy=false;
    bool load(const std::string& path);
    bool load_from_memory(const unsigned char* data, size_t size);
    bool save(const std::string& path) const;
    int evaluate(const Position& pos) const;
    int evaluate(const Position& pos, Accumulator& acc) const;
    int evaluate_handcrafted(const Position& pos) const;
    // policy logit for one owen-form move (call per legal move, then softmax in search)
    static int policy_index(Move m);
    float policy_logit(const std::array<int16_t,L2>& l2, int polIdx) const;
    // fill l2 hidden layer (shared by value + policy paths)
    void hidden_l2(const std::array<int16_t,H>& acc, std::array<int16_t,L2>& l2) const;
    void hidden_l2_from_l1(const std::array<int16_t,L1>& l1, std::array<int16_t,L2>& l2) const;
private:
    int forward(const std::array<int16_t,H>& acc) const;
};

extern Network g_network;

} // namespace owen2::nnue
