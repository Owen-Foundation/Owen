#pragma once
#include "../types.h"
#include <vector>
#include <cstdint>
#include <mutex>

namespace owen2 {

struct TTEntry {
    uint64_t key=0;
    int16_t  value=0;
    int16_t  depth=0;
    uint8_t  flag=0; // 0 exact, 1 lower, 2 upper
    uint8_t  age=0;
    Move move=0;
};

class TranspositionTable {
public:
    void resize(size_t mb);
    void clear();
    void new_search() { ++age_; }
    TTEntry* probe(uint64_t key, bool &hit);
    Move probe_move(uint64_t key, bool &hit); // thread-safe: returns stored move under lock
    void store(uint64_t key, Value v, int depth, uint8_t flag, Move m);
    size_t hashfull() const; // per mille
private:
    std::vector<TTEntry> table_;
    size_t mask_=0;
    uint8_t age_=0;
    // Lazy SMP scaling: one lock for the whole table serializes 12 cores.
    // Split by bucket so probes/stores on different buckets never collide.
    static constexpr int kShards = 64;
    mutable std::mutex mu_[kShards];
    size_t shard_of(size_t idx) const { return (idx >> 6) & (kShards - 1); }
};

} // namespace owen2
