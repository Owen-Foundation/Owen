#pragma once
#include "../types.h"
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace owen2::nnue {

// ── Position-keyed eval cache ──────────────────────────────────────
// Caches RAW NNUE stand-pat values (post-clamp). Never stores mate or
// terminal scores — callers only consult it for the static eval.
// Sound: evaluate() is a pure function of board+stm, both in the key.
// Sharded mutexes (64) keep Lazy-SMP workers from serializing on it.
class EvalCache {
public:
    explicit EvalCache(size_t entries_pow2 = 16) { resize(entries_pow2); }
    void resize(size_t entries_pow2);
    void clear();
    // Returns true on hit (value set). Misses leave value untouched.
    bool probe(uint64_t key, int& value) const;
    void store(uint64_t key, int value);
    uint64_t hits() const { return hits_.load(); }
    uint64_t misses() const { return misses_.load(); }
private:
    struct Entry { uint64_t key = 0; int16_t value = 0; };
    static constexpr int kShards = 64;
    std::vector<Entry> table_;
    size_t mask_ = 0;
    mutable std::mutex mu_[kShards];
    mutable std::atomic<uint64_t> hits_{0}, misses_{0};
};

} // namespace owen2::nnue
