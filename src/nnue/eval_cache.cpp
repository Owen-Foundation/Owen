#include "eval_cache.h"

namespace owen2::nnue {

void EvalCache::resize(size_t entries_pow2) {
    size_t n = size_t(1) << entries_pow2;
    table_.assign(n, {});
    mask_ = n - 1;
}

void EvalCache::clear() {
    for (int s = 0; s < kShards; ++s) {
        std::lock_guard<std::mutex> lk(mu_[s]);
        for (size_t i = s; i < table_.size(); i += kShards) table_[i] = {};
    }
    hits_.store(0);
    misses_.store(0);
}

bool EvalCache::probe(uint64_t key, int& value) const {
    if (table_.empty() || key == 0) return false;
    size_t idx = (key ^ (key >> 32)) & mask_;
    std::lock_guard<std::mutex> lk(mu_[idx % kShards]);
    const Entry& e = table_[idx];
    if (e.key == key) {
        value = e.value;
        hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void EvalCache::store(uint64_t key, int value) {
    if (table_.empty() || key == 0) return;
    if (value > 15000) value = 15000;
    if (value < -15000) value = -15000;
    size_t idx = (key ^ (key >> 32)) & mask_;
    std::lock_guard<std::mutex> lk(mu_[idx % kShards]);
    Entry& e = table_[idx];
    e.key = key;
    e.value = int16_t(value);
}

} // namespace owen2::nnue
