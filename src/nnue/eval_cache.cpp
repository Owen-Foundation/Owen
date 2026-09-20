#include "eval_cache.h"

namespace owen2::nnue {

void EvalCache::resize(size_t entries_pow2) {
    size_t n = size_t(1) << entries_pow2;
    table_.assign(n, {});
    mask_ = n - 1;
}

void EvalCache::clear() {
    for (auto& e : table_) e = {};
    hits_.store(0);
    misses_.store(0);
}

bool EvalCache::probe(uint64_t key, int& value) const {
    if (table_.empty() || key == 0) return false;
    size_t idx = (key ^ (key >> 32)) & mask_;
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
    Entry& e = table_[idx];
    e.key = key;
    e.value = int16_t(value);
}

} // namespace owen2::nnue
