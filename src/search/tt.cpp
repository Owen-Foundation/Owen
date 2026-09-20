#include "tt.h"
#include <cstring>
#include <algorithm>

namespace owen2 {

void TranspositionTable::resize(size_t mb){
    size_t bytes = mb * 1024 * 1024;
    size_t n = bytes / sizeof(TTEntry);
    // power of two
    size_t p=1; while(p < n) p <<= 1;
    p >>= 1; if(p==0) p=1;
    table_.assign(p, {});
    mask_ = p-1;
}
void TranspositionTable::clear(){ for(auto &e: table_) e={}; age_=0; }

TTEntry* TranspositionTable::probe(uint64_t key, bool &hit){
    // Legacy API: returns pointer to thread-local snapshot so Lazy SMP
    // readers don't race with concurrent stores.
    if(table_.empty()){ hit=false; return nullptr; }
    static thread_local TTEntry snap;
    Move m = probe_move(key, hit);
    // probe_move already set hit; also fill snapshot for callers that read value/depth
    {
        size_t idx = (key ^ (key>>32)) & mask_;
        std::lock_guard<std::mutex> lk(mu_[shard_of(idx)]);
        snap = table_[idx];
    }
    (void)m;
    return &snap;
}
Move TranspositionTable::probe_move(uint64_t key, bool &hit){
    if(table_.empty()){ hit=false; return 0; }
    size_t idx = (key ^ (key>>32)) & mask_;
    std::lock_guard<std::mutex> lk(mu_[shard_of(idx)]);
    const TTEntry &e = table_[idx];
    hit = (e.key == key);
    return hit ? e.move : 0;
}
void TranspositionTable::store(uint64_t key, Value v, int depth, uint8_t flag, Move m){
    if(table_.empty()) return;
    size_t idx = (key ^ (key>>32)) & mask_;
    std::lock_guard<std::mutex> lk(mu_[shard_of(idx)]);
    TTEntry &e = table_[idx];
    // replacement: empty, newer generation, or deeper (with +2 slack) wins
    if(e.key==0 || depth+2 >= e.depth || e.age != age_){
        e.key=key; e.value=int16_t(v); e.depth=int16_t(depth); e.flag=flag; e.move=m; e.age=age_;
    }
}
size_t TranspositionTable::hashfull() const {
    if(table_.empty()) return 0;
    size_t cnt=0, sample= std::min<size_t>(1000, table_.size());
    for(size_t i=0;i<sample;++i) if(table_[i].key) cnt++;
    return cnt * 1000 / sample;
}

} // namespace owen2
