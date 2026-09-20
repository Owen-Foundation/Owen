#pragma once
#include "../position.h"
#include "tt.h"
#include "marrow.h"
#include <chrono>
#include <atomic>
#include <functional>
#include <algorithm>

namespace owen2::search {

struct SearchLimits {
    int depth = 64;
    int64_t movetime_ms = -1; // -1 = no limit
    int64_t wtime_ms = -1, btime_ms = -1;
    int64_t winc_ms = 0, binc_ms = 0;
    int movestogo = 0;
    bool infinite = false;
    bool ponder = false;
    int64_t nodes = -1;
    int mate = -1; // -1 = no mate search limit
    std::vector<std::string> searchmoves; // if non-empty, restrict search to these moves
};

struct SearchResult {
    Move bestMove=0;
    Move ponderMove=0;
    Value score=0;
    int depth=0;
    uint64_t nodes=0;
    int64_t time_ms=0;
};

class Searcher {
public:
    Searcher();
    void set_tt_size(int mb){
        tt_.resize(mb);
        // Size the eval cache with the TT (~1/4 of Hash bytes): 1M entries per
        // 16MB TT. The old fixed 2^16=64k table thrashed past ~100k QS evals.
        int p = 14;
        for(int m = std::max(1, mb); m > 1; m >>= 1) ++p;
        p = std::clamp(p, 16, 26);
        size_t entries_pow2 = size_t(p);
        evalCache_.resize(entries_pow2);
    }
    void set_threads(int n){ threads_=std::max(1,n); }
    void set_marrow_c(double c){ marrowCfg_.C = c; }
    void set_multipv(int n){ marrowCfg_.multiPV = std::max(1,n); }
    void set_move_overhead(int ms){ moveOverheadMs_ = std::max(0, ms); }
    void set_slow_mover(int v){ slowMover_ = std::clamp(v, 10, 1000); }
    void set_show_wdl(bool v){ showWDL_ = v; }
    void set_limit_strength(bool v){ limitStrength_ = v; }
    void set_uci_elo(int e){ uciElo_ = std::clamp(e, 1320, 4100); }
    void set_policy_blend(double b){ marrowCfg_.policy_blend = std::clamp(b, 0.0, 1.0); }
    void set_dirichlet(double eps, double alpha){ marrowCfg_.dirichlet_eps = eps; marrowCfg_.dirichlet_alpha = alpha; }
    void set_bound_prune(bool on){ marrowCfg_.bound_prune = on; }
    void set_best_first(bool on){ marrowCfg_.best_first = on; }
    int64_t elo_node_cap() const; // -1 = no cap
    void new_game(){ tt_.clear(); evalCache_.clear(); }
    void set_position(const Position& p){ pos_=p; }
    TranspositionTable& tt(){ return tt_; }

    // Blocking search — respects limits, polls stop flag.
    // In uci.cpp this is called from a dedicated search thread so isready/position/stop
    // remain responsive.
    SearchResult search(const SearchLimits& lim,
                        std::atomic<bool>& stop,
                        std::function<void(const std::string&)> info_cb = nullptr);

    void stop(){ stopFlag_.store(true); }
    nnue::EvalCache& eval_cache(){ return evalCache_; }
private:
    Position pos_;
    TranspositionTable tt_;
    nnue::EvalCache evalCache_;
    MarrowConfig marrowCfg_;
    int threads_=1;
    int moveOverheadMs_=10;
    int slowMover_=100;
    bool showWDL_=false;
    bool limitStrength_=false;
    int uciElo_=1320;
    std::atomic<bool> stopFlag_{false};
    int64_t time_budget_ms(const SearchLimits& lim) const;
};

} // namespace owen2::search
