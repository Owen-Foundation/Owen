#pragma once
#include "../position.h"
#include "../movegen.h"
#include "../nnue/network.h"
#include "../nnue/eval_cache.h"
#include "tt.h"
#include <vector>
#include <memory>
#include <cmath>
#include <atomic>
#include <chrono>
#include <functional>

namespace owen2::search {

// ── Marrow Tree Search ──────────────────────────────────────────
// Best-first, UCB-guided tree search. Fixes alpha-beta pathologies:
//  - no null-move (zugzwang-safe)
//  - no static LMR tables (allocation by uncertainty)
//  - multi-PV aware: top-K children stay hot
//  - proof-urgency bias for forced lines

struct MarrowNode {
    Move move=0;                 // move that led here (0 for root)
    int visits=0;
    double total_value=0;        // sum of negamax values (from node's player view)
    double prior=0;              // policy prior (from move ordering / net if available)
    double history=0;            // history heuristic (classical scaling)
    int depth=0;                 // ply from root
    bool expanded=false;
    bool is_terminal=false;
    // Proof flags are NODE-relative: is_proven_loss on a child means the
    // child player is mated/done = a WIN for the parent. is_proven_win on a
    // child means the opponent wins from there = avoid it.
    bool is_proven_win=false;
    bool is_proven_loss=false;
    int proven_depth=1000000;    // plies from this node to the proven end (min for wins)
    Value terminal_value=0;
    // Negamax best-confirmed bound (node perspective): max over children of
    // -child.bound. hasBound=false until the first leaf eval back-propagates.
    // When cfg.best_first is on, backup/selection use THIS (not the visit
    // average) so the tree behaves like best-first minimax: visits collapse
    // onto the frontier that can still improve, giving depth instead of
    // breadth-wide sampling on a 450-visit budget.
    double bound=-1e30;
    bool hasBound=false;
    std::vector<std::unique_ptr<MarrowNode>> children;

    double q() const { return visits ? total_value / visits : 0.0; }
};

struct MarrowConfig {
    double C = 1.35;             // UCB exploration (decays with depth)
    double policy_weight = 0.15; // blend prior into selection
    double history_weight = 0.08;// history heuristic blend (classical)
    double C_decay = 0.97;       // C *= C_decay per ply (deeper = less explore)
    int prog_widen_base = 4;     // progressive widening: first 4 children at new node
    int multiPV = 1;
    int max_nodes = 8000000;     // safety cap
    // Long-depth extensions (beat SF at classical 40/15)
    double long_depth_C_decay = 0.995; // slower decay past depth 12
    int classical_depth = 12;    // depth where we switch to long-depth mode
    double proven_bonus = 2.0;   // boost proven lines at deep search
    // Learned policy head (Lc0-style PUCT priors, own code)
    double policy_blend = 0.0;   // 0 = handcrafted only; 0.7 typical with v3 net
    double dirichlet_eps = 0.0;  // root noise mix (self-play/training only)
    double dirichlet_alpha = 0.3;// noise concentration
    // Bounds-based alpha-beta flavored pruning (own design, no SF code):
    // fail-low prune — a child whose negamax bound can no longer beat the
    // node's best confirmed value is cut after enough visits (default OFF;
    // ON proved safe via A/B before shipping).
    bool bound_prune = false;
    int bound_prune_min_visits = 6;  // give a child at least this many visits
    int bound_prune_margin = 40;     // cp slack so pruning stays sound-ish
    // Best-first minimax backup: recompute node bound as exact max(-child.bound)
    // along the path. Sound (bound can decrease when a child improves) and used
    // by the optional bound pruner. SELECTING by the bound is NOT used — it is
    // an optimistic overestimate on partially-explored nodes (produced phantom
    // mates). Selection stays on visit-average UCB.
    bool best_first = false;     // kept for A/B; selection by bound proved wrong
    int best_first_min_visits = 3;   // trust a child's bound after this many visits
};

class MarrowTree {
public:
    MarrowTree(const Position& rootPos, TranspositionTable& tt,
               nnue::EvalCache& evalCache, const MarrowConfig& cfg)
        : rootPos_(rootPos), tt_(tt), evalCache_(evalCache), cfg_(cfg)
    {
        root_ = std::make_unique<MarrowNode>();
        root_->expanded=false;
        buildLogTab();
        buildTanhTab();
        buildCEff();
    }

    // Run search until stop() returns true or budget exhausted.
    // Returns best move by visit count.
    Move search_until(const std::function<bool()>& stop,
                      std::function<void(int visits, Value score, Move best)> on_info = nullptr);

    int total_visits() const { return totalVisits_; }
    int max_depth() const { return maxDepth_; }
    MarrowNode* root() { return root_.get(); }
    long long ec_hits() const { return ecHits_; }
    long long ec_misses() const { return ecMisses_; }

private:
    Position rootPos_;
    TranspositionTable& tt_;
    nnue::EvalCache& evalCache_;
    MarrowConfig cfg_;
    std::unique_ptr<MarrowNode> root_;
    int totalVisits_=0;
    int maxDepth_=0;
    long long ecHits_=0, ecMisses_=0;
    // Precomputed depth→C_eff table: kills pow() from ucb_score.
    static constexpr int kMaxDepth = 256;
    double cEff_[kMaxDepth]{};
    // fast log table for ucb_score: log(n+1) for n < 32769 (root visits grow past 2048).
    static constexpr int kLogN = 32769;
    static double logTab_[kLogN];
    static bool logTabReady_;
    static void buildLogTab();
    static inline double fastLog(int n){
        return (n >= 0 && n < kLogN) ? logTab_[n] : std::log(double(n + 1));
    }
    // exact tanh(h/8192) table: history is a bounded int (decay keeps |h|<16384).
    static constexpr int kHistMin = -16384, kHistMax = 16384;
    static constexpr int kHistN = kHistMax - kHistMin + 1;
    static double tanhTab_[kHistN];
    static bool tanhTabReady_;
    static void buildTanhTab();
    static inline double fastTanh(int h){
        if(h < kHistMin) h = kHistMin;
        if(h > kHistMax) h = kHistMax;
        return tanhTab_[h - kHistMin];
    }
    // history table [from 64][to 64] for classical scaling — simple but effective
    int history_[64][64]{};
    // killer moves per tree-ply: quiet moves that were good for their mover
    Move killers_[96][2]{};
    // Scratch state reused across iterations (heap capacity retained, no
    // per-node malloc): working position, legality-filter positions (one
    // per quiescence level + expand + leaf check), and move/path buffers.
    Position workPos_;
    Position expScratch_;
    Position qscratch_[8];
    Position evalScratch_;
    std::vector<MarrowNode*> pathBuf_;
    std::vector<Move> expMoves_;
private:

    // one iteration: select -> expand/evaluate -> backup
    // returns leaf value from leaf player's perspective
    Value iterate(Position& pos);

    struct SelectFrame { MarrowNode* node; int childIdx; };
    void select_path(Position& pos, std::vector<MarrowNode*>& path);

    void expand_node(MarrowNode* node, const Position& pos);
    double ucb_score(const MarrowNode* parent, const MarrowNode* child, int parentVisits,
                     double parentLog, double parentSqrt) const;
    void buildCEff();
    Value evaluate_leaf(Position& pos);
    Value quiescence(Position& pos, Value alpha, Value beta, int depth);
    // Raw NNUE stand-pat through the eval cache (never mates/terminals).
    int eval_cached(const Position& pos);
    void backup(std::vector<MarrowNode*>& path, Value leafValue);
};

} // namespace owen2::search
