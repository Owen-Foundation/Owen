#pragma once
#include "../position.h"
#include "../movegen.h"
#include "../nnue/network.h"
#include "../nnue/eval_cache.h"
#include "tt.h"
#include <vector>
#include <memory>
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
};

class MarrowTree {
public:
    MarrowTree(const Position& rootPos, TranspositionTable& tt,
               nnue::EvalCache& evalCache, const MarrowConfig& cfg)
        : rootPos_(rootPos), tt_(tt), evalCache_(evalCache), cfg_(cfg)
    {
        root_ = std::make_unique<MarrowNode>();
        root_->expanded=false;
    }

    // Run search until stop() returns true or budget exhausted.
    // Returns best move by visit count.
    Move search_until(const std::function<bool()>& stop,
                      std::function<void(int visits, Value score, Move best)> on_info = nullptr);

    int total_visits() const { return totalVisits_; }
    int max_depth() const { return maxDepth_; }
    MarrowNode* root() { return root_.get(); }

private:
    Position rootPos_;
    TranspositionTable& tt_;
    nnue::EvalCache& evalCache_;
    MarrowConfig cfg_;
    std::unique_ptr<MarrowNode> root_;
    int totalVisits_=0;
    int maxDepth_=0;
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

    // one iteration: select -> expand/evaluate -> backup
    // returns leaf value from leaf player's perspective
    Value iterate(Position& pos);

    struct SelectFrame { MarrowNode* node; int childIdx; };
    void select_path(Position& pos, std::vector<MarrowNode*>& path);

    void expand_node(MarrowNode* node, const Position& pos);
    double ucb_score(const MarrowNode* parent, const MarrowNode* child, int parentVisits) const;
    Value evaluate_leaf(Position& pos);
    Value quiescence(Position& pos, Value alpha, Value beta, int depth);
    // Raw NNUE stand-pat through the eval cache (never mates/terminals).
    int eval_cached(const Position& pos);
    void backup(std::vector<MarrowNode*>& path, Value leafValue);
};

} // namespace owen2::search
