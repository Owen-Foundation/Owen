#include "marrow.h"
#include "see.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

namespace owen2::search {

double MarrowTree::ucb_score(const MarrowNode* parent, const MarrowNode* child, int parentVisits) const {
    double proven_bonus = (child->depth >= cfg_.classical_depth) ? cfg_.proven_bonus : 1.0;
    // Node-relative flags: a proven-LOSS child is mated for the opponent =
    // best possible for the parent; a proven-WIN child wins for the opponent.
    if(child->is_proven_loss) return 1e12 * proven_bonus;
    if(child->is_proven_win) return -1e12;
    if(child->visits==0){
        // Progressive bias: keep shallow-first but break ties by prior so good
        // moves like Nf3/Nc3 are tried before h4/f3 within the same depth.
        return 5e8 - child->depth*1e6 + child->prior * 1e5;
    }
    double q = child->q();
    double q_parent = -q;
    // depth-decayed exploration: slower decay past classical_depth for long-depth search
    double C_eff;
    if(child->depth < cfg_.classical_depth)
        C_eff = cfg_.C * std::pow(cfg_.C_decay, child->depth);
    else
        C_eff = cfg_.C * std::pow(cfg_.C_decay, cfg_.classical_depth) * std::pow(cfg_.long_depth_C_decay, child->depth - cfg_.classical_depth);
    double explore = C_eff * std::sqrt(std::log(double(parentVisits+1)) / double(child->visits));
    double prior_term = cfg_.policy_weight * child->prior * std::sqrt(double(parentVisits)) / (1+child->visits);
    // history heuristic — quiet moves that caused beta cuts get boost (scales to classical)
    double hist = cfg_.history_weight * std::tanh(child->history / 8192.0);
    // progressive widening penalty: beyond base, need more visits to be considered
    // at long depth, widen more (explore forks) but with proof bonus
    int widen = cfg_.prog_widen_base + (child->depth >= cfg_.classical_depth ? 2 : 0);
    double widen_pen = 0;
    if((int)parent->children.size() > widen && parent->visits < 800){
        // find child's rank
        int rank=0; for(auto &c: parent->children) if(c.get()==child) break; else rank++;
        if(rank >= widen) widen_pen = -0.35 * (rank - widen + 1) * (1.0 - double(parent->visits)/800.0);
    }
    // long-depth Q stabilizer: reduce noisy Q at deep plies
    double depth_stabilize = 1.0 + 0.02 * std::max(0, child->depth - cfg_.classical_depth);
    return q_parent / depth_stabilize + explore + prior_term + hist + widen_pen;
}

void MarrowTree::expand_node(MarrowNode* node, const Position& pos){
    if(node->expanded) return;
    if(expMoves_.size() < (size_t)kMaxMoves) expMoves_.resize(kMaxMoves);
    int nmoves = generate_legal_buf(expScratch_, pos, expMoves_.data(), kMaxMoves);
    if(nmoves==0){
        node->is_terminal=true;
        node->expanded=true;
        if(pos.in_check()) node->terminal_value = mated_in(pos.ply());
        else node->terminal_value = VALUE_DRAW;
        node->is_proven_loss = pos.in_check();
        return;
    }
    // TT move is only used as a tie-breaker/ordering hint, NOT a +10000
    // domination that collapses priors to delta after reuse. We cap the TT
    // bonus to a modest value so multipv-style visit rebalancing still works
    // and so a stale TT move cannot permanently pin one child at prior ~1.0
    // (that was the "root b1c3 -> g1f3 flip then stuck" bug).
    bool hit; Move ttMove = tt_.probe_move(pos.key(), hit);

    struct Scored { Move m; int score; double hist; int pst; };
    // Tiny hand-crafted root bias so f3/f6 never leads when everything
    // else is equal (unloaded/untrained net or flat eval line). Keyed to
    // startpos geometry; deeper positions are dominated by eval/history.
    auto pst_move_bonus = [&](Move m)->int{
        Square from = move_from(m), to = move_to(m);
        // penalize f-pawn pushes to f3/f4/f6/f5 at the root; reward knights/centre pawns
        Color us = pos.side_to_move();
        if(type_of(pos.piece_on(from))==PAWN && file_of(from)==5){
            // f-file pawn move is terrible in the opening
            if((us==WHITE && rank_of(to)==2) || (us==BLACK && rank_of(to)==5)) return -95; // f3/f6
            if((us==WHITE && rank_of(to)==3) || (us==BLACK && rank_of(to)==4)) return -70; // f4/f5
        }
        if(type_of(pos.piece_on(from))==KNIGHT){
            int tf=file_of(to), tr=rank_of(to);
            if(us==WHITE && tf==5&&tr==2) return 52; // g1f3 — world-class fave
            if(us==WHITE && tf==2&&tr==2) return 36; // b1c3
            if(us==BLACK && tf==5&&tr==5) return 52; // g8f6
            if(us==BLACK && tf==2&&tr==5) return 36; // b8c6
        }
        if(type_of(pos.piece_on(from))==PAWN){
            // centre pawns e4/d4/c4
            int tf=file_of(to), tr=rank_of(to);
            if(us==WHITE && tr==3 && (tf==4||tf==3)) return 28; // e4,d4
            if(us==BLACK && tr==4 && (tf==4||tf==3)) return 28;
            if(us==WHITE && tr==2 && (tf==4||tf==3)) return 18; // e3/d3 second best
            if(us==BLACK && tr==5 && (tf==4||tf==3)) return 18;
        }
        // bishops to good squares (c4/f4 etc) slight bonus
        if(type_of(pos.piece_on(from))==BISHOP){
            int tf=file_of(to), tr=rank_of(to);
            if((us==WHITE && tr>=2 && tr<=3) || (us==BLACK && tr>=4 && tr<=5)){
                if(tf>=2 && tf<=5) return 10;
            }
        }
        return 0;
    };
    // Stack-resident scoring (no heap): moves already in expMoves_[0..nmoves).
    Scored scored[kMaxMoves];
    const int kslot = std::min(node->depth, 95);
    for(int i=0;i<nmoves;++i){
        Move m = expMoves_[i];
        int s=0;
        if(m==ttMove) s+=220; // modest TT bias, not 10000
        if(is_capture(m) || is_promo(m)){
            // SEE-ordered captures: winning swaps first, losing swaps sink
            // below quiets so UCB doesn't burn visits on blunders.
            int sv = see(pos, m);
            if(sv >= 0) s += 1000 + std::min(900, sv / 2);
            else s += 300 + std::max(-500, sv / 2);
        } else {
            if(m == killers_[kslot][0]) s += 200;
            else if(m == killers_[kslot][1]) s += 150;
            int f = move_from(m), t = move_to(m);
            double h = (f<64 && t<64) ? history_[f][t] : 0;
            s += int(std::tanh(h/4096.0)*120);
        }
        int f = move_from(m), t = move_to(m);
        double h = (f<64 && t<64) ? history_[f][t] : 0;
        int pst = pst_move_bonus(m);
        s += pst;
        scored[i]={m,s,h,pst};
    }
    // Stable tie-break: equal scores keep legality order (g1f3 before f2f3 depends on movegen) but our pst
    // already separates f3; use stable_sort so deterministic.
    std::stable_sort(scored, scored+nmoves, [](auto& a, auto& b){return a.score>b.score;});
    double maxS = nmoves?scored[0].score:0;
    double sum=0; double ex[kMaxMoves];
    for(int i=0;i<nmoves;++i){ ex[i]=std::exp((scored[i].score-maxS)/400.0); sum+=ex[i]; }
    // Learned policy head (Lc0-style PUCT priors, own code): blend the
    // handcrafted softmax with the net's policy when a v3 net is loaded.
    // Dirichlet noise at the root keeps self-play diverse (training only).
    const auto& net = nnue::g_network;
    bool usePol = cfg_.policy_blend > 0 && net.loaded && net.has_policy;
    double cap = usePol ? 0.90 : 0.30;
    // Clamp any single prior so one child can't dominate UCB exploration
    // forever after a lucky TT hit. This lets Marrow actually search.
    for(int i=0;i<nmoves;++i) ex[i] = std::min(ex[i] / sum, cap);
    // Renormalize after clamp (rare, but keeps sum==1)
    { double s2=0; for(int i=0;i<nmoves;++i) s2+=ex[i]; if(s2>1e-9) for(int i=0;i<nmoves;++i) ex[i]/=s2; sum=1.0; }
    if(usePol){
        nnue::Accumulator acc{};
        refresh_accumulator(pos, acc, net.feature_weights.data());
        const auto& avec = (pos.side_to_move()==WHITE) ? acc.white : acc.black;
        std::array<int16_t, nnue::Network::L2> l2{};
        net.hidden_l2(avec, l2);
        double mx = -1e30; double lg[kMaxMoves];
        for(int i=0;i<nmoves;++i){
            lg[i] = net.policy_logit(l2, nnue::Network::policy_index(scored[i].m));
            if(lg[i] > mx) mx = lg[i];
        }
        double s = 0; double pp[kMaxMoves];
        for(int i=0;i<nmoves;++i){ pp[i]=std::exp(lg[i]-mx); s+=pp[i]; }
        if(s > 1e-9) for(int i=0;i<nmoves;++i) pp[i]/=s;
        if(node->depth == 0 && cfg_.dirichlet_eps > 0){
            thread_local std::mt19937 drng(0xD17C);
            std::gamma_distribution<double> gam(cfg_.dirichlet_alpha, 1.0);
            double gs = 0; double nu[kMaxMoves];
            for(int i=0;i<nmoves;++i){ nu[i]=gam(drng); gs+=nu[i]; }
            if(gs > 1e-9) for(int i=0;i<nmoves;++i){
                pp[i] = (1.0 - cfg_.dirichlet_eps) * pp[i] + cfg_.dirichlet_eps * (nu[i] / gs);
            }
        }
        for(int i=0;i<nmoves;++i) ex[i] = (1.0 - cfg_.policy_blend) * ex[i] + cfg_.policy_blend * pp[i];
    }
    node->children.reserve(nmoves);
    for(int i=0;i<nmoves;++i){
        auto child = std::make_unique<MarrowNode>();
        child->move = scored[i].m;
        child->prior = ex[i];
        child->history = scored[i].hist;
        child->depth = node->depth + 1;
        node->children.push_back(std::move(child));
    }
    node->expanded=true;
}

void MarrowTree::select_path(Position& pos, std::vector<MarrowNode*>& path){
    MarrowNode* cur = root_.get();
    path.push_back(cur);
    while(cur->expanded && !cur->is_terminal && !cur->children.empty()){
        // Proof-guided jump (node-relative): a proven-LOSS child is mated
        // for the opponent = immediate win. Take the SHORTEST mate.
        int provenIdx=-1; int provenDepth=1000000;
        for(size_t i=0;i<cur->children.size();++i){
            const auto& ch = cur->children[i];
            if(ch->is_proven_loss && ch->proven_depth < provenDepth){
                provenDepth = ch->proven_depth; provenIdx=(int)i;
            }
        }
        int bestIdx = provenIdx;
        double bestScore = -1e100;
        if(bestIdx<0){
            for(size_t i=0;i<cur->children.size();++i){
                if(cur->children[i]->is_proven_win) continue; // opponent wins there
                double s = ucb_score(cur, cur->children[i].get(), cur->visits);
                if(s > bestScore){ bestScore=s; bestIdx=(int)i; }
            }
        }
        if(bestIdx<0){
            // all children proven wins for the opponent — delay mate maximally
            int maxD=-1;
            for(size_t i=0;i<cur->children.size();++i){
                const auto& ch = cur->children[i];
                if(ch->proven_depth > maxD ||
                   (ch->proven_depth == maxD && bestIdx >= 0 &&
                    ucb_score(cur, ch.get(), cur->visits) > bestScore)){
                    maxD = ch->proven_depth; bestIdx=(int)i;
                    bestScore = ucb_score(cur, ch.get(), cur->visits);
                }
            }
        }
        if(bestIdx<0) break;
        MarrowNode* nxt = cur->children[bestIdx].get();
        pos.do_move(nxt->move);
        path.push_back(nxt);
        cur = nxt;
        if(!cur->expanded) break;
        if(cur->visits==0) break;
        // LMR-style early stop: quiet deep nodes need more visits before deepening (scale to classical)
        if(cur->depth >= 8 && !is_capture(cur->move) && !is_promo(cur->move) && cur->visits < 3) break;
    }
}

Value MarrowTree::quiescence(Position& pos, Value alpha, Value beta, int depth){
    // Capture-only negamax with stand-pat. Fixes the no-quiescence horizon
    // blindness: hanging pieces / recaptures are resolved before the NNUE
    // value is backed up. Check evasions search all moves.
    // Scratch positions are per-level (qscratch_) so nested legality filters
    // never clobber each other; qdepth 6..0 maps to slot 6-depth.
    if(pos.is_draw()) return VALUE_DRAW;
    int stand = eval_cached(pos);
    if(depth <= 0){
        // Even at depth 0, return stand-pat bounded — caller handles terminal.
        return Value(std::max<int>(alpha, std::min<int>(beta, stand)));
    }
    const int qslot = std::min(7, std::max(0, 6 - depth));
    if(pos.in_check()){
        // Evasion: full-width but shallow (captures-only gen would miss
        // quiet escapes, so full legality here).
        Move legal[kMaxMoves];
        int nlegal = generate_legal_buf(qscratch_[qslot], pos, legal, kMaxMoves);
        if(nlegal==0) return mated_in(pos.ply());
        Value best = Value(-VALUE_INFINITE);
        for(int i=0;i<nlegal;++i){
            Move m = legal[i];
            pos.do_move(m);
            Value v = Value(-quiescence(pos, Value(-beta), Value(-alpha), depth - 1));
            pos.undo_move(m);
            if(v > best) best = v;
            if(best > alpha) alpha = best;
            if(alpha >= beta) break;
        }
        return best;
    }
    if(stand >= beta) return Value(stand);
    if(stand > alpha) alpha = Value(stand);
    // Non-evasion: captures/promotions only. Empty set means stand pat —
    // (quiet) stalemate is resolved at full-width levels, not here.
    Move clist[kMaxMoves];
    int nlegal = generate_captures_buf(qscratch_[qslot], pos, clist, kMaxMoves);
    if(nlegal==0) return Value(stand);
    // Order captures by SEE (exact swap, not MVV-LVA); delta-prune captures
    // that cannot reach alpha even in the best case. Stack-resident (no heap).
    static const int pval[6] = {100,320,330,500,900,800};
    struct CM { Move m; int see; int vmax; };
    CM caps[kMaxMoves];
    int ncaps = 0;
    for(int i=0;i<nlegal;++i){
        Move m = clist[i];
        if(!is_capture(m) && !is_promo(m)) continue;
        int sv = see(pos, m);
        Piece victim = pos.piece_on(move_to(m));
        int vv = (victim == NO_PIECE) ? 0 : pval[type_of(victim)];
        if(is_promo(m)) vv += 700; // near-queen value, promotion upside
        caps[ncaps++] = {m, sv, vv};
    }
    std::sort(caps, caps+ncaps, [](auto& a, auto& b){ return a.see > b.see; });
    static const int DELTA_MARGIN = 200;
    Value best = Value(stand);
    for(int i=0;i<ncaps;++i){
        auto& cm = caps[i];
        // Delta pruning: stand-pat + best-case swing + margin can't hit alpha.
        if(stand + cm.vmax + DELTA_MARGIN < (int)alpha) continue;
        pos.do_move(cm.m);
        Value v = Value(-quiescence(pos, Value(-beta), Value(-alpha), depth - 1));
        pos.undo_move(cm.m);
        if(v > best) best = v;
        if(best > alpha) alpha = best;
        if(alpha >= beta) break;
    }
    // Checks in quiet positions: after captures, search quiet moves that give
    // check (discovered attacks, skewers, pins). Only 1 ply ahead to stay cheap.
    if(depth > 1){
        Move qlist[kMaxMoves];
        int nq = generate_legal_buf(qscratch_[qslot], pos, qlist, kMaxMoves);
        for(int i=0;i<nq;++i){
            Move m = qlist[i];
            if(is_capture(m) || is_promo(m)) continue;
            pos.do_move(m);
            if(!pos.in_check()){
                pos.undo_move(m);
                continue;
            }
            Value v = Value(-quiescence(pos, Value(-beta), Value(-alpha), depth - 2));
            pos.undo_move(m);
            if(v > best) best = v;
            if(best > alpha) alpha = best;
            if(alpha >= beta) break;
        }
    }
    return best;
}

int MarrowTree::eval_cached(const Position& pos) {
    int v = 0;
    if (evalCache_.probe(pos.key(), v)) return v;
    v = nnue::g_network.evaluate(pos);
    evalCache_.store(pos.key(), v);
    return v;
}

Value MarrowTree::evaluate_leaf(Position& pos){
    if(pos.is_draw()) return VALUE_DRAW;
    // Terminal check on the caller's working position (no copy: quiescence
    // below mutates symmetrically via do/undo, net-zero on return).
    Move term[kMaxMoves];
    int nterm = generate_legal_buf(evalScratch_, pos, term, kMaxMoves);
    if(nterm==0){
        if(pos.in_check()) return mated_in(pos.ply());
        return VALUE_DRAW;
    }
    Value v = quiescence(pos, Value(-VALUE_INFINITE), Value(VALUE_INFINITE), 6);
    if(v > 15000) v = 15000;
    if(v < -15000) v = -15000;
    return v;
}

void MarrowTree::backup(std::vector<MarrowNode*>& path, Value leafValue){
    if((int)path.size() - 1 > maxDepth_) maxDepth_ = (int)path.size() - 1;
    // Leaf evaluation propagates bottom-up (negamax) plus AND/OR proof status.
    // NOTE on signs: cur at node i is node-i-relative. The move leading to
    // node i was played by node i's OPPONENT, so a move is good for its mover
    // iff cur is BAD for node i (cur < -threshold). The old code rewarded
    // history on cur > +200, i.e. it trained the mover's critics — inverted.
    Value cur = leafValue;
    for(int i=(int)path.size()-1; i>=0; --i){
        MarrowNode* n = path[i];
        n->visits++;
        n->total_value += double(cur);
        // AND/OR proof recompute (node-relative, exact):
        //   WIN  if ANY child is proven LOSS (opponent mated) — min distance;
        //   LOSS if ALL children proven WIN (opponent wins all) — max distance.
        // Terminal nodes have no children: keep expansion-set flags (mate = 0).
        if(!n->children.empty()){
            bool anyLoss = false; int minWin = 1000000;
            bool allWin = true; int maxLoss = -1;
            for(auto& ch : n->children){
                if(ch->is_proven_loss){ anyLoss = true; minWin = std::min(minWin, ch->proven_depth); }
                else allWin = false;
                if(ch->is_proven_win) maxLoss = std::max(maxLoss, ch->proven_depth);
            }
            if(anyLoss){ n->is_proven_win = true; n->is_proven_loss = false; n->proven_depth = minWin + 1; }
            else if(allWin){ n->is_proven_loss = true; n->is_proven_win = false; n->proven_depth = maxLoss + 1; }
            else { n->is_proven_win = false; n->is_proven_loss = false; }
        } else if(n->is_proven_win || n->is_proven_loss){
            n->proven_depth = 0;
        }
        // history + killers: reward the MOVER, i.e. cur bad for node i.
        if(i>0 && cur < -200){
            Move m = path[i]->move;
            int f = move_from(m), t = move_to(m);
            if(f<64 && t<64 && !is_capture(m) && !is_promo(m)){
                // depth-weighted history — deeper good moves get more
                int bonus = 512 * (path[i]->depth + 1);
                history_[f][t] += bonus;
                // age decay to keep bounded
                if(history_[f][t] > 16000) for(int a=0;a<64;++a) for(int b=0;b<64;++b) history_[a][b]/=2;
                // killer slots per tree-ply
                int ks = std::min(path[i]->depth, 95);
                if(killers_[ks][0] != m){ killers_[ks][1] = killers_[ks][0]; killers_[ks][0] = m; }
            }
        }
        cur = Value(-cur);
    }
}

Move MarrowTree::search_until(const std::function<bool()>& stop,
                              std::function<void(int,Value,Move)> on_info)
{
    expand_node(root_.get(), rootPos_);
    if(root_->children.empty()) return 0;
    if(root_->children.size()==1) return root_->children[0]->move;

    int lastInfoVisits=0;
    // Working state hoisted out of the loop: assignment reuses heap capacity
    // (memcpy only, no malloc after warmup). Quiescence restores by undo.
    if(pathBuf_.capacity() < 96) pathBuf_.reserve(96);
    while(!stop()){
        if(totalVisits_ >= cfg_.max_nodes) break;
        workPos_ = rootPos_;
        pathBuf_.clear();
        select_path(workPos_, pathBuf_);
        MarrowNode* leaf = pathBuf_.back();
        Value v;
        if(leaf->is_terminal){
            v = leaf->terminal_value;
        } else {
            if(!leaf->expanded){
                expand_node(leaf, workPos_);
                if(leaf->is_terminal) v = leaf->terminal_value;
                else v = evaluate_leaf(workPos_);
            } else {
                v = evaluate_leaf(workPos_);
            }
        }
        backup(pathBuf_, v);
        totalVisits_++;

        // Throttle info callbacks: every 1024 visits (was 2048) and no extra stop() poll every 256
        if(on_info && totalVisits_ - lastInfoVisits >= 1024){
            lastInfoVisits = totalVisits_;
            MarrowNode* best=nullptr; int bestV=-1;
            for(auto &c: root_->children) if(c->visits > bestV){ bestV=c->visits; best=c.get(); }
            // Root-relative score: child's mean is opponent-relative, negate.
            if(best) on_info(totalVisits_, Value(-best->q()), best->move);
        }
    }
    MarrowNode* best=nullptr;
    for(auto &c: root_->children){
        // Node-relative: proven-LOSS child = opponent mated = our win.
        // Prefer any forced win (shortest first), avoid forced losses,
        // else most visits with Q tie-break.
        if(!best){ best=c.get(); continue; }
        bool cL=c->is_proven_loss, bL=best->is_proven_loss;
        bool cW=c->is_proven_win, bW=best->is_proven_win;
        if(cL && !bL){ best=c.get(); continue; }
        if(bL && !cL) continue;
        if(cL && bL){
            if(c->proven_depth < best->proven_depth) best=c.get();
            continue;
        }
        if(cW && !bW) continue;
        if(bW && !cW){ best=c.get(); continue; }
        if(c->visits > best->visits || (c->visits==best->visits && c->q() > best->q()))
            best=c.get();
    }
    if(best) tt_.store(rootPos_.key(), Value(best->q()), maxDepth_, 0, best->move);
    return best? best->move : 0;
}

} // namespace owen2::search
