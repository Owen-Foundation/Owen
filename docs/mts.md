# Marrow Tree Search (MTS) — Design Note

MTS is a best-first, UCB-guided search with NNUE leaves. Goal: fix alpha-beta pathologies without cloning Stockfish.

## Intuition

Alpha-beta asks: *is this move good enough to refute the parent?* (cutoff-driven). Marrow asks: *where am I most uncertain about the best move?* (uncertainty-driven). The tree is expanded where UCB says the answer matters most.

## Algorithm (one iteration)

1. **Select**: from root, repeatedly pick child maximizing

   ```
   score = -Q(child) + C * sqrt(log N / n) + W * prior * sqrt(N)/(1+n)
   ```

   `Q` is the child's mean value from its own perspective (negamax: parent sees `-Q`). Unvisited children get `+inf` — expanded eagerly.

2. **Expand**: generate all legal moves, score for priors (TT move, captures, promos → softmax temperature 400), create one child per move.

3. **Evaluate**: NNUE (HalfKP+Threat 81920→1024→16→32→1) from side-to-move view, plus capture-only quiescence (6 plies, check evasions full-width), or terminal mate/draw value. Stand-pat horizon is resolved by quiescence; deeper tactical horizon is handled by proof-urgency expansion.

4. **Backup**: negamax flip into node means, plus exact AND/OR proof
    propagation (node-relative): a node is a proven WIN if ANY child is a
    proven LOSS for the opponent (take the shortest mate), and a proven LOSS
    only if ALL children are proven wins (delay mate maximally). Selection
    jumps to the fastest forced win and never visits opponent-wins. History
    and killer updates reward the MOVER (negative node-relative value).

Move ordering in expansion is SEE-based (winning swaps first, losing swaps
sink below quiets), with killer-move and history bonuses for quiets.
Quiescence orders captures by SEE and delta-prunes hopeless swings.

Best move = most visits (robust to evaluation noise). TT stores best move.

## What it deliberately does NOT do

- No null-move pruning (zugzwang pathologies).
- No classic LMR/history reductions (replaced by UCB allocation).
- No alpha-beta window — values are backed up as expectations, not bounds.

## Time control

`search_until(stop)` runs iterations until `stop()` is true. `Searcher` builds `stop` from UCI limits (`movetime`, `wtime/btime+inc`, `depth`, `nodes`, `infinite`). For `go depth` we use a visit budget (`depth * 8000`) as a soft depth proxy — marrow has no fixed depth.

## MultiPV

Keep top-K root children hot: `cfg.multiPV` is reserved for future UCB bias that prevents starvation of 2nd/3rd best lines. Currently visits naturally spread, but an explicit bias is planned.

## Future work

- Incremental NNUE update along the path (vs full refresh; AVX2 refresh is done).
- Learned policy head for priors (requires policy data in sdata).
- ~~Lazy SMP: N threads each run a tree, share TT + best move via atomic.~~ Done (v2): N trees share mutex-guarded TT, vote by visits.
- Proof-number urgency term for forced mates.
