# Changelog — Owen 2

All notable changes to this project will be documented in this file.

## Unreleased

- Initial public development.

## v0.1.0 — 2026-09-01

- Initial project setup (GPL-3.0).

## v0.2.0 — 2026-09-04

- Bitboard attack tables + plain sliding attacks, position make/unmake, move generation with perft verification.
- Marrow Tree Search (MTS) — UCB selection with NNUE evaluation.
- UCI support (`Hash`, `Threads`, `NNUEFile`, `MTS_C`).
- Self-play data generation (`sdata`) and PyTorch trainer.

## Unreleased

- sdata v2 (71B): preserves castling rights + en-passant for unbiased Stockfish distillation (legacy 69B still readable).
- NNUE AVX2 paths (accumulator adds + 1024->16 affine), capture-only quiescence in MTS leaves, TT stores real depth + thread-safe probe for Lazy SMP.
- Lazy SMP (N trees share TT), functional `UCI_Elo`/`UCI_LimitStrength` limiter.
- MTS math batch: exact swap-list SEE (`src/search/see.cpp`) driving capture
  ordering + quiescence delta pruning; killer moves; fixed inverted proof
  flags (engine dodged mate-in-1) and inverted history sign; exact AND/OR
  proof   propagation with mate-distance preference (fastest mate wins, slowest
  loss resists); root-relative info/bestmove scores.
- Speed batch (profile-driven): position-keyed eval cache (38-45% hits,
  sharded for SMP); clean-room magic bitboards (threat loop 1.5us->0.3us,
  init ~140ms, exhaustively self-verified); allocation-free search
  (buffer movegen API, stack-resident scoring, hoisted working state —
  startpos NPS +27%, kiwipete +54%); UCI `bench` for repeatable numbers.
- Quiescence uses captures-only movegen (full legality kept for check
  evasions; stalemate stays a full-width verdict): kiwipete NPS +39%
  (+114% cumulative over the pre-batch baseline).
- Fixed SMP depth-budget oversearch (each worker ran the full budget; now
  split — 3.5x on 4 threads for equal work) and SMP seldepth reporting.
- Strength tooling: `tools/make_book.py` (PGN->EPD book), `tools/run_match.py`
  (UCI matches, adjudication, PGN, Elo), `scripts/learn_from_sf.sh`
  (match vs Stockfish -> distill -> mixed train loop); `train.py --sdata`
  accepts comma-separated mixed pools (repeat a file to upweight).
