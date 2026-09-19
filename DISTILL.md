# Distilling Stockfish into Owen 2

This document explains how Owen 2 reuses Stockfish's *knowledge* without copying its *code or weights*.

## Idea

> **Knowledge distillation**: a strong teacher labels positions with scalar evaluations; a student network learns to predict those scalars. The student never sees the teacher's weights.

Same method as neural machine translation and DeepSeek-style model distillation — only chess-flavored.

## What Is Distilled (and What Is Not)

| | |
|---|---|
| **Teacher** | Stockfish 18 (AVX2 build, UCI) — any binary works |
| **Student** | Owen 2 NNUE (`HalfKP+Threat`, input 81920, hidden 1024/512) + MTS search — original architecture |
| **Labels** | `score cp` integers from `go depth D` (default 9–12) via UCI — one scalar per position |
| **Loss** | MSE against `cp`; target blends `result` (WDL) + teacher eval |
| **Not copied** | No Stockfish source, no `SFNN` weights, no `.nnue` file, no search code |

The labels are factual evaluations of public chess positions — the student learns numbers, not code.

## Pipeline

```
PGN (e.g. Lichess)  ──►  sdata (69 B records)  ──►  distilled sdata  ──►  .o2nn net
     pgn_to_sdata.py           distill.py (UCI, N workers)        train.py (PyTorch)
```

### 1. PGN → sdata

```bash
python3 trainer/pgn_to_sdata.py \
  --pgn lichess.pgn.zst \
  --out data/sdata.bin \
  --max 10000000 --threads 12 --min-ply 30 --elo-min 2000
```

Each `sdata` record is 71 bytes v2 (packed, legacy 69B still readable):

| field | size | meaning |
|---|---|---|
| `board[64]` | 64 B | `0..11` piece, `12` empty |
| `stm` | 1 B | side to move |
| `eval` | 2 B | placeholder `0` until distillation (`int16 cp`) |
| `result` | 1 B | `0` loss / `1` draw / `2` win (from STM view) |
| `ply` | 1 B | ply count |
| `castling` | 1 B | `K=1 Q=2 k=4 q=8` (v2; v1 files assume `0`) |
| `ep` | 1 B | en-passant square `0..63`, `64` none (v2; v1 files assume `64`) |

v1 (69B) records without castling/EP distill to `- -` FENs (biased labels — regenerate with v2 writers). v2 emits full `KQkq` + EP so teacher evals match the real position.

Filters: `elo-min` and `min-ply` skip low-quality / opening positions. Positions are sampled evenly.

### 2. Distill — annotate with Stockfish

```bash
python3 trainer/distill.py \
  --sdata data/sdata.bin \
  --out   data/sdata-distilled.bin \
  --stockfish ~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2 \
  --depth 12 --threads 8
```

`distill.py` spawns **N persistent UCI workers** (one `Stockfish` process each, `Hash 32 MB`, `Threads 1`):

```
for each position i:
  FEN = board_to_fen(board, stm, ply)
  send: position fen <FEN>  →  go depth <D>
  parse: last "score cp <N>" before "bestmove"  →  cp
  pack: struct "<h" cp into record bytes 65..66
```

`cp` is clamped to `[-15000, 15000]`; mate scores are clamped similarly. This is the **only** information taken from Stockfish — a single integer per position, the same number any UCI engine would report for that FEN.

Throughput is CPU-bound by Stockfish AVX2 (no GPU needed for the teacher).

### 3. Train — fit Owen's net to the labels

```bash
python3 trainer/train.py \
  --sdata data/sdata-distilled.bin \
  --out   nets/o2-distilled.o2nn \
  --device cuda --epochs 80 --batch 512
```

`train.py` (PyTorch):

* Input features: `HalfKP + Threat` (81920), recomputed from `board[64]` + king squares — must match `src/nnue/features.cpp`.
* Target: blend of teacher `cp` and game `result` (WDL), mapped through the same scaling as the engine's `evaluate()` so learned values are directly usable.
* Optimizer: AdamW, cosine schedule, gradient clipping; mixed precision on GPU.
* Export: `.o2nn` (Owen 2 NNUE format `4f32 4e 4e`, version 2, little-endian). Loaded by `src/nnue/network.cpp` or baked with `-DOWEN_EMBED_NET=ON`.

## Why This Is Not "Cloning"

* Stockfish remains a separate, unmodified binary — used only as a UCI subprocess.
* The `.o2nn` weights are freshly initialized and optimized by Owen's trainer; they do not contain Stockfish weights.
* The search is unrelated: Owen uses **Marrow Tree Search (MTS)** (UCB + proof-guided), not Stockfish's alpha-beta + null-move + LMR.
* Analogy: training a new model on the *outputs* of an existing model is standard distillation (Hinton et al., 2015). The resulting artifact is original.

## Reproducibility

* Swap the teacher: any UCI engine that reports `score cp` works — change `--stockfish` and `--depth`.
* Swap the data: any PGN source works — self-play (`owen2-sdata`), Lichess dumps, or custom sets.
* No Stockfish source or binary is vendored; the user provides their own local build at distill time.

## Legal

* Owen 2's code is **GPL-3.0-only**. See [LICENSE](LICENSE).
* Stockfish is GPL-3.0. Using it as a teacher via its public UCI protocol does not combine the programs; Owen's repository contains none of Stockfish's source or weights.

Further reading: `trainer/README.md`, `docs/mts.md`, `src/nnue/features.cpp`.
