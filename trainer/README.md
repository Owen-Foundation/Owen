# Trainer

## Requirements

```bash
pip install torch numpy
# GPU: install CUDA build of torch per pytorch.org
```

## Generate sdata

```bash
cmake --build build -j
./build/owen2-sdata --games 200000 --out data/sdata.bin
ls -lh data/sdata.bin
```

Record size is 71 bytes v2 (pragma pack 1): 64B board + stm + eval(i16) + result + ply + castling + ep. Legacy 69B files (no castling/EP) still train. See `sdata.py`.

## Lc0 data (no Stockfish teacher)

Leela publishes open self-play chunks (ODbL) — the same data family
Stockfish trains on. Convert straight to sdata v2 (result + search-Q labels,
no UCI teacher, no FEN round-trip):

```bash
# one tar, streamed (never fully extracted):
wget https://storage.lczero.org/files/training_data/training-run1--20230505-0917.tar
python3 trainer/lc0_to_sdata.py --in training-run1--20230505-0917.tar \
    --out data/lc0-1M.bin --max 1000000
python3 trainer/train.py --sdata data/lc0-1M.bin --out nets/o2-lc0.o2nn \
    --device cuda --epochs 80
```

Notes: supports V3/V4/V5/V6 chunks (`.tar`/`.tar.gz`/`.gz`/raw/dir);
eval = Lc0 `best_q` via Lc0's own cp curve; game result is STM-relative;
EP is absent in chunks (stored none — harmless, features ignore EP);
policy head is ignored (Owen has no policy net yet).

## Train

```bash
python trainer/train.py --sdata data/sdata.bin --out nets/o2-v1.o2nn --epochs 80 --gpu
# CPU fallback:
python trainer/train.py --sdata data/sdata.bin --out nets/o2-v1.o2nn --epochs 20 --device cpu
```

The trainer exports `.o2nn` (Owen 2 NNUE format). Load in engine:

```bash
./build/owen2
> setoption name NNUEFile value nets/o2-v1.o2nn
> isready
```

## Notes

- Loss blends game result (WDL) and search eval (distillation). Edit `SDataDataset.__getitem__` to change blend.
- `feature_indices` must match `src/nnue/features.cpp`.
- Use `--batch 8192` on large GPUs; lower to 1024 on CPU/MPS.
