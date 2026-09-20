#!/usr/bin/env bash
# Owen 2 — learn from Stockfish: match -> positions -> distill -> train.
# Usage:
#   bash scripts/learn_from_sf.sh [pairs] [movetime] [epochs]
#   bash scripts/learn_from_sf.sh 16 0.5 3   # 32 games, demo-sized
# Full runs: larger pairs (64+), more epochs (20-80), then rematch.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAIRS=${1:-16}
MOVETIME=${2:-0.5}
EPOCHS=${3:-3}
SF_BIN=${SF_BIN:-$(command -v stockfish 2>/dev/null || echo /home/hemesh/Documents/stockfish/stockfish-linux-x86-64-universal)}
ROUND=${ROUND:-1}

NET_PREV="$ROOT/nets/o2-sf18-r$((ROUND-1)).o2nn"
[ -f "$NET_PREV" ] || NET_PREV="$ROOT/nets/o2-final.o2nn"
OUT_PGN="$ROOT/data/vs-sf18-r${ROUND}.pgn"
OUT_SDATA="$ROOT/data/vs-sf18-r${ROUND}.bin"
OUT_DIST="$ROOT/data/vs-sf18-r${ROUND}-distilled.bin"
OUT_NET="$ROOT/nets/o2-sf18-r${ROUND}.o2nn"

echo "=== round $ROUND: $NET_PREV vs $SF_BIN ($((PAIRS*2)) games) ==="
python3 "$ROOT/tools/run_match.py" \
  --white-cmd "$ROOT/build/owen2" --white-name owen --white-opt "Threads=1 NNUEFile=$NET_PREV" \
  --black-cmd "$SF_BIN" --black-name sf18 --black-opt "Threads=2 Hash=128" \
  --book "$ROOT/tools/book.epd" --games "$PAIRS" --movetime "$MOVETIME" \
  --concurrency 6 --out "$OUT_PGN"

echo "=== pgn -> sdata ==="
python3 "$ROOT/trainer/pgn_to_sdata.py" --pgn "$OUT_PGN" --out "$OUT_SDATA" \
  --max 100000 --threads 4 --sample-every 1 --min-ply 4 --elo-min 0 --chunk-games 100

echo "=== distill @ depth 12 ==="
python3 "$ROOT/trainer/distill.py" --sdata "$OUT_SDATA" --out "$OUT_DIST" \
  --stockfish "$SF_BIN" --depth 12 --threads 8

echo "=== train (match upweighted 8x against base pool) ==="
SDATA="$ROOT/data/sdata-final-distilled.bin"
for _ in $(seq 1 8); do SDATA="$SDATA,$OUT_DIST"; done
DEV="cpu"; python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && DEV="cuda"
python3 "$ROOT/trainer/train.py" --sdata "$SDATA" --out "$OUT_NET" \
  --device "$DEV" --epochs "$EPOCHS" --batch 512 --no-compile --workers 2

echo "=== validation: new net vs previous (8 pairs, must score >=45%) ==="
VAL_OUT="$ROOT/data/validate-r${ROUND}.pgn"
VAL_LOG=$(python3 "$ROOT/tools/run_match.py" \
  --white-cmd "$ROOT/build/owen2" --white-name new --white-opt "Threads=1 NNUEFile=$OUT_NET" \
  --black-cmd "$ROOT/build/owen2" --black-name prev --black-opt "Threads=1 NNUEFile=$NET_PREV" \
  --book "$ROOT/tools/book.epd" --games 8 --movetime 0.3 \
  --concurrency 6 --out "$VAL_OUT" 2>&1 | tail -3)
echo "$VAL_LOG"
SCORE=$(echo "$VAL_LOG" | grep -oE 'Score [0-9.]+%' | tail -1 | grep -oE '[0-9.]+' || echo 0)
echo "validation score: ${SCORE}%"
LOW=$(python3 -c "print(1 if float('$SCORE') < 45 else 0)")
if [ "$LOW" = "1" ]; then
  mkdir -p "$ROOT/nets/rejected"
  mv "$OUT_NET" "$ROOT/nets/rejected/"
  echo "REJECTED (<45%): net moved to nets/rejected/. $NET_PREV stays champion."
else
  echo "PROMOTED: $OUT_NET is the new champion. Rematch with ROUND=$((ROUND+1))."
fi
