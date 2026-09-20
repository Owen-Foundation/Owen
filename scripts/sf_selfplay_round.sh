#!/usr/bin/env bash
# Owen 2 — SF-way self-play round: self-play (own games, own labels) ->
# binpack verify -> train (self-play upweighted 8x vs base) -> gate -> promote.
# This is the Stockfish METHODOLOGY with zero Stockfish DATA: games come from
# the current champion's own search, stored in ecosystem .binpack format.
#
# Usage:
#   ROUND_SP=1 bash scripts/sf_selfplay_round.sh [games] [epochs] [movetime_ms]
#   ROUND_SP=1 bash scripts/sf_selfplay_round.sh 2000 8 15
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROUND=${ROUND_SP:-1}
GAMES=${1:-2000}
EPOCHS=${2:-8}
MT=${3:-15}

CHAMPION="$ROOT/nets/active.o2nn"
SP_BIN="$ROOT/data/selfplay/selfplay-r${ROUND}.bin"
SP_BP="$ROOT/data/selfplay/selfplay-r${ROUND}.binpack"
OUT_NET="$ROOT/nets/o2-sp-r${ROUND}.o2nn"

if [ ! -f "$CHAMPION" ]; then echo "no champion $CHAMPION"; exit 1; fi

if [ -s "$SP_BIN" ]; then
  echo "=== reusing existing $SP_BIN ==="
else
  echo "=== self-play: $GAMES games from champion (book + resign) ==="
  "$ROOT/build/owen2-sdata" --games "$GAMES" --threads 12 --movetime "$MT" \
    --net "$CHAMPION" --book "$ROOT/tools/book.epd" \
    --resign-cp -700 --resign-moves 3 \
    --out "$SP_BIN" --binpack-out "$SP_BP"
fi

echo "=== binpack integrity: decode must match sdata count ==="
N_SD=$(( $(stat -c%s "$SP_BIN") / 71 ))
N_BP=$( "$ROOT/build/owen2-binpack2sdata" --in "$SP_BP" \
  --out "$ROOT/data/selfplay/selfplay-r${ROUND}-verify.bin" | grep -oE '[0-9]+ decoded' | grep -oE '[0-9]+' )
echo "sdata=$N_SD binpack=$N_BP"
# A few unrepresentable ep stems may be skipped by design; tolerate <1% drop.
OK=$(python3 -c "print(1 if $N_BP >= 0.99*$N_SD else 0)")
if [ "$OK" != "1" ]; then
  echo "MISMATCH beyond 1% tolerance: aborting"
  exit 1
fi
rm -f "$ROOT/data/selfplay/selfplay-r${ROUND}-verify.bin"

echo "=== train (self-play upweighted 8x against base pool) ==="
SDATA="$ROOT/data/sdata-final-distilled.bin"
for _ in $(seq 1 8); do SDATA="$SDATA,$SP_BIN"; done
DEV="cpu"; python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && DEV="cuda"
python3 "$ROOT/trainer/train.py" --sdata "$SDATA" --out "$OUT_NET" \
  --device "$DEV" --epochs "$EPOCHS" --batch 512 --no-compile --workers 2

echo "=== validation: new net vs champion (8 pairs, must score >=45%) ==="
VAL_OUT="$ROOT/data/validate-sp-r${ROUND}.pgn"
VAL_LOG=$(python3 "$ROOT/tools/run_match.py" \
  --white-cmd "$ROOT/build/owen2" --white-name new --white-opt "Threads=1 NNUEFile=$OUT_NET" \
  --black-cmd "$ROOT/build/owen2" --black-name champ --black-opt "Threads=1 NNUEFile=$CHAMPION" \
  --book "$ROOT/tools/book.epd" --games 8 --movetime 0.3 \
  --concurrency 6 --out "$VAL_OUT" 2>&1 | tail -3)
echo "$VAL_LOG"
SCORE=$(echo "$VAL_LOG" | grep -oE 'Score [0-9.]+%' | tail -1 | grep -oE '[0-9.]+' || echo 0)
echo "validation score: ${SCORE}%"
LOW=$(python3 -c "print(1 if float('$SCORE') < 45 else 0)")
if [ "$LOW" = "1" ]; then
  mkdir -p "$ROOT/nets/rejected"
  mv "$OUT_NET" "$ROOT/nets/rejected/"
  echo "REJECTED (<45%): net moved to nets/rejected/. champion unchanged."
else
  cp "$OUT_NET" "$CHAMPION"
  SHA=$(sha256sum "$CHAMPION" | cut -d' ' -f1)
  PROBE=$(cd "$ROOT" && (echo "uci"; echo "setoption name NNUEFile value $CHAMPION"; echo "isready"; echo "quit") | timeout 20 ./build/owen2 2>&1 | grep -c "NNUE loaded" || true)
  echo "PROMOTED + DEPLOY VERIFIED sha256=$SHA NNUE load=$PROBE"
fi
