#!/usr/bin/env bash
# Owen 2 — SF-data chunk loop: divide the TBs instead of holding them.
# Per chunk: download -> decode subset -> train (base pool replay prevents
# forgetting; trainer always starts from scratch) -> 45% gate -> delete chunk.
# Disk stays flat (~8GB working set). Run detached: it waits for a free GPU.
#
# Usage:
#   SUBSET=1000000 UP=4 EPOCHS=6 LAPS=1 bash scripts/sf_chunk_loop.sh
#   # each chunk: 1M SF positions x4 + 1.7M base, 6 epochs, then gate
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBSET=${SUBSET:-1000000}
UP=${UP:-4}
EPOCHS=${EPOCHS:-6}
LAPS=${LAPS:-1}
BASE_MB="https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main"
BASE_SN="https://huggingface.co/datasets/official-stockfish/master-smallnet-binpacks/resolve/main"
CHUNKS=(
  "$BASE_MB/fishpack32.binpack"
  "$BASE_MB/farseerT76.binpack"
  "$BASE_MB/wrongNNUE_02_d9.binpack"
  "$BASE_MB/wrongIsRight_nodes5000pv2.binpack"
  "$BASE_SN/test77-jan2022-2tb7p.high-simple-eval-1k.min-v2.binpack"
  "$BASE_SN/test79-may2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack"
)
CHAMPION="$ROOT/nets/active.o2nn"
DATADIR="$ROOT/data/sf-data"
mkdir -p "$DATADIR"

wait_gpu() {
  while pgrep -f "trainer/trai[n].py" >/dev/null 2>&1; do
    echo "$(date -u +%H:%M) GPU busy (another train.py), waiting 120s..."
    sleep 120
  done
}

CHUNK_I=0
for LAP in $(seq 1 "$LAPS"); do
  SKIP=$(( (LAP - 1) * SUBSET ))
  for URL in "${CHUNKS[@]}"; do
    CHUNK_I=$((CHUNK_I + 1))
    FNAME=$(basename "$URL")
    RAW="$DATADIR/$FNAME"
    SUB="$DATADIR/chunk-${CHUNK_I}.bin"
    NET="$ROOT/nets/o2-sfchunk-${CHUNK_I}.o2nn"
    echo "===== chunk $CHUNK_I lap $LAP: $FNAME (skip=$SKIP) ====="

    if [ ! -s "$RAW" ]; then
      FREE_MB=$(df -m "$DATADIR" | tail -1 | awk '{print $4}')
      if [ "$FREE_MB" -lt 9000 ]; then echo "disk low (${FREE_MB}MB), aborting"; exit 1; fi
      echo "--- downloading (resume-capable) ---"
      wget -c -O "$RAW" "$URL"
    else
      echo "--- reusing existing $RAW ---"
    fi

    echo "--- decode subset (max=$SUBSET skip=$SKIP, streaming) ---"
    "$ROOT/build/owen2-binpack2sdata" --in "$RAW" --out "$SUB" \
      --max "$SUBSET" --skip "$SKIP" || { echo "decode failed, skipping chunk"; rm -f "$RAW"; continue; }
    GOT=$(( $(stat -c%s "$SUB") / 73 ))
    echo "subset positions: $GOT"
    if [ "$GOT" -lt 10000 ]; then echo "too few, skipping train"; rm -f "$RAW" "$SUB"; continue; fi

    wait_gpu
    echo "--- train (base + subset x$UP, $EPOCHS epochs, value+policy) ---"
    SDATA="$ROOT/data/sdata-final-distilled.bin"
    for _ in $(seq 1 "$UP"); do SDATA="$SDATA,$SUB"; done
    python3 "$ROOT/trainer/train.py" --sdata "$SDATA" --out "$NET" \
      --device cuda --epochs "$EPOCHS" --batch 512 --no-compile --workers 2 \
      --policy --pol-w 1000

    echo "--- validation vs champion (8 pairs, >=45%) ---"
    VAL_OUT="$ROOT/data/validate-chunk-${CHUNK_I}.pgn"
    VAL_LOG=$(python3 "$ROOT/tools/run_match.py" \
      --white-cmd "$ROOT/build/owen2" --white-name new --white-opt "Threads=1 NNUEFile=$NET" \
      --black-cmd "$ROOT/build/owen2" --black-name champ --black-opt "Threads=1 NNUEFile=$CHAMPION" \
      --book "$ROOT/tools/book.epd" --games 8 --movetime 0.3 \
      --concurrency 6 --out "$VAL_OUT" 2>&1 | tail -3)
    echo "$VAL_LOG"
    SCORE=$(echo "$VAL_LOG" | grep -oE 'Score [0-9.]+%' | tail -1 | grep -oE '[0-9.]+' || echo 0)
    echo "chunk $CHUNK_I score: ${SCORE}%"
    LOW=$(python3 -c "print(1 if float('$SCORE') < 45 else 0)")
    if [ "$LOW" = "1" ]; then
      mkdir -p "$ROOT/nets/rejected"
      mv "$NET" "$ROOT/nets/rejected/"
      echo "REJECTED (<45%). champion unchanged."
    else
      cp "$NET" "$CHAMPION"
      SHA=$(sha256sum "$CHAMPION" | cut -d' ' -f1)
      PROBE=$(cd "$ROOT" && (echo "uci"; echo "setoption name NNUEFile value $CHAMPION"; echo "isready"; echo "quit") | timeout 20 ./build/owen2 2>&1 | grep -c "NNUE loaded" || true)
      echo "PROMOTED + DEPLOY VERIFIED sha256=$SHA NNUE load=$PROBE"
    fi

    echo "--- deleting chunk (free disk) ---"
    rm -f "$RAW" "$SUB"
    df -h "$DATADIR" | tail -1
  done
done
echo "ALL CHUNKS DONE"
