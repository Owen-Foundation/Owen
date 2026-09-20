#!/usr/bin/env bash
# Owen 2 Chunk Trainer - Road to 4600
# Splits massive generation into shards, trains iteratively, resumes on crash/sleep
# Usage: ./scripts/chunk_train.sh  [games_per_chunk] [total_chunks] [threads]
#        ./scripts/chunk_train.sh 200000 5 12  -> 5 chunks of 200k = 1M games = 50M pos
#        ./scripts/chunk_train.sh 1000000 260 12 -> 130B pos (260 chunks) - will take months
set -e

GAMES_PER_CHUNK=${1:-200000}
TOTAL_CHUNKS=${2:-5}
THREADS=${3:-12}
DEPTH=8
EPOCHS=80

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
NETS="$ROOT/nets"
LOG="$ROOT/data/chunk.log"
STATE="$ROOT/data/.chunk_state"

mkdir -p "$DATA" "$NETS"

# Resume state: last completed chunk
LAST=0
if [[ -f "$STATE" ]]; then LAST=$(cat "$STATE"); fi
START=$((LAST + 1))

echo "=============================================="
echo " Owen 2 - Chunk Training to 4600"
echo " Chunk size : $GAMES_PER_CHUNK games (~$((GAMES_PER_CHUNK * 50)) pos)"
echo " Total chunks: $TOTAL_CHUNKS  Total pos: ~$((GAMES_PER_CHUNK * 50 * TOTAL_CHUNKS))"
echo " Threads     : $THREADS  Depth: $DEPTH  Epochs: $EPOCHS"
echo " Resuming from chunk $START (last done: $LAST)"
echo " Log: $LOG"
echo " Ctrl+C to pause anytime - will resume"
echo "=============================================="

# Ensure build exists
if [[ ! -x "$ROOT/build/owen2-sdata" ]]; then
  echo "Building Owen2..."
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ROOT/build" -j
fi

# Install torch if missing
if ! python3 -c "import torch" 2>/dev/null; then
  echo "Installing PyTorch (CUDA)..."
  pip install torch numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -n 5
fi

for ((CHUNK=START; CHUNK<=TOTAL_CHUNKS; CHUNK++)); do
  echo ""
  echo "=============================="
  echo " CHUNK $CHUNK / $TOTAL_CHUNKS"
  echo "=============================="

  SDATA="$DATA/sdata-v${CHUNK}.bin"
  NET="$NETS/o2-v${CHUNK}.o2nn"
  PREV_NET=""
  if (( CHUNK > 1 )); then PREV_NET="$NETS/o2-v$((CHUNK-1)).o2nn"; fi

  # Skip if chunk already fully done
  if [[ -f "$NET" && -f "$SDATA" ]]; then
    echo "Chunk $CHUNK already complete (net exists), skipping"
    echo "$CHUNK" > "$STATE"
    continue
  fi

  # 1. Generate sdata for this chunk (skip if sdata exists)
  if [[ ! -f "$SDATA" ]]; then
    echo "[Chunk $CHUNK] Generating $GAMES_PER_CHUNK games with $THREADS threads..."
    # Use previous net for quality if available, else handcrafted (first chunk)
    GEN_ARGS=(--games "$GAMES_PER_CHUNK" --out "$SDATA" --threads "$THREADS")
    if [[ -n "$PREV_NET" && -f "$PREV_NET" ]]; then
      GEN_ARGS+=(--depth "$DEPTH" --net "$PREV_NET")
      echo " -> Using net $PREV_NET at depth $DEPTH (high quality)"
    else
      echo " -> Using handcrafted eval (first chunk, movetime 15ms)"
    fi
    set -x
    "$ROOT/build/owen2-sdata" "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG"
    set +x
    if [[ ! -f "$SDATA" ]]; then echo "FAILED to generate $SDATA"; exit 1; fi
    SIZE=$(stat -c%s "$SDATA" 2>/dev/null || stat -f%z "$SDATA" 2>/dev/null || echo "?")
    echo "[Chunk $CHUNK] sdata: $SIZE bytes ($(($SIZE / 69)) positions)"
  else
    echo "[Chunk $CHUNK] sdata already exists, skipping generation"
  fi

  # 2. Build combined sdata up to this chunk for training (cumulative)
  # Training is cumulative: v3 is trained on v1+v2+v3
  COMBINED="$DATA/sdata-combined-v${CHUNK}.bin"
  if [[ ! -f "$COMBINED" ]]; then
    echo "[Chunk $CHUNK] Building cumulative dataset..."
    rm -f "$COMBINED"
    for ((i=1; i<=CHUNK; i++)); do
      cat "$DATA/sdata-v${i}.bin" >> "$COMBINED"
    done
    CSIZE=$(stat -c%s "$COMBINED" 2>/dev/null || stat -f%z "$COMBINED" 2>/dev/null || echo "?")
    echo " Combined: $CSIZE bytes"
  fi

  # 3. Train
  if [[ ! -f "$NET" ]]; then
    echo "[Chunk $CHUNK] Training net v${CHUNK} on $(stat -c%s "$COMBINED" 2>/dev/null || echo ?) bytes cumulative..."
    # Pick device
    DEVICE="auto"
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then DEVICE="cuda"; fi
    set -x
    python3 "$ROOT/trainer/train.py" --sdata "$COMBINED" --out "$NET" --epochs "$EPOCHS" --device "$DEVICE" 2>&1 | tee -a "$LOG"
    set +x
  else
    echo "[Chunk $CHUNK] net already exists"
  fi

  echo "$CHUNK" > "$STATE"

  # 4. Report
  if [[ -f "$NET" ]]; then
    NSIZE=$(stat -c%s "$NET" 2>/dev/null || echo "?")
    echo ""
    echo ">>> CHUNK $CHUNK DONE <<<"
    echo " Net: $NET ($NSIZE bytes)"
    echo " Test it: ./build/owen2 -> setoption name NNUEFile value $NET -> go depth 12"
    echo " Combined positions so far: ~$(($(stat -c%s "$COMBINED" 2>/dev/null || echo 0)/69))"
  fi

  echo ""
  echo "--- Sleeping 5s before next chunk (Ctrl+C to pause) ---"
  sleep 5
done

echo ""
echo "=============================================="
echo " ALL $TOTAL_CHUNKS CHUNKS COMPLETE"
echo " Final net: $NETS/o2-v${TOTAL_CHUNKS}.o2nn"
echo " Matches vs Stockfish:"
echo "  cutechess-cli -engine cmd=./build/owen2 arg=setoption\\ name\\ NNUEFile\\ value\\ $NETS/o2-v${TOTAL_CHUNKS}.o2nn \\"
echo "                -engine cmd=stockfish -each tc=60+0.5 -games 100 -pgnout pgn/final.pgn"
echo "=============================================="
