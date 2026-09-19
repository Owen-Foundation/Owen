#!/usr/bin/env bash
# Owen 2 — Auto Distill Loop to beat Stockfish 18 (DeepSeek style)
# Each chunk: generate -> distill via ~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2 -> train cumulative
# Usage: bash scripts/chunk_distill.sh [games_per_chunk] [total_chunks] [threads] [sf_depth] [sf_threads]
#   bash scripts/chunk_distill.sh 50000 100 12 12 8  -> 100 chunks x 50k = 5M games = 250M pos distilled
# Resume: just re-run same command, reads data/.chunk_state
set -e
GAMES_PER_CHUNK=${1:-50000}
TOTAL_CHUNKS=${2:-100}
THREADS=${3:-12}
SF_DEPTH=${4:-12}
SF_THREADS=${5:-8}
EPOCHS=80
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
NETS="$ROOT/nets"
LOG="$ROOT/data/chunk_distill.log"
STATE="$ROOT/data/.chunk_distill_state"
SF_BIN="$HOME/Videos/stockfish/stockfish-ubuntu-x86-64-avx2"
# fallback
if [[ ! -x "$SF_BIN" ]]; then SF_BIN="$ROOT/../stockfish/stockfish-ubuntu-x86-64-avx2"; fi
if [[ ! -x "$SF_BIN" ]]; then SF_BIN="/usr/games/stockfish"; fi

mkdir -p "$DATA" "$NETS"
LAST=0; if [[ -f "$STATE" ]]; then LAST=$(cat "$STATE"); fi
START=$((LAST+1))
echo "=============================================="
echo " Owen 2 — Auto Distill Loop (DeepSeek)"
echo " Chunk: $GAMES_PER_CHUNK games (~$((GAMES_PER_CHUNK*50)) pos)"
echo " Total: $TOTAL_CHUNKS chunks = $((GAMES_PER_CHUNK*TOTAL_CHUNKS)) games = $((GAMES_PER_CHUNK*50*TOTAL_CHUNKS)) pos distilled"
echo " Threads: $THREADS  SF: $SF_BIN depth $SF_DEPTH x$SF_THREADS  Epochs: $EPOCHS"
echo " Resuming from chunk $START (last $LAST)"
echo " Log: $LOG  State: $STATE"
echo "=============================================="
if [[ ! -x "$ROOT/build/owen2-sdata" ]]; then
  echo "Building Owen2..."
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release -DOWEN_AVX2=ON > /dev/null
  cmake --build "$ROOT/build" -j 6
fi
if [[ ! -x "$SF_BIN" ]]; then echo "ERROR: Stockfish not found at $SF_BIN"; exit 1; fi
echo " SF: $($SF_BIN --help 2>&1 | head -1)"; ls -lh "$SF_BIN" | awk '{print $9, $5}'
if ! python3 -c "import torch" 2>/dev/null; then pip install torch numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3; fi

for ((CHUNK=START; CHUNK<=TOTAL_CHUNKS; CHUNK++)); do
  echo ""
  echo "=============================="
  echo " CHUNK $CHUNK / $TOTAL_CHUNKS"
  echo "=============================="
  SDATA="$DATA/sdata-v${CHUNK}.bin"
  SDATA_D="$DATA/sdata-v${CHUNK}-distilled.bin"
  NET="$NETS/o2-v${CHUNK}.o2nn"
  PREV_NET=""; if (( CHUNK>1 )); then PREV_NET="$NETS/o2-v$((CHUNK-1)).o2nn"; fi
  COMBINED_D="$DATA/sdata-combined-distilled-v${CHUNK}.bin"

  if [[ -f "$NET" && -f "$SDATA_D" ]]; then echo "Chunk $CHUNK done, skipping"; echo "$CHUNK" > "$STATE"; continue; fi

  # 1. Generate
  if [[ ! -f "$SDATA" ]]; then
    echo "[Chunk $CHUNK] Generating $GAMES_PER_CHUNK games..."
    GEN_ARGS=(--games "$GAMES_PER_CHUNK" --out "$SDATA" --threads "$THREADS")
    if [[ -n "$PREV_NET" && -f "$PREV_NET" ]]; then
      GEN_ARGS+=(--depth 8 --net "$PREV_NET"); echo " -> depth 8 + $PREV_NET"
    else
      echo " -> handcrafted (first chunk, or v1 not yet built — will distill anyway)"
      # allow explicit handcrafted for chunk1
    fi
    "$ROOT/build/owen2-sdata" "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG"
    # handle part files if killed with -9 earlier
    if [[ ! -f "$SDATA" && -f "${SDATA}.part0" ]]; then echo "Merging parts..."; cat "${SDATA}.part"* > "$SDATA"; fi
    if [[ ! -f "$SDATA" ]]; then echo "FAILED $SDATA"; exit 1; fi
    echo "[Chunk $CHUNK] sdata $SDATA $(stat -c%s "$SDATA" 2>/dev/null) bytes ($(($(stat -c%s "$SDATA" 2>/dev/null)/69)) pos)"
  else
    echo "[Chunk $CHUNK] sdata exists"
  fi

  # 2. Distill via SF18 AVX2 (skip if already distilled)
  if [[ ! -f "$SDATA_D" ]]; then
    echo "[Chunk $CHUNK] Distilling via $SF_BIN depth $SF_DEPTH x$SF_THREADS..."
    python3 "$ROOT/trainer/distill.py" --sdata "$SDATA" --out "$SDATA_D" --stockfish "$SF_BIN" --depth "$SF_DEPTH" --threads "$SF_THREADS" 2>&1 | tee -a "$LOG"
    if [[ ! -f "$SDATA_D" ]]; then echo "FAILED distill $SDATA_D"; exit 1; fi
  else
    echo "[Chunk $CHUNK] distilled exists"
  fi

  # 3. Build cumulative distilled dataset v1..CHUNK
  if [[ ! -f "$COMBINED_D" ]]; then
    echo "[Chunk $CHUNK] Building cumulative distilled..."
    rm -f "$COMBINED_D"
    for ((i=1;i<=CHUNK;i++)); do cat "$DATA/sdata-v${i}-distilled.bin" >> "$COMBINED_D"; done
    echo " Combined distilled: $(stat -c%s "$COMBINED_D" 2>/dev/null) bytes ($(($(stat -c%s "$COMBINED_D" 2>/dev/null)/69)) pos)"
  fi

  # 4. Train cumulative on RTX 2050 (DeepSeek tricks: mmap FP16 512x8)
  if [[ ! -f "$NET" ]]; then
    echo "[Chunk $CHUNK] Training $NET on cumulative distilled..."
    DEVICE="auto"; if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then DEVICE="cuda"; fi
    python3 "$ROOT/trainer/train.py" --sdata "$COMBINED_D" --out "$NET" --epochs "$EPOCHS" --device "$DEVICE" 2>&1 | tee -a "$LOG"
  fi

  echo "$CHUNK" > "$STATE"
  echo ">>> CHUNK $CHUNK DONE <<< Net: $NET  Combined: $(($(stat -c%s "$COMBINED_D" 2>/dev/null)/69)) pos"
  echo "--- sleep 5s (Ctrl+C to pause, re-run to resume) ---"; sleep 5
done
echo ""
echo "ALL $TOTAL_CHUNKS CHUNKS COMPLETE Final: $NETS/o2-v${TOTAL_CHUNKS}.o2nn"
echo "Test: ./build/owen2 -> setoption name NNUEFile value $NETS/o2-v${TOTAL_CHUNKS}.o2nn -> go depth 12"
