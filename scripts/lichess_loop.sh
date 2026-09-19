#!/usr/bin/env bash
# Owen 2 — Lichess loop to 60M distilled (target: approach SF18 ~3720 CCRL — unproven, measure with cutechess)
# Auto-downloads Lichess PGN zst, converts -> distills -> trains. Resume: just re-run.
# Usage: bash scripts/lichess_loop.sh         -> 60M (~25 chunks x 2.43M)
#        bash scripts/lichess_loop.sh 100     -> 100M / whatever you set
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"; NETS="$ROOT/nets"; LOG="$ROOT/data/lichess_loop.log"
STATE="$ROOT/data/.lichess_loop_state"
SF_BIN="$HOME/Videos/stockfish/stockfish-ubuntu-x86-64-avx2"
TARGET_M=${1:-60}  # million positions target
CHUNK_M=2  # ~2.43M per 1.3G file (we reuse same file + wget more for next chunks)
URLS=(
  "https://database.lichess.org/standard/lichess_db_standard_rated_2024-06.pgn.zst"
)
mkdir -p "$DATA" "$NETS"
LAST=0; [[ -f "$STATE" ]] && LAST=$(cat "$STATE")
START=$((LAST+1))
echo "=============================================="
echo " Owen 2 — Lichess Loop to ${TARGET_M}M"
echo " Start chunk $START (last $LAST) SF depth12 x8"
echo " Log $LOG State $STATE"
echo "=============================================="
if [[ ! -x "$SF_BIN" ]]; then echo "ERROR SF not at $SF_BIN"; exit 1; fi
if ! python3 -c "import torch" 2>/dev/null; then pip install torch numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3; fi

CHUNK=$START
while true; do
  HAVE=$(cat "$DATA"/lichess_loop_chunk_*.bin 2>/dev/null | wc -c 2>/dev/null | awk '{print int($1/69)}' 2>/dev/null); HAVE=${HAVE:-0}
  HAVE_M=$((HAVE/1000000))
  if (( HAVE_M >= TARGET_M )); then echo "TARGET ${TARGET_M}M reached ($HAVE)"; break; fi
  echo ""; echo "=============================="; echo " LICHESS CHUNK $CHUNK (have ${HAVE_M}M / ${TARGET_M}M)"; echo "=============================="
  # 1. Ensure PGN exists (wget more for chunk>1)
  PGN="$HOME/Videos/lichess.pgn.zst"
  if [[ ! -f "$PGN" || ! -s "$PGN" ]]; then echo "Downloading Lichess..."; wget -c -O "$PGN" "${URLS[0]}" 2>&1 | tail -5; fi
  # For chunk>1 we need more data: just re-use pgn_to_sdata offset? Simpler: each chunk re-samples with shuffle via --max + random.
  OUT="$DATA/lichess_loop_chunk_${CHUNK}.bin"
  OUT_D="$DATA/lichess_loop_chunk_${CHUNK}-distilled.bin"
  if [[ ! -f "$OUT" ]]; then
    echo "[Chunk $CHUNK] pgn_to_sdata ~2.4M..."
    python3 "$ROOT/trainer/pgn_to_sdata.py" --pgn "$PGN" --out "$OUT" --max 10000000 --threads 12 --min-ply 30 --elo-min 2000 2>&1 | tee -a "$LOG"
  fi
  if [[ ! -f "$OUT_D" ]]; then
    echo "[Chunk $CHUNK] distilling..."
    python3 "$ROOT/trainer/distill.py" --sdata "$OUT" --out "$OUT_D" --stockfish "$SF_BIN" --depth 10 --threads 8 2>&1 | tee -a "$LOG"
  fi
  # 3. Build cumulative
  COMBINED="$DATA/lichess_combined_distilled.bin"
  echo "[Chunk $CHUNK] building cumulative..."
  cat "$DATA"/lichess_loop_chunk_*-distilled.bin > "$COMBINED" 2>/dev/null || true
  CNS=$(stat -c%s "$COMBINED" 2>/dev/null || echo 0); CNP=$((CNS/69))
  echo " Combined $CNP pos"
  # 4. Train cumulative
  NET="$NETS/o2-lichess-${CHUNK}.o2nn"
  if [[ ! -f "$NET" ]]; then
    DEV="auto"; python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && DEV="cuda"
    echo "[Chunk $CHUNK] training $NET on $CNP pos..."
    python3 "$ROOT/trainer/train.py" --sdata "$COMBINED" --out "$NET" --epochs 80 --device "$DEV" 2>&1 | tee -a "$LOG"
  fi
  echo "$CHUNK" > "$STATE"
  echo ">>> LICHESS CHUNK $CHUNK DONE $CNP pos Net $NET"
  CHUNK=$((CHUNK+1))
  echo "--- sleep 5 (Ctrl+C to pause, re-run to resume) ---"; sleep 5
done
echo "ALL DONE $TARGET_M M Final $NETS/o2-lichess-${CHUNK}.o2nn"
