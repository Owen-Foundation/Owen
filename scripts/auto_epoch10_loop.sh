#!/usr/bin/env bash
# Auto: wait epoch10 -> Ctrl+C train -> bake -> wget -> lichess_loop 80 depth10 H=512 3760 > SF
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/data/train-distilled.log"
NET="$ROOT/nets/o2-distilled.o2nn"
echo "Watching train for epoch 10..."
while true; do
  if grep -q "epoch  10" "$LOG" 2>/dev/null; then
    echo "Epoch 10 hit!"
    # kill train cuda
    pkill -f "trainer/train.py --sdata data/classical-10M-distilled" || true
    sleep 3
    echo "Baking 167M..."
    cmake -B "$ROOT/build-baked" -DOWEN_EMBED_NET=ON -DOWEN_EMBED_NET_PATH="$NET" > /dev/null
    cmake --build "$ROOT/build-baked" -j12 > /dev/null
    ls -lh "$ROOT/build-baked/owen2"
    echo "Wget 20G Lichess..."
    mkdir -p "$HOME/Videos"
    wget -c "https://database.lichess.org/standard/lichess_db_standard_rated_2024-07.pgn.zst" -O "$HOME/Videos/lichess2.pgn.zst" &
    # patch loop to depth10 tiny phone light beat SF
    sed -i 's/--depth 12/--depth 10/g' "$ROOT/scripts/lichess_loop.sh" 2>/dev/null || true
    echo "Starting loop 80 depth10 -> 76h to 3760 > SF..."
    nohup bash "$ROOT/scripts/lichess_loop.sh" 80 > "$ROOT/data/loop.log" 2>&1 &
    echo "Done: train stop at 10, baked, wget, loop started. tail -f data/loop.log"
    break
  fi
  sleep 10
done
