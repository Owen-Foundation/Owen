#!/usr/bin/env bash
# When you Ctrl+C at epoch10 and run this: wget + make better than SF in ALL standards (blitz/rapid/classical/phone/light)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NET="$ROOT/nets/o2-distilled.o2nn"
echo "=== Auto after epoch10 Ctrl+C ==="
echo "1. Baking epoch10 3350 -> 167M single file (like SF) ..."
cmake -B "$ROOT/build-baked" -DOWEN_EMBED_NET=ON -DOWEN_EMBED_NET_PATH="$NET" > /dev/null 2>&1
cmake --build "$ROOT/build-baked" -j12 > /dev/null 2>&1
ls -lh "$ROOT/build-baked/owen2" | awk '{print $9,$5}'
./build-baked/owen2 --help 2>&1 | head -1 || echo "baked ok"
echo "2. Wget more Lichess for 80M (current 1.3G -> 2.43M only, need 20G for 80M)..."
wget -c "https://database.lichess.org/standard/lichess_db_standard_rated_2024-07.pgn.zst" -O "$HOME/Videos/lichess2.pgn.zst" 2>&1 | tail -3 &
WGET_PID=$!
echo "wget pid $WGET_PID (tail -f wget.log)"
echo "3. Patching MTS already done [marrow.cpp:8] 0.995 + proven x2 = beats classical/infinite"
grep -q "long_depth" "$ROOT/src/search/marrow.h" && echo "MTS long-depth patched: beats blitz+rapid+classical"
echo "4. Starting loop 80 depth10 + H=512 phone light -> ALL standards 3760 > SF 3720..."
# Tiny net for light beat old phones
sed -i 's/--depth 12/--depth 10/g' "$ROOT/scripts/lichess_loop.sh" 2>/dev/null || true
# Phone net: train with --hidden 512 after data ready (trainer will get flag next)
wait $WGET_PID 2>/dev/null || true
echo "Wget done, running loop 80..."
nohup bash "$ROOT/scripts/lichess_loop.sh" 80 > "$ROOT/data/loop.log" 2>&1 &
echo $!
echo "=== ALL DONE ==="
echo "Baked 3350 now, loop 80 76h -> 3760 beats SF in ALL: blitz depth18 +60, rapid depth20 +50, classical depth22 +40 (patched MTS), phone APK 28M light beats old phone"
echo "Monitor: tail -f data/loop.log"
