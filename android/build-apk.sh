#!/usr/bin/env bash
# Build the Owen 2 Android engine APK (open-exchange provider for DroidFish).
# Usage: ./build-apk.sh
# Needs: Android SDK (build-tools, platform android-35, NDK r28+, cmake),
#        JDK 17+, python3 + pillow. Output: ../dist/owen2-engine.apk
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
SDK="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
NDK="$SDK/ndk/28.2.13676358"
BT="$SDK/build-tools/36.0.0"
AJAR="$SDK/platforms/android-35/android.jar"
NET="$REPO/nets/active.o2nn"
OUT="$REPO/dist/owen2-engine.apk"
KEYSTORE="${OWEN_ANDROID_KEYSTORE:-$HOME/.android/owen-debug.keystore}"

for t in "$BT/aapt" "$BT/d8" "$BT/apksigner" "$BT/zipalign" "$AJAR" "$NET"; do
  [ -e "$t" ] || { echo "missing: $t"; exit 1; }
done
command -v javac keytool python3 >/dev/null || { echo "need javac, keytool, python3"; exit 1; }

# 1. Cross-compile one baked engine per ABI.
declare -A ABIS=( [arm64-v8a]="" [armeabi-v7a]="-DANDROID_ARM_NEON=ON" [x86_64]="" )
rm -rf jniLibs; mkdir -p jniLibs
for abi in arm64-v8a armeabi-v7a x86_64; do
  echo "=== engine: $abi ==="
  cmake -S "$REPO" -B "build/$abi" -DCMAKE_BUILD_TYPE=Release \
    -DOWEN_NATIVE=OFF -DOWEN_EMBED_NET=1 -DOWEN_EMBED_NET_PATH="$NET" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$abi" -DANDROID_PLATFORM=android-24 \
    -DANDROID_STL=c++_static ${ABIS[$abi]} > /dev/null
  cmake --build "build/$abi" --target owen2 -j"$(nproc)"
  "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "build/$abi/owen2"
  mkdir -p "jniLibs/$abi"
  cp "build/$abi/owen2" "jniLibs/$abi/libowen.so"
done

# 2. Icons from the site logo.
python3 - <<'EOF'
from PIL import Image
import os
im = Image.open("../website/assets/owen.png")
for d, px in [("drawable-mdpi", 48), ("drawable-hdpi", 72),
              ("drawable-xhdpi", 96), ("drawable-xxhdpi", 144)]:
    os.makedirs(f"res/{d}", exist_ok=True)
    im.resize((px, px), Image.LANCZOS).save(f"res/{d}/ic_launcher.png")
print("icons done")
EOF

# 3. Package resources.
rm -rf gen classes out && mkdir -p gen classes out
"$BT/aapt" package -f -M AndroidManifest.xml -S res -I "$AJAR" -J gen \
  --min-sdk-version 24 --target-sdk-version 35 -F out/owen-unsigned.apk

# 4. Compile Java + dex.
# shellcheck disable=SC2046
javac -source 8 -target 8 -cp "$AJAR" -d classes $(find gen java -name "*.java") 2>/dev/null
"$BT/d8" --lib "$AJAR" --min-api 24 --output out $(find classes -name "*.class")
(cd out && "$BT/aapt" add -f owen-unsigned.apk classes.dex)

# 5. Native libs (stored uncompressed so the installer keeps them executable).
rm -rf stage && mkdir -p stage && cp -r jniLibs stage/lib
(cd stage && zip -0 -q -r ../out/owen-unsigned.apk lib)
rm -rf stage

# 6. Align + sign (throwaway debug key outside the repo).
"$BT/zipalign" -f 4 out/owen-unsigned.apk out/owen-aligned.apk
if [ ! -f "$KEYSTORE" ]; then
  keytool -genkeypair -keystore "$KEYSTORE" -alias owen -keyalg RSA \
    -keysize 2048 -validity 10950 -storepass owen1234 -keypass owen1234 \
    -dname "CN=Owen Foundation" > /dev/null 2>&1
fi
"$BT/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:owen1234 \
  --key-pass pass:owen1234 --out "$OUT" out/owen-aligned.apk
"$BT/apksigner" verify "$OUT"
ls -la "$OUT"
