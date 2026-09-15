# Contributing to Owen 2

Owen 2 is **GPL-3.0** by [Owen Foundation](https://github.com/Owen-Foundation).

## Quick Start

```bash
git clone https://github.com/Owen-Foundation/owen2 && cd owen2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/owen2-perft   # must print 97862 (Kiwipete)
ctest --test-dir build
```

## Areas to Contribute

| Area | Code | Notes |
|---|---|---|
| **MTS search** | `src/search/marrow.cpp` | UCB + proof-guided selection |
| **NNUE** | `src/nnue/` | `HalfKP+Threat 81920`, input feature design |
| **Trainer** | `trainer/` | PyTorch pipeline (`sdata → .o2nn`) |
| **Packaging** | `cmake/` | Linux, Windows, macOS, Android, WASM |

## Guidelines

1. **Original code only** — do not copy Stockfish or other engines. The search (`MTS`) and evaluation are original; NNUE architecture follows public literature, not Stockfish sources.
2. **Perft stays green** — any change to move generation must keep perft correct (Kiwipete 97862).
3. **Attribution** — `id author` is `Owen Foundation`.
4. **Tests required** — run `ctest` and `depth` smoke tests before submitting a PR.

## Building

```bash
# Standard
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# Baked single-file (embedded net)
cmake -B build-baked -DOWEN_EMBED_NET=ON -DOWEN_EMBED_NET_PATH=$(pwd)/nets/o2-distilled.o2nn && cmake --build build-baked -j
# Windows cross
x86_64-w64-mingw32-cmake -B build-win && cmake --build build-win -j
# macOS universal
cmake -B build-mac -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

## Distillation

See [DISTILL.md](DISTILL.md) — how Owen 2 distills Stockfish evaluations (UCI `score cp` labels via `trainer/distill.py`) without copying code or weights.

## Releasing

* Bump `project(owen2 VERSION X.Y.Z)` in `CMakeLists.txt`
* Tag `git tag vX.Y.Z && git push --tags`
* Update `CHANGELOG.md`

## Code Style

* C++20, `clang-format`, `-Wall -Wextra`.

PRs welcome — run `ctest` green!
