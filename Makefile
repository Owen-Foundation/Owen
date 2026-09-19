# Owen 2 — OpenBench-compatible build wrapper around the CMake build.
# Usage (as invoked by OpenBench clients):
#   make EXE=Engine-XXXX            build ./Engine-XXXX with default compiler
#   make EXE=Engine-XXXX CXX=clang++  build with a specific C++ compiler
#   make EVALFILE=/path/to/net.nnue EXE=...  embed a network at compile time
#   make clean
CXX      ?= g++
EXE      ?= owen2
EVALFILE ?=
BUILD_DIR ?= build-openbench
NPROC     ?= $(shell nproc 2>/dev/null || echo 4)

EMBED_SRC  = $(BUILD_DIR)/embedded_net.cpp
CMAKE_DEFS = -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(CXX) $(CMAKE_EXTRA)

all: $(EXE)

$(EXE):
	@if [ -n "$(EVALFILE)" ]; then \
	  if [ ! -f "$(EVALFILE)" ]; then echo "EVALFILE not found: $(EVALFILE)"; exit 1; fi; \
	  echo "Embedding network $(EVALFILE)"; \
	fi
	@mkdir -p $(BUILD_DIR)
	@if [ -n "$(EVALFILE)" ]; then \
	  cmake -S . -B $(BUILD_DIR) $(CMAKE_DEFS) -DOWEN_EMBED_NET=1 -DOWEN_EMBED_NET_PATH=$(EVALFILE) > /dev/null; \
	else \
	  cmake -S . -B $(BUILD_DIR) $(CMAKE_DEFS) > /dev/null; \
	fi
	cmake --build $(BUILD_DIR) --target owen2 -j$(NPROC)
	cp $(BUILD_DIR)/owen2 $(EXE)

clean:
	rm -rf $(BUILD_DIR) $(EXE)

# Always rebuild: OpenBench passes a unique EXE per build and expects
# current sources/flags honored on every invocation.
.PHONY: all clean $(EXE)
