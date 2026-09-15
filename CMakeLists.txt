cmake_minimum_required(VERSION 3.20)
project(owen2 VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OWEN_NATIVE "Build with -march=native" ON)
option(OWEN_AVX2   "Enable AVX2 for NNUE" OFF)
option(OWEN_EMBED_NET "Bake .o2nn into binary (like Stockfish)" OFF)
set(OWEN_EMBED_NET_PATH "nets/o2-final.o2nn" CACHE STRING "Path to net to embed")

add_compile_options(-Wall -Wextra -Wpedantic)
if(OWEN_NATIVE)
  add_compile_options(-march=native)
endif()
if(OWEN_AVX2)
  add_compile_options(-mavx2 -mbmi2)
  add_compile_definitions(OWEN_AVX2)
endif()

# ── core library (everything except main) ──────────────────────────
file(GLOB CORE_SRC
  src/bitboard.cpp
  src/position.cpp
  src/movegen.cpp
  src/nnue/features.cpp
  src/nnue/network.cpp
  src/nnue/eval_cache.cpp
  src/search/tt.cpp
  src/search/see.cpp
  src/search/marrow.cpp
  src/search/search.cpp
  src/uci.cpp
)
add_library(owen2_core STATIC ${CORE_SRC})
target_include_directories(owen2_core PUBLIC src)
target_link_libraries(owen2_core PUBLIC pthread)

# ── main engine ───────────────────────────────────────────────────
# ── bake net into engine when requested (like Stockfish) ──────
if(OWEN_EMBED_NET)
  include(cmake/embed_net.cmake)
  add_executable(owen2 src/main.cpp)
  target_link_libraries(owen2 PRIVATE owen2_core)
  embed_net(${OWEN_EMBED_NET_PATH})
else()
  add_executable(owen2 src/main.cpp)
  target_link_libraries(owen2 PRIVATE owen2_core)
endif()

# ── perft standalone ──────────────────────────────────────────────
add_executable(owen2-perft src/perft_main.cpp)
target_link_libraries(owen2-perft PRIVATE owen2_core)

# ── sdata generator ───────────────────────────────────────────────
add_executable(owen2-sdata src/sdata_gen.cpp)
target_link_libraries(owen2-sdata PRIVATE owen2_core)

# ── binpack <-> sdata converter (ecosystem .binpack format, own games) ──
add_executable(owen2-binpack2sdata src/binpack2sdata.cpp)
target_link_libraries(owen2-binpack2sdata PRIVATE owen2_core)

# ── tests ─────────────────────────────────────────────────────────
enable_testing()
file(GLOB TEST_SRC tests/*.cpp)
if(TEST_SRC)
  add_executable(owen2-tests ${TEST_SRC})
  target_link_libraries(owen2-tests PRIVATE owen2_core)
  add_test(NAME owen2-tests COMMAND owen2-tests)
endif()

install(TARGETS owen2 DESTINATION bin)
install(FILES nets/o2-v1.o2nn DESTINATION share/owen2 OPTIONAL)
