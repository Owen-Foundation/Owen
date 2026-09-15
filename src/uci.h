#pragma once
#include "position.h"
#include "search/search.h"
#include <string>

namespace owen2 {

void uci_loop();
// Deterministic startpos bench (single thread, no net file needed).
// Returns the "bench depth D nodes N nps M ..." line (OpenBench parses it).
std::string run_bench(int depth);

} // namespace owen2
