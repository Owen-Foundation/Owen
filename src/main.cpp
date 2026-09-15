#include "uci.h"
#include "nnue/network.h"
#include <iostream>
#include <string>
#include <algorithm>

#ifdef OWEN_EMBED_NET
extern const unsigned char g_embedded_net[];
extern const unsigned char g_embedded_net_end[];
static void load_embedded_net(){
    const unsigned char* data = g_embedded_net;
    size_t sz = (size_t)(g_embedded_net_end - g_embedded_net);
    if(sz >= 8 && data[0]=='O' && data[1]=='2' && data[2]=='N' && data[3]=='N'){
        if(owen2::nnue::g_network.load_from_memory(data, sz))
            std::cerr << "info string Owen 2 embedded net loaded (" << sz << " bytes)\n";
        else
            std::cerr << "info string Owen 2 embedded net FAILED to load\n";
    } else {
        std::cerr << "info string Owen 2 embedded net has bad magic\n";
    }
}
#else
static void load_embedded_net(){}
#endif

int main(int argc, char** argv){
    // OpenBench compliance: ./owen2 bench [depth] runs a deterministic
    // startpos bench to stdout (no UCI loop, no net file needed) and exits.
    if(argc > 1 && std::string(argv[1]) == "bench"){
        int depth = 6;
        if(argc > 2){
            try { depth = std::clamp(std::stoi(argv[2]), 1, 10); }
            catch(...) { depth = 6; }
        }
        owen2::init_attacks();
        owen2::Position::init_zobrist();
        std::cout << owen2::run_bench(depth) << "\n" << std::flush;
        return 0;
    }
    load_embedded_net();
    owen2::uci_loop();
    return 0;
}
