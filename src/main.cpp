#include "uci.h"
#include "nnue/network.h"
#include <iostream>
#include <string>
#include <algorithm>

#ifdef OWEN_EMBED_NET
#ifdef OWEN_EMBED_WINRC
// Windows/MSVC: the net is linked as an RCDATA resource (see baked_net.rc).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static bool embedded_net_bytes(const unsigned char*& data, size_t& sz){
    HRSRC h = FindResourceA(NULL, MAKEINTRESOURCEA(1), RT_RCDATA);
    if(!h) return false;
    HGLOBAL g = LoadResource(NULL, h);
    if(!g) return false;
    data = (const unsigned char*)LockResource(g);
    sz = (size_t)SizeofResource(NULL, h);
    return data != nullptr;
}
#else
// Unix: the net lives in a .incbin translation unit (see embedded_net.cpp).
extern const unsigned char g_embedded_net[];
extern const unsigned char g_embedded_net_end[];
static bool embedded_net_bytes(const unsigned char*& data, size_t& sz){
    data = g_embedded_net;
    sz = (size_t)(g_embedded_net_end - g_embedded_net);
    return true;
}
#endif
static void load_embedded_net(){
    const unsigned char* data = nullptr;
    size_t sz = 0;
    if(!embedded_net_bytes(data, sz))
        std::cerr << "info string Owen 2 embedded net unavailable\n";
    else if(sz >= 8 && data[0]=='O' && data[1]=='2' && data[2]=='N' && data[3]=='N'){
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
    load_embedded_net();
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
    owen2::uci_loop();
    return 0;
}
