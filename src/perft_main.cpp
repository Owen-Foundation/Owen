#include "position.h"
#include "movegen.h"
#include "bitboard.h"
#include <iostream>
#include <chrono>
using namespace owen2;
int main(){
    init_attacks(); Position::init_zobrist();
    struct Case{ const char* fen; int depth; uint64_t expected; };
    Case cases[]={
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
    };
    for(auto &c: cases){
        Position p; p.set_fen(c.fen);
        for(int d=1; d<=c.depth; ++d){
            Position tmp=p;
            auto t0=std::chrono::steady_clock::now();
            uint64_t n=perft(tmp,d);
            auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();
            std::cout << "FEN: " << c.fen << "\n  perft " << d << " = " << n << " ("<<ms<<" ms)";
            if(d==c.depth) std::cout << (n==c.expected?"  OK":"  FAIL expected "+std::to_string(c.expected));
            std::cout << "\n";
        }
    }
    return 0;
}
