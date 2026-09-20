#include "../src/position.h"
#include "../src/movegen.h"
#include "../src/bitboard.h"
#include "../src/search/see.h"
#include <cassert>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
using namespace owen2;
// NOTE: Release builds define NDEBUG (assert is a no-op), so tests use
// CHECK, which always aborts on failure.
#define CHECK(cond) do{ if(!(cond)){ std::cerr<<"CHECK FAILED line "<<__LINE__<<": "<<#cond<<"\n"; return 1; } }while(0)
int main(){
    init_attacks(); Position::init_zobrist();
    Position p;
    // Kiwipete
    p.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    Position t=p;
    uint64_t n=perft(t,3);
    std::cout<<"kiwipete perft 3 = "<<n<<" expect 97862\n";
    CHECK(n==97862);
    std::cout<<"perft tests passed\n";

    // SEE: quiet moves score 0, hanging pieces are winning, defended grabs lose.
    // Knight on e4 attacks c5 (not d5!). Pawn b6 defends c5.
    {
        // White Ne4 -> d6 is quiet (d6 empty): SEE 0.
        Position q; q.set_fen("8/8/8/2p5/4N3/8/8/4K2k w - - 0 1");
        Move m = parse_uci_move(q, "e4d6");
        CHECK(m && !is_capture(m));
        CHECK(see(q, m) == 0);
    }
    {
        // Nxc5 with pawn defender b6: wins pawn (100), loses knight (320).
        Position q; q.set_fen("8/8/1p6/2p5/4N3/8/8/4K2k w - - 0 1");
        Move cx = parse_uci_move(q, "e4c5");
        CHECK(cx && is_capture(cx));
        int v = see(q, cx);
        std::cout<<"SEE Nxc5 defended = "<<v<<" expect <0\n";
        CHECK(v < 0);
    }
    {
        // Same but undefended: Nxc5 wins 100.
        Position q; q.set_fen("8/8/8/2p5/4N3/8/8/4K2k w - - 0 1");
        Move cx = parse_uci_move(q, "e4c5");
        CHECK(cx && is_capture(cx));
        int v = see(q, cx);
        std::cout<<"SEE Nxc5 hanging = "<<v<<" expect 100\n";
        CHECK(v == 100);
    }
    {
        // EP: white pawn d5, black pawn e5, EP square e6 -> dxe6 wins a pawn.
        Position q; q.set_fen("8/8/8/3Pp3/8/8/8/4K2k w - e6 0 1");
        Move ep = parse_uci_move(q, "d5e6");
        CHECK(ep && is_capture(ep));
        int v = see(q, ep);
        std::cout<<"SEE dxe6 e.p. = "<<v<<" expect 100\n";
        CHECK(v == 100);
    }
    std::cout<<"see tests passed\n";

    // captures-only generator == capture/promo subset of full legal, on
    // random positions (middlegame + endgame + checks).
    {
        std::mt19937 rng(20260912);
        const char* fens[] = {
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "8/8/4p3/3p4/4N3/8/8/4K2k w - - 0 1",
            "8/8/8/3Pp3/8/8/8/4K2k w - e6 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        };
        for(auto fen : fens){
            Position q; q.set_fen(fen);
            for(int g=0; g<40; ++g){
                auto full = generate_legal(q);
                auto caps = generate_captures(q);
                std::vector<Move> expect;
                for(auto m : full) if(is_capture(m) || is_promo(m)) expect.push_back(m);
                auto sortv = [](std::vector<Move>& v){ std::sort(v.begin(), v.end()); };
                sortv(full); sortv(caps); sortv(expect);
                CHECK(caps == expect);
                if(full.empty()) break;
                q.do_move(full[rng() % full.size()]);
            }
        }
    }
    std::cout<<"captures tests passed\n";
    extern int test_incremental_main();
    int rc_inc = test_incremental_main();
    if (rc_inc != 0) return rc_inc;
    return 0;
}
