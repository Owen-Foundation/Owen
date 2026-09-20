#include "../src/position.h"
#include "../src/movegen.h"
#include "../src/bitboard.h"
#include "../src/nnue/features.h"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
using namespace owen2;
using namespace owen2::nnue;
// NOTE: Release builds define NDEBUG (assert is a no-op), so tests use
// CHECK, which always aborts on failure.
#define CHECK(cond) do{ if(!(cond)){ std::cerr<<"CHECK FAILED line "<<__LINE__<<": "<<#cond<<"\n"; return 1; } }while(0)

static std::vector<int16_t> random_weights(unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<int16_t> w((size_t)INPUT_SIZE * HIDDEN_SIZE);
    for (size_t i = 0; i < w.size(); i++) w[i] = (int16_t)(rng() % 65 - 32);
    return w;
}

static bool acc_equal(const Accumulator& a, const Accumulator& b) {
    return a.white == b.white && a.black == b.black;
}

// Walk random games, comparing incremental slots against full refresh.
static int walk(const std::vector<int16_t>& weights, Position pos,
                std::mt19937_64& rng, int moves) {
    std::vector<AccState> slots(moves + 2);
    refresh_acc_state(pos, slots[0], weights.data());
    // baseline slot 0
    {
        AccState ref;
        refresh_acc_state(pos, ref, weights.data());
        CHECK(acc_equal(slots[0].acc, ref.acc));
        CHECK(slots[0].threatW == ref.threatW && slots[0].threatB == ref.threatB);
    }
    Position scratch;
    Move buf[256];
    for (int d = 0; d < moves; d++) {
        int n = generate_legal_buf(scratch, pos, buf, 256);
        if (n == 0) break;  // terminal: stop walk
        Move m = buf[rng() % (unsigned)n];
        AccMove u = capture_acc_move(pos, m);
        pos.do_move(m);
        apply_acc_move(pos, u, weights.data(), slots[d], slots[d + 1]);
        AccState ref;
        refresh_acc_state(pos, ref, weights.data());
        if (!acc_equal(slots[d + 1].acc, ref.acc) ||
            slots[d + 1].threatW != ref.threatW ||
            slots[d + 1].threatB != ref.threatB) {
            std::cerr << "MISMATCH at walk depth " << d
                      << " move " << move_to_uci(m)
                      << " fen " << pos.fen() << "\n";
            int shown = 0;
            for (int i = 0; i < HIDDEN_SIZE && shown < 5; i++) {
                if (slots[d + 1].acc.white[i] != ref.acc.white[i]) {
                    std::cerr << "  white[" << i << "] inc=" << slots[d + 1].acc.white[i]
                              << " ref=" << ref.acc.white[i] << "\n";
                    shown++;
                }
            }
            for (int i = 0; i < HIDDEN_SIZE && shown < 10; i++) {
                if (slots[d + 1].acc.black[i] != ref.acc.black[i]) {
                    std::cerr << "  black[" << i << "] inc=" << slots[d + 1].acc.black[i]
                              << " ref=" << ref.acc.black[i] << "\n";
                    shown++;
                }
            }
            std::cerr << "  masks W inc=" << std::hex << slots[d + 1].threatW
                      << " ref=" << ref.threatW << " B inc=" << slots[d + 1].threatB
                      << " ref=" << ref.threatB << std::dec << "\n";
            return 1;
        }
        // forward values follow from acc equality (same weights, same
        // accumulator => identical dot products by construction).
    }
    return 0;
}

int test_incremental_main() {
    init_attacks();
    Position::init_zobrist();
    auto weights = random_weights(0xC0FFEE);
    std::vector<const char*> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b kq - 0 1",
        "8/2P5/1K6/8/8/1k6/8/8 w - - 0 1",
        "8/1p6/1K6/8/8/1k6/8/8 b - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "2rr3k/pp3pp1/1nn2n1p/3p4/3P4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 1",
    };
    std::mt19937_64 rng(12345);
    // TEMP: ground truth — my refresh_acc_state vs production refresh_accumulator
    {
        Position p;
        p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        AccState mine;
        refresh_acc_state(p, mine, weights.data());
        Accumulator prod;
        refresh_accumulator(p, prod, weights.data());
        int dw = 0, db = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            if (mine.acc.white[i] != prod.white[i] && dw < 3)
                { std::cerr << "GT-WHITE i=" << i << " mine=" << mine.acc.white[i] << " prod=" << prod.white[i] << "\n"; dw++; }
            if (mine.acc.black[i] != prod.black[i] && db < 3)
                { std::cerr << "GT-BLACK i=" << i << " mine=" << mine.acc.black[i] << " prod=" << prod.black[i] << "\n"; db++; }
        }
        std::cerr << "ground truth: white " << (dw ? "DIFFERS" : "same")
                  << ", black " << (db ? "DIFFERS" : "same") << "\n";
    }

    long total = 0;
    for (int round = 0; round < 40; round++) {
        for (auto f : fens) {
            Position p;
            p.set_fen(f);
            if (walk(weights, p, rng, 60)) return 1;
            total += 60;
        }
    }
    // different weights, more seeds
    auto weights2 = random_weights(0xDEADBEEF);
    for (int round = 0; round < 10; round++) {
        Position p;
        p.set_fen(fens[round % fens.size()]);
        if (walk(weights2, p, rng, 80)) return 1;
        total += 80;
    }
    std::cout << "incremental OK: " << total << " positions verified\n";
    return 0;
}
