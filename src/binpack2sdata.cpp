// binpack2sdata — convert an ecosystem .binpack file (Owen's own games, or any
// compatible binpack) into Owen 71-byte v2 sdata records for training.
// Usage: owen2-binpack2sdata --in file.binpack --out file.bin [--max N]
#include "binpack.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#pragma pack(push,1)
struct SDataRecord {
    uint8_t board[64];
    uint8_t stm;
    int16_t eval;
    uint8_t result;
    uint8_t ply;
    uint8_t castling;
    uint8_t ep;
    uint16_t move16; // v3 policy label
};
#pragma pack(pop)
static_assert(sizeof(SDataRecord) == 73, "pack broken");

int main(int argc, char** argv) {
    std::string in, out = "data/from-binpack.bin";
    long maxN = -1, skipN = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--in" && i + 1 < argc) in = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--max" && i + 1 < argc) maxN = std::stol(argv[++i]);
        else if (a == "--skip" && i + 1 < argc) skipN = std::stol(argv[++i]);
        else if (a == "--verbose" || a == "-v") binpack::Reader::verbose = true;
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: owen2-binpack2sdata --in file.binpack --out file.bin [--max N]\n";
            return 0;
        }
    }
    if (in.empty()) { std::cerr << "need --in\n"; return 1; }
    owen2::init_attacks();
    owen2::Position::init_zobrist();

    std::ofstream f(out, std::ios::binary | std::ios::trunc);
    if (!f) { std::cerr << "cannot open " << out << "\n"; return 1; }
    long n = 0, skipped = 0, seen = 0;
    struct Done {}; // unwinds the stream once the quota is met
    struct Writer : binpack::Reader::Sink {
        std::ofstream& f; long &n, &skipped, &seen;
        long maxN, skipN;
        Writer(std::ofstream& f_, long& n_, long& s_, long& seen_, long mx, long sk)
            : f(f_), n(n_), skipped(s_), seen(seen_), maxN(mx), skipN(sk) {}
        void emit(const binpack::DecodedPos& dp) override {
            ++seen;
            if (skipped < skipN) { ++skipped; return; }
            if (maxN >= 0 && n >= maxN) throw Done{};
            SDataRecord r{};
            for (int s = 0; s < 64; ++s) r.board[s] = (uint8_t)std::clamp(dp.board[s], 0, 12);
            r.stm = (uint8_t)(dp.stm & 1);
            r.eval = (int16_t)std::clamp(dp.score, -15000, 15000);
            int ri = std::clamp(dp.result, -1, 1);
            r.result = (uint8_t)(ri + 1); // -1->0 loss, 0->1 draw, +1->2 win
            r.ply = (uint8_t)std::min(dp.ply, 255);
            r.castling = (uint8_t)(dp.rights & 15);
            r.ep = (uint8_t)(dp.ep >= 64 ? 64 : dp.ep);
            r.move16 = dp.move16();
            f.write((char*)&r, sizeof(r));
            ++n;
        }
    };
    Writer w(f, n, skipped, seen, maxN, skipN);
    try {
        binpack::Reader::stream(in, w);
    } catch (const Done&) {
        // quota met, clean stop
    } catch (const std::exception& e) {
        std::cerr << "read failed after " << seen << " positions: " << e.what() << "\n";
        return 1;
    }
    std::cout << "binpack2sdata: " << seen << " decoded, " << n << " written -> " << out << "\n";
    return 0;
}
