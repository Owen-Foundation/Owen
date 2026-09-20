#include <iostream>
#include <memory>
#include <mutex>
#include <cstring>
#include <string>
#include <sstream>
#include <filesystem>
#include <vector>
#include <array>
#include <optional>

#include "sf_bridge.h"
#include "../position.h"

#define private public
#include "/home/hemesh/sf-src/src/types.h"
#include "/home/hemesh/sf-src/src/position.h"
#include "/home/hemesh/sf-src/src/attacks.h"
#include "/home/hemesh/sf-src/src/evaluate.h"
#include "/home/hemesh/sf-src/src/nnue/network.h"
#include "/home/hemesh/sf-src/src/nnue/nnue_accumulator.h"
#undef private

namespace owen2::sf_nnue {

static std::unique_ptr<Stockfish::Eval::NNUE::Network> g_sf_network = nullptr;
static std::unique_ptr<Stockfish::Eval::NNUE::AccumulatorCaches> g_sf_caches = nullptr;
static bool g_is_loaded = false;
static std::mutex g_sf_mutex;

static const Stockfish::Piece OWEN_TO_SF_PIECE[13] = {
    Stockfish::W_PAWN, Stockfish::W_KNIGHT, Stockfish::W_BISHOP, Stockfish::W_ROOK, Stockfish::W_QUEEN, Stockfish::W_KING,
    Stockfish::B_PAWN, Stockfish::B_KNIGHT, Stockfish::B_BISHOP, Stockfish::B_ROOK, Stockfish::B_QUEEN, Stockfish::B_KING,
    Stockfish::NO_PIECE
};

bool load_sf_net(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_sf_mutex);
    Stockfish::Attacks::init();
    Stockfish::Position::init();

    if (!g_sf_network) {
        g_sf_network = std::make_unique<Stockfish::Eval::NNUE::Network>();
    }

    Stockfish::Eval::NNUE::EvalFile evalFile;
    evalFile.current = std::nullopt;
    g_sf_network->load("", path, evalFile);

    if (evalFile.current.has_value()) {
        g_sf_caches = std::make_unique<Stockfish::Eval::NNUE::AccumulatorCaches>(*g_sf_network);
        g_is_loaded = true;
        return true;
    }

    g_is_loaded = false;
    return false;
}

bool is_sf_net_loaded() {
    return g_is_loaded && g_sf_network != nullptr;
}

// Fast converter that skips string formatting completely
static inline void set_sf_position_fast(const owen2::Position& src, Stockfish::Position& dst, Stockfish::StateInfo* si) {
    std::memset(si, 0, sizeof(Stockfish::StateInfo));
    dst.st = si;
    dst.gamePly = src.ply();
    dst.sideToMove = Stockfish::Color(src.side_to_move());

    std::memset(dst.pieceCount, 0, sizeof(dst.pieceCount));
    dst.byTypeBB.fill(0);
    dst.byColorBB.fill(0);
    dst.board.fill(Stockfish::NO_PIECE);

    for (int s = 0; s < 64; ++s) {
        owen2::Piece opc = src.piece_on(owen2::Square(s));
        if (opc != owen2::NO_PIECE) {
            Stockfish::Piece spc = OWEN_TO_SF_PIECE[opc];
            Stockfish::Square ssq = Stockfish::Square(s);
            Stockfish::Color sc = Stockfish::color_of(spc);
            Stockfish::PieceType spt = Stockfish::type_of(spc);

            dst.board[ssq] = spc;
            dst.byTypeBB[Stockfish::ALL_PIECES] |= Stockfish::square_bb(ssq);
            dst.byTypeBB[spt] |= Stockfish::square_bb(ssq);
            dst.byColorBB[sc] |= Stockfish::square_bb(ssq);
            dst.pieceCount[spc]++;
        }
    }

    dst.set_state();
}

int evaluate(const owen2::Position& pos) {
    if (!is_sf_net_loaded()) return 0;

    Stockfish::StateInfo si;
    Stockfish::Position sf_pos;
    set_sf_position_fast(pos, sf_pos, &si);

    Stockfish::Eval::NNUE::AccumulatorStack accumulators;
    auto [psqt, positional] = g_sf_network->evaluate(sf_pos, accumulators, *g_sf_caches);
    
    int raw_score = (int)(psqt + positional);
    return raw_score / 16;
}

}
