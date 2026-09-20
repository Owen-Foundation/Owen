#pragma once
#include "types.h"
#include "bitboard.h"
#include <vector>
#include <string>
#include <array>

namespace owen2 {

struct StateInfo {
    uint64_t key = 0;
    Bitboard checkers = 0;
    Square ep_square = 64;
    int castling = 0;
    int rule50 = 0;
    Piece captured = NO_PIECE;
    Piece moved = NO_PIECE;
    Move last_move = 0;
};

class Position {
public:
    Position(){ set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

    void set_fen(const std::string& fen);
    std::string fen() const;

    Color side_to_move() const { return stm_; }
    Bitboard pieces(Color c, PieceType pt) const { return byColorType_[c][pt]; }
    Bitboard pieces(Color c) const { return byColor_[c]; }
    Bitboard pieces(PieceType pt) const { return byType_[pt]; }
    Bitboard occupancy() const { return byColor_[WHITE] | byColor_[BLACK]; }
    Piece piece_on(Square s) const { return board_[s]; }
    uint64_t key() const { return st_.key; }
    int ply() const { return ply_; }
    int rule50() const { return st_.rule50; }
    Square ep_square() const { return st_.ep_square; }
    int castling_rights() const { return st_.castling; }
    Bitboard checkers() const { return st_.checkers; }
    bool in_check() const { return st_.checkers != 0; }
    Square king_sq(Color c) const { return kingSq_[c]; }

    void do_move(Move m);
    void do_move(Move m, bool record);
    void undo_move(Move m);
    bool is_draw() const;

    Bitboard attackers_to(Square s, Bitboard occ) const;
    bool square_attacked(Square s, Color by) const;

    static void init_zobrist();

private:
    void update_checkers();
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void recompute_occupancy();

    Piece board_[64];
    Bitboard byColor_[2]{};
    Bitboard byColorType_[2][6]{};
    Bitboard byType_[6]{};
    Square kingSq_[2]{4,60};
    Color stm_ = WHITE;
    StateInfo st_{};
    int ply_ = 0;
    std::vector<StateInfo> history_;

    static uint64_t ZobristPiece[12][64];
    static uint64_t ZobristSide;
    static uint64_t ZobristCastle[16];
    static uint64_t ZobristEP[8];
    static bool zobristInit_;
};

} // namespace owen2
