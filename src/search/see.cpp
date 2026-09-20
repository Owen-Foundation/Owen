#include "see.h"
#include "../bitboard.h"

namespace owen2 {

namespace {
constexpr int V[6] = {100, 320, 330, 500, 900, 20000};
} // namespace

int see(const Position& pos, Move m) {
    if (!is_capture(m) && !is_promo(m)) return 0;
    Square from = move_from(m), to = move_to(m);
    Piece moving = pos.piece_on(from);
    if (moving == NO_PIECE) return 0;
    Color stm = pos.side_to_move();
    int flags = move_flags(m);

    Piece cap = pos.piece_on(to);
    Square capSq = to;
    if (flags & MoveFlag::ENPASSANT) {
        capSq = make_square(file_of(to), rank_of(from));
        cap = pos.piece_on(capSq);
        if (cap == NO_PIECE || type_of(cap) != PAWN) return 0;
    }
    // King capture ends the game on the spot.
    if (cap != NO_PIECE && type_of(cap) == KING) return V[KING];

    // cap[0]: initial victim (+ promotion delta). atk[0]: capturer's value.
    int capV[32];
    int atkV[32];
    capV[0] = (cap == NO_PIECE ? 0 : V[type_of(cap)]);
    atkV[0] = V[type_of(moving)];
    if (flags & MoveFlag::PROMO) {
        PieceType pr = move_promo(m);
        capV[0] += V[pr] - V[PAWN];
        atkV[0] = V[pr];
    }
    // A king capture takes for free: destination was proven safe by legality.
    if (type_of(moving) == KING && !(flags & MoveFlag::PROMO)) return capV[0];

    Bitboard occ = pos.occupancy() & ~sq_bb(from);
    if ((flags & MoveFlag::ENPASSANT) && capSq != to) occ &= ~sq_bb(capSq);

    // Build the swap list. Kings excluded as recapturers (see header).
    int d = 0;
    Color side = Color(stm ^ 1);
    while (d < 31) {
        Bitboard atk = pos.attackers_to(to, occ);
        Square asq = -1;
        PieceType apt = KING;
        for (int pt = PAWN; pt <= QUEEN; ++pt) {
            Bitboard bb = atk & pos.pieces(side, (PieceType)pt);
            if (bb) { asq = lsb(bb); apt = (PieceType)pt; break; }
        }
        if (asq < 0) break;
        ++d;
        capV[d] = atkV[d - 1]; // recapture takes the previous capturer
        atkV[d] = V[apt];
        occ &= ~sq_bb(asq);
        side = Color(side ^ 1);
    }

    // Backward induction with stand-pat: G[d+1] = 0; for i>=1 keep the
    // capture iff it improves mover i's own outcome. Answer forces the
    // initial capture (the move under consideration IS made).
    int future = 0; // G[i+1], stm-relative
    for (int i = d; i >= 1; --i) {
        int sigma = (i % 2 == 0) ? 1 : -1; // mover i is stm iff i even
        if (capV[i] + sigma * future > 0)
            future = sigma * capV[i] + future;
        else
            future = 0; // mover i stands pat: sequence ends, prior stands
    }
    return capV[0] + future;
}

} // namespace owen2
