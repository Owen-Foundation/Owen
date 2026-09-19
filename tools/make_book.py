#!/usr/bin/env python3
"""Owen 2 — PGN -> EPD opening book for engine matches.

Takes positions at a fixed early ply from real games (deduped), so
deterministic engines still produce varied games.

Usage:
  python3 tools/make_book.py --pgn ~/Videos/Abdusattorov.pgn \
      --out tools/book.epd --positions 32 --ply 12
"""
import argparse
import chess
import chess.pgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", default="tools/book.epd")
    ap.add_argument("--positions", type=int, default=32)
    ap.add_argument("--ply", type=int, default=12,
                    help="plies from start to take the book position at")
    ap.add_argument("--min-ply-game", type=int, default=20,
                    help="skip games shorter than this")
    args = ap.parse_args()

    seen = set()
    fens = []
    with open(args.pgn, encoding="utf-8", errors="ignore") as f:
        while len(fens) < args.positions:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                break
            if game is None:
                break
            board = game.board()
            plies = list(game.mainline_moves())
            if len(plies) < args.min_ply_game:
                continue
            for mv in plies[:args.ply]:
                board.push(mv)
            if board.is_game_over():
                continue
            epd = board.epd()
            key = " ".join(board.fen().split()[:4])
            if key in seen:
                continue
            seen.add(key)
            fens.append(board.fen())

    with open(args.out, "w") as out:
        for fen in fens:
            out.write(fen + "\n")
    print(f"Wrote {len(fens)} openings -> {args.out}")


if __name__ == "__main__":
    main()
