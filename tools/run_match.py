#!/usr/bin/env python3
"""Owen 2 — UCI match runner (self-play / calibration).

No cutechess needed: drives two UCI engines via python-chess with opening
variety from an EPD book, adjudication, concurrency, PGN output and Elo.

Usage:
  # net vs handcrafted (fallback = bogus NNUE path), 32 openings x colors:
  python3 tools/run_match.py \
    --white-cmd ./build/owen2 --white-name o2-final \
    --white-opt Threads=1 NNUEFile=nets/o2-final.o2nn \
    --black-cmd ./build/owen2 --black-name handcrafted \
    --black-opt Threads=1 NNUEFile=/nonexistent.o2nn \
    --book tools/book.epd --movetime 0.3 --concurrency 6 --out tools/logs/match.pgn
  # vs Stockfish:
  python3 tools/run_match.py --white-cmd ./build/owen2 ... \
    --black-cmd /usr/games/stockfish --black-name "SF16" \
    --black-opt "Threads=4 Hash=256" --games 8 --movetime 0.5

Notes:
  - Engines are deterministic, so every (opening, color) pair is one game;
    --games caps the number of PAIRS (each pair = both colors).
  - Adjudication: resign when a side's own score <= -700cp for 3 consecutive
    own moves (mate scores count); draw at 250 plies or dead positions.
"""
import argparse
import concurrent.futures
import datetime
import math
import threading

import chess
import chess.engine
import chess.pgn


def parse_opts(s):
    d = {}
    if not s:
        return d
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def play_game(white_spec, black_spec, fen, movetime, resign_cp, resign_plies,
              max_plies):
    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "Owen match"
    game.headers["Site"] = "local"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["White"] = white_spec["name"]
    game.headers["Black"] = black_spec["name"]
    game.headers["FEN"] = fen
    game.setup(board)
    node = game

    engines = {}
    try:
        for color, spec in (("w", white_spec), ("b", black_spec)):
            eng = chess.engine.SimpleEngine.popen_uci(spec["cmd"])
            try:
                eng.ucinewgame()
            except Exception:
                pass
            for k, v in spec["opts"].items():
                try:
                    eng.configure({k: v})
                except Exception as e:
                    print(f"  [{spec['name']}] setoption {k} failed: {e}")
            engines[color] = eng

        bad_streak = {"w": 0, "b": 0}
        result = None
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            turn = "w" if board.turn == chess.WHITE else "b"
            eng = engines[turn]
            try:
                res = eng.play(board, chess.engine.Limit(time=movetime),
                               info=chess.engine.INFO_SCORE)
            except Exception as e:
                print(f"  engine {turn} error: {e}")
                result = "0-1" if turn == "w" else "1-0"
                break
            board.push(res.move)
            node = node.add_variation(res.move)
            # resign tracking from mover's own score (info scores are
            # side-to-move-relative, and the mover was to move).
            try:
                sc = res.info.get("score")
                mover_cp = None
                if sc is not None:
                    rel = sc.relative
                    if rel.is_mate():
                        m = rel.mate() or 0
                        mover_cp = 100000 if m > 0 else -100000
                    else:
                        mover_cp = rel.score()
                if mover_cp is not None and mover_cp <= -abs(resign_cp):
                    bad_streak[turn] += 1
                else:
                    bad_streak[turn] = 0
                if bad_streak[turn] >= resign_plies:
                    result = "0-1" if turn == "w" else "1-0"
                    break
            except Exception:
                pass
        if result is None:
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            elif (board.is_stalemate() or board.is_insufficient_material()
                    or board.is_seventyfive_moves()
                    or board.is_fivefold_repetition()):
                result = "1/2-1/2"
            else:
                result = "1/2-1/2"  # max plies / agreement-by-exhaustion
    finally:
        for eng in engines.values():
            try:
                eng.quit()
            except Exception:
                pass
    game.headers["Result"] = result
    return result, game


def elo_report(w, d, l):
    n = w + d + l
    if n == 0:
        return "no games"
    s = (w + d / 2) / n
    if s <= 0:
        return f"Elo: -inf  (0/{n})"
    if s >= 1:
        return f"Elo: +inf  ({n}/{n})"
    elo = -400 * math.log10(1 / s - 1)
    var = (w * (1 - s) ** 2 + l * s ** 2 + d * (0.5 - s) ** 2) / n
    se = math.sqrt(var / n) if var > 0 else 0.0
    derr = 400 / (math.log(10) * s * (1 - s)) if 0 < s < 1 else 0.0
    err = 1.96 * se * derr
    return (f"Score {s * 100:.1f}%  W{w} D{d} L{l}  "
            f"Elo diff {elo:+.0f} ± {err:.0f} (95%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--white-cmd", required=True)
    ap.add_argument("--black-cmd", required=True)
    ap.add_argument("--white-name", default="White")
    ap.add_argument("--black-name", default="Black")
    ap.add_argument("--white-opt", default="")
    ap.add_argument("--black-opt", default="")
    ap.add_argument("--book", default="tools/book.epd")
    ap.add_argument("--games", type=int, default=0,
                    help="opening PAIRS (x2 games with swapped colors); 0 = whole book")
    ap.add_argument("--movetime", type=float, default=0.3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="tools/logs/match.pgn")
    ap.add_argument("--resign-cp", type=int, default=700)
    ap.add_argument("--resign-plies", type=int, default=3)
    ap.add_argument("--max-plies", type=int, default=250)
    args = ap.parse_args()

    with open(args.book) as f:
        openings = [line.strip() for line in f if line.strip()]
    if args.games > 0:
        openings = openings[:args.games]
    print(f"{len(openings)} openings x 2 colors = {len(openings) * 2} games")

    white_spec = {"cmd": args.white_cmd, "name": args.white_name,
                  "opts": parse_opts(args.white_opt)}
    black_spec = {"cmd": args.black_cmd, "name": args.black_name,
                  "opts": parse_opts(args.black_opt)}

    jobs = []
    for fen in openings:
        jobs.append((white_spec, black_spec, fen))   # as listed
        jobs.append((black_spec, white_spec, fen))   # colors swapped

    results = []
    lock = threading.Lock()

    def one(job):
        wspec, bspec, fen = job
        r, g = play_game(wspec, bspec, fen, args.movetime,
                         args.resign_cp, args.resign_plies, args.max_plies)
        with lock:
            # attribute from White(outer)-Black(outer) perspective
            if wspec["name"] == white_spec["name"]:
                results.append((r, g))
            else:
                flip = {"1-0": "0-1", "0-1": "1-0"}.get(r, r)
                results.append((flip, g))
            w = sum(1 for x, _ in results if x == "1-0")
            d = sum(1 for x, _ in results if x == "1/2-1/2")
            l = sum(1 for x, _ in results if x == "0-1")
            shown = results[-1][0]
            print(f"  [{len(results)}/{len(jobs)}] {shown:7s}  {elo_report(w, d, l)}",
                  flush=True)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, jobs))

    w = sum(1 for x, _ in results if x == "1-0")
    d = sum(1 for x, _ in results if x == "1/2-1/2")
    l = sum(1 for x, _ in results if x == "0-1")
    print(f"FINAL {args.white_name} vs {args.black_name}: {elo_report(w, d, l)}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for _, g in results:
            print(g, file=f, end="\n\n")
    print(f"PGN -> {args.out}")


if __name__ == "__main__":
    main()
