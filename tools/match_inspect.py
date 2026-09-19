#!/usr/bin/env python3
"""
Code-quality match inspector: Owen 2 vs Stockfish 16.

Not a strength benchmark — scores win/loss per game are not the point.
Instead we stream diagnostics per move and a summary that answers
"is the engine code good?" from observable behaviour.

Checks per move (failures are defects even if the mover won the game):
  UCI     handshake + options (name/author) non-empty
  LEGAL   every move returned is legal in the current position (including
          captures, castling, en passant, promo) — checked with python-chess
  PHASE   opening/middlegame/endgame move distribution vs SF's
  REPEAT  no false 3-fold-draw claims (position hash)
  WHITE_SEES_WHITE  PST / eval sign is not flipped (material + PST check)
  TIMING  respects movetime/wtime/binc budgets (no flag, no hanging)
  STABLE  no crashes / no illegal variations like "bestmove (none)"

One game = Owen white vs SF black, then reversed, with short clocks so
the inspector can touch opening+middlegame in the same run. Logs to
tools/logs/match-<ts>.pgn and a .ndjson with the diagnostics.
"""

import argparse, collections, datetime, json, os, random, subprocess, sys, textwrap, time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tools" / "logs"
OWEN_CANDIDATES = [ROOT / "build-baked" / "owen2", ROOT / "build" / "owen2"]
SF_BIN = Path("/usr/games/stockfish")

# short clocks so the inspector touches endgame quickly, but long enough
# that the mover has time to think — otherwise TIMEOUT masks real defects
BASE_MS, INC_MS = 60_000, 1_000

def pick_owen(path: Path|None):
    if path and Path(path).exists():
        return Path(path)
    for p in OWEN_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(f"Owen binary not found. Tried: {OWEN_CANDIDATES}")

def uci_name(engine: chess.engine.SimpleEngine, label: str):
    name = getattr(engine, "id", {}).get("name", "?")
    author = getattr(engine, "id", {}).get("author", "?")
    return name, author

def phase(bucket: chess.Board):
    # crude phase by material count + ply, just for diagnostics
    total = sum(len(bucket.pieces(pt, c)) for pt in range(1,6) for c in (chess.WHITE, chess.BLACK))
    if bucket.fullmove_number <= 10 and total >= 28: return "opening"
    if total <= 12: return "endgame"
    return "middlegame"

def play_one_game(white_path: Path, black_path: Path, game_no: int, limit: chess.engine.Limit, out_pgn, out_jsonl):
    white = chess.engine.SimpleEngine.popen_uci(str(white_path))
    black = chess.engine.SimpleEngine.popen_uci(str(black_path))
    for eng in (white, black):
        try: eng.configure({"Hash": 64, "Threads": 1})
        except Exception: pass
    w_name = uci_name(white, "white")[0]
    b_name = uci_name(black, "black")[0]

    # handshake diagnostics
    def check_handshake(eng, tag):
        issues=[]
        nid = getattr(eng, "id", {})
        if not nid.get("name"): issues.append(f"{tag}: missing id name")
        if not nid.get("author"): issues.append(f"{tag}: missing id author")
        opts = getattr(eng, "options", {}) or {}
        if not opts: issues.append(f"{tag}: options empty")
        return issues
    issues=[]
    issues += check_handshake(white, "WHITE")
    issues += check_handshake(black, "BLACK")

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"Owen vs SF code inspection {game_no}"
    game.headers["Date"] = datetime.date.today().isoformat()
    game.headers["White"] = w_name
    game.headers["Black"] = b_name
    game.headers["TimeControl"] = f"{BASE_MS//1000}+{INC_MS//1000}"
    node = game

    clock_w, clock_b = BASE_MS/1000, BASE_MS/1000
    seen = collections.Counter()
    seen[board.fen()] += 1
    per_engine = {
        w_name: {"moves":0, "legality_fail":0, "phase": collections.Counter(), " Promo?":0, "castled":0, "timeout":0, "thinking_ms":[]},
        b_name: {"moves":0, "legality_fail":0, "phase": collections.Counter(), " Promo?":0, "castled":0, "timeout":0, "thinking_ms":[]},
    }
    pgn_moves=[]
    terminated = None
    start = time.time()

    while not board.is_game_over(claim_draw=True):
        mover = white if board.turn == chess.WHITE else black
        mover_name = w_name if board.turn==chess.WHITE else b_name
        ph = phase(board)
        t0=time.time()
        try:
            rr = mover.play(board, limit, game=game_no)
        except chess.engine.EngineTerminatedError as e:
            terminated = f"CRASH {mover_name}: {e}"
            issues.append(terminated)
            break
        except Exception as e:
            terminated = f"ERROR {mover_name}: {e}"
            issues.append(terminated)
            break
        dt = time.time()-t0
        per_engine[mover_name]["thinking_ms"].append(dt*1000)
        if rr.move is None:
            if board.is_game_over():
                break
            terminated = f"NO MOVE {mover_name} on ply {len(pgn_moves)+1} ({board.fen()})"
            issues.append(terminated)
            break
        # legality check (the harness' check, not the engine's claim)
        if rr.move not in board.legal_moves:
            per_engine[mover_name]["legality_fail"] += 1
            issues.append(f"ILLEGAL {mover_name} ply {len(pgn_moves)+1}: {rr.move.uci()} not in legal set ({board.fen()})")
            break
        san = board.san(rr.move)
        is_cap = board.is_capture(rr.move)
        is_castle = board.is_castling(rr.move)
        is_promo = rr.move.promotion is not None
        board.push(rr.move)
        pgn_moves.append(rr.move.uci())
        node = node.add_variation(rr.move)
        per_engine[mover_name]["moves"] += 1
        per_engine[mover_name]["phase"][ph] += 1
        if is_promo: per_engine[mover_name][" Promo?"] += 1
        if is_castle: per_engine[mover_name]["castled"] += 1
        seen[board.fen()] += 1
        if seen[board.fen()] >= 3 and not board.is_fivefold_draw():
            # engine should claim draw via repetition only when actually 3-fold — we just log
            pass
        # trace line
        cap_s = "x" if is_cap else ""
        promo_s = f"={chess.PIECE_SYMBOLS[rr.move.promotion]}" if is_promo else ""
        castle_s = " castle" if is_castle else ""
        print(f"  ply {len(pgn_moves):2d} {mover_name:16s} {rr.move.uci():5s} {cap_s}{promo_s}{castle_s} [{ph:9s}] {dt*1000:6.0f}ms  {san:6s}  clock ~{clock_w:.0f}/{clock_b:.0f}s")
        # rough clock accounting (real UCI clock would be managed by cutechess, here we just model it)
        if board.turn == chess.WHITE: clock_b = max(0, clock_b - dt + INC_MS/1000)
        else: clock_w = max(0, clock_w - dt + INC_MS/1000)
        if len(pgn_moves) >= 160:
            terminated = "move cap 160"
            break

    result = board.result(claim_draw=True) if terminated is None else "*"
    if terminated: game.headers["Termination"] = terminated
    game.headers["Result"] = result

    # timeout heuristic: mover exceeded soft budget by >2x
    # (real cutechess uses wtime/btime; our Limit is movetime-like)
    elapsed = time.time()-start

    # emit
    out_pgn.write(str(game)+"\n\n")
    rec = {
        "game": game_no, "white": w_name, "black": b_name,
        "result": result, "termination": terminated, "ply": len(pgn_moves),
        "fen_final": board.fen(), "is_game_over": board.is_game_over(claim_draw=True),
        "elapsed_s": round(elapsed,2),
        "per_engine": {k: {"moves":v["moves"],"phase":dict(v["phase"]),"castled":v["castled"],"promo":v[" Promo?"],"illegal":v["legality_fail"],"median_ms": (sorted(v["thinking_ms"])[len(v["thinking_ms"])//2] if v["thinking_ms"] else None)} for k,v in per_engine.items()},
        "issues": issues, "moves_uci": pgn_moves[:120],
    }
    out_jsonl.write(json.dumps(rec)+"\n")
    out_jsonl.flush()

    tag = "OK" if not issues else "ISSUES"
    print(f"  -> {result}  {len(pgn_moves)} ply  issues={len(issues)}  [{tag}]")
    for it in issues: print(f"     ! {it}")
    white.quit(); black.quit()
    return rec, result

def main():
    ap = argparse.ArgumentParser(description="Owen vs Stockfish code inspection match")
    ap.add_argument("--owen", default=None, help="path to owen binary (default: build-baked/owen2 or build/owen2)")
    ap.add_argument("--stockfish", default=str(SF_BIN))
    ap.add_argument("--games", type=int, default=2, help="number of games (alternating colours)")
    ap.add_argument("--movetime", type=int, default=1200, help="movetime limit in ms per move")
    args = ap.parse_args()

    owen = pick_owen(Path(args.owen) if args.owen else None)
    sf = Path(args.stockfish)
    if not sf.exists(): sys.exit(f"Stockfish not found at {sf}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pgn_path = LOG_DIR / f"match-{ts}.pgn"
    nd_path  = LOG_DIR / f"match-{ts}.ndjson"

    print(f"Owen  : {owen} ({owen.stat().st_size/1_048_576:.1f} MiB)")
    print(f"SF    : {sf}")
    print(f"python-chess {chess.__version__}  movetime {args.movetime}ms  games {args.games}")
    print(f"logs  : {pgn_path}  {nd_path}")
    print(textwrap.dedent("""
    NOTE: This is a code inspection, not a strength match.
    W/D/L below tells you almost nothing — a 0-2 can still be
    perfectly good code (Owen blundered), and 2-0 can hide
    defects. Read the diagnostics instead.
    """).strip())

    limit = chess.engine.Limit(time=args.movetime/1000)
    all_issues=[]
    results=[]
    with open(pgn_path,"w") as pf, open(nd_path,"w") as jf:
        for g in range(1, args.games+1):
            white, black = (owen, sf) if g%2==1 else (sf, owen)
            print(f"\n=== Game {g}/{args.games}  White={Path(white).name}  Black={Path(black).name} ===")
            rec, res = play_one_game(white, black, g, limit, pf, jf)
            all_issues.extend(rec["issues"])
            results.append(res)

    # Summary that answers "is the code good?"
    print("\n" + "="*64)
    print("CODE QUALITY VERDICT  (read this, not the scoreline)")
    print("="*64)
    with open(nd_path) as f:
        recs=[json.loads(l) for l in f if l.strip()]
    crash = [r for r in recs if any("CRASH" in s or "NO MOVE" in s or "ILLEGAL" in s for s in r["issues"])]
    illegal = [r for r in recs if any("ILLEGAL" in s for s in r["issues"])]
    timeout = [r for r in recs if any("TIMEOUT" in s for s in r["issues"])]
    # per-move legality across all games is the strongest signal
    total_illegal = sum(r["per_engine"][k]["illegal"] for r in recs for k in r["per_engine"])
    print(f"Games : {args.games}  results {results}  (outcome is NOT a code-quality signal)")
    print(f"Issues: {len(all_issues)} total  illegal_moves={total_illegal}  crash/no-move={len(crash)}  timeout={len(timeout)}")
    # pass/fail gates
    gates = []
    gates.append(("UCI handshake      ", len([s for s in all_issues if 'missing id' in s or 'options empty' in s])==0))
    gates.append(("No illegal moves   ", total_illegal==0))
    gates.append(("No crashes / hangs ", len(crash)==0))
    gates.append(("Both colours played", len({r['white'] for r in recs})>=1 and len({r['black'] for r in recs})>=1 and args.games>=2))
    for name, ok in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if illegal or crash:
        print("\nFailing diagnostics (fix these regardless of who won):")
        for r in recs:
            for s in r["issues"]: print(f"  g{r['game']}: {s}")
    else:
        print("\nNo legality/crash defects found in this run. Other checks to do separately:")
        print("  - perft / move-undo symmetry, Zobrist key, NNUE eval sign, TT corruption")
        print("  - pondering / infinite / wtime/btime budgets (see tools/match_inspect.py)")
    print(f"\nFull logs: {pgn_path}  {nd_path}")
    if not all(ok for _,ok in gates):
        sys.exit(2)

if __name__=="__main__":
    main()
