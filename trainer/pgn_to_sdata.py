#!/usr/bin/env python3
"""
Owen 2 — PGN -> sdata (69-byte) using python-chess
Filters for classical if needed, samples every N plies to keep file small.
12 threads via ProcessPool (chess is Python, so processes not threads).

Usage:
  python3 trainer/pgn_to_sdata.py --pgn lichess.pgn --out data/lichess-10M.bin --max 10000000 --threads 12
  python3 trainer/pgn_to_sdata.py --pgn lichess_db_*.pgn.zst --out data/classical-10M.bin --threads 12 --max 10000000 --min-ply 60
  python3 trainer/pgn_to_sdata.py --pgn lichess.pgn.zst --out data/lichess.bin --threads 12 --sample-every 4
"""
import argparse, os, struct, gzip, io, sys, time, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RECORD_SIZE = 71  # v2: 64B board + stm + eval(i16) + result + ply + castling + ep
RECORD_FMT_V2 = "<64B B h B B B B"
PIECE_MAP = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

def board_castling_ep(board):
    """Extract UCI-style castling bits (K=1,Q=2,k=4,q=8) and ep square (0..63, 64 none)."""
    c = 0
    # python-chess: board.has_kingside_castling_rights(chess.WHITE) etc.
    try:
        import chess
        if board.has_kingside_castling_rights(chess.WHITE): c |= 1
        if board.has_queenside_castling_rights(chess.WHITE): c |= 2
        if board.has_kingside_castling_rights(chess.BLACK): c |= 4
        if board.has_queenside_castling_rights(chess.BLACK): c |= 8
        ep = board.ep_square if board.ep_square is not None else 64
        # python-chess square 0=a1..63=h8 matches our make_square(file,rank)
        if ep is None: ep = 64
    except Exception:
        c, ep = 0, 64
    return c, int(ep)

def board_to_record(board, result_cp=0):
    # board is chess.Board
    arr = bytearray(64)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            arr[sq] = 12
        else:
            sym = piece.symbol()
            arr[sq] = PIECE_MAP[sym]
    stm = 0 if board.turn else 1  # chess: True white
    # result from stm view
    result_str = board.headers.get("Result","*") if hasattr(board,"headers") else "*"
    # we encode result as 0 loss 1 draw 2 win from stm view — need game result
    # We'll pass result via outer scope, here just placeholder; caller fixes
    return bytes(arr), stm

def game_result_to_stm(result_str, stm):
    # result_str "1-0" "0-1" "1/2-1/2"
    if result_str == "1-0":
        return 2 if stm==0 else 0
    if result_str == "0-1":
        return 2 if stm==1 else 0
    return 1

def open_pgn(path):
    if str(path).endswith(".zst"):
        import zstandard
        fh = open(path, 'rb')
        dctx = zstandard.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding='utf-8', errors='ignore'), fh
    elif str(path).endswith(".gz"):
        return gzip.open(path, 'rt', encoding='utf-8', errors='ignore'), None
    else:
        return open(path, 'r', encoding='utf-8', errors='ignore'), None

def process_chunk(args):
    pgn_path, chunk_idx, chunk_size, sample_every, min_ply, elo_min = args
    import chess.pgn
    import chess
    f, fh2 = open_pgn(pgn_path)
    # seek to chunk start by skipping games
    count=0; written=0; out_buf=bytearray()
    try:
        # skip chunk_idx * chunk_size games
        for _ in range(chunk_idx * chunk_size):
            g = chess.pgn.read_game(f)
            if g is None: break
        for _ in range(chunk_size):
            g = chess.pgn.read_game(f)
            if g is None: break
            # filter by elo if set
            if elo_min:
                try:
                    we = int(g.headers.get("WhiteElo","0") or 0)
                    be = int(g.headers.get("BlackElo","0") or 0)
                    if min(we,be) < elo_min: continue
                except: pass
            # filter time control for classical
            tc = g.headers.get("TimeControl","-")
            # allow "-" (correspondence) or numeric >= 900 for classical
            result_str = g.headers.get("Result","*")
            if result_str not in ("1-0","0-1","1/2-1/2"): continue
            board = g.board()
            ply=0
            for move in g.mainline_moves():
                board.push(move)
                ply+=1
                if ply < min_ply: continue
                if ply % sample_every != 0: continue
                # encode
                arr = bytearray(64)
                for sq in range(64):
                    pc = board.piece_at(sq)
                    arr[sq] = PIECE_MAP[pc.symbol()] if pc else 12
                stm = 0 if board.turn else 1
                res = game_result_to_stm(result_str, stm)
                # eval placeholder 0 — will be overwritten by distill via SF
                castling, ep = board_castling_ep(board)
                rec = struct.pack(RECORD_FMT_V2, *arr, stm, 0, res, ply & 0xFF, castling & 0xFF, ep & 0xFF)
                out_buf.extend(rec)
                written+=1
    except Exception as e:
        print(f"chunk {chunk_idx} error {e}", file=sys.stderr)
    finally:
        try: f.close();
        except: pass
        if fh2:
            try: fh2.close()
            except: pass
    return chunk_idx, bytes(out_buf), written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="path to .pgn or .pgn.zst (supports zstd)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=10000000, dest="max_pos", help="max positions")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--sample-every", type=int, default=1, help="keep every Nth ply (2 = half)")
    ap.add_argument("--min-ply", type=int, default=8, help="skip opening 8 plies")
    ap.add_argument("--elo-min", type=int, default=2000, help="min WhiteElo/BlackElo, 0 to disable")
    ap.add_argument("--chunk-games", type=int, default=5000)
    args = ap.parse_args()

    pgn = Path(args.pgn)
    if not pgn.exists():
        # try glob
        import glob
        matches = glob.glob(args.pgn)
        if matches:
            pgn = Path(matches[0])
        else:
            print(f"not found: {args.pgn}", file=sys.stderr); sys.exit(1)

    # count games quickly for chunking
    print(f"PGN: {pgn} -> {args.out}  max {args.max_pos}  threads {args.threads}  sample {args.sample_every}  elo>={args.elo_min}  minply {args.min_ply}")
    # estimate chunks needed: each game ~40 plies sampled -> ~40 pos, so 10M needs 250k games
    est_games = (args.max_pos * args.sample_every) // 35 + 1000
    n_chunks = max(1, (est_games + args.chunk_games -1)//args.chunk_games)
    # cap chunks to threads*8 to avoid huge overhead on tiny files
    if n_chunks > args.threads * 32:
        args.chunk_games = max(1000, est_games // (args.threads*16))
        n_chunks = (est_games + args.chunk_games -1)//args.chunk_games
    print(f" Spawning {n_chunks} chunks x{args.chunk_games} games (est {est_games} games)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    start=time.time()
    total=0
    out_f = open(args.out, 'wb')
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        chunk_args = [(str(pgn), i, args.chunk_games, args.sample_every, args.min_ply, args.elo_min) for i in range(n_chunks)]
        for idx, buf, written in ex.map(process_chunk, chunk_args):
            if total + len(buf)//RECORD_SIZE > args.max_pos:
                # trim
                keep = args.max_pos - total
                buf = buf[:keep*RECORD_SIZE]
                out_f.write(buf)
                total += keep
                print(f" [{total}/{args.max_pos}] chunk {idx} +{written} -> hit max, stopping")
                break
            out_f.write(buf)
            total += len(buf)//RECORD_SIZE
            elapsed=time.time()-start
            rate=total/elapsed if elapsed>0 else 0
            print(f" [{total}/{args.max_pos}] chunk {idx} +{written}  {rate:.0f} pos/s  elapsed {elapsed/60:.1f}m")
            if total >= args.max_pos:
                break
    out_f.close()
    sz=os.path.getsize(args.out)
    print(f"Done: {args.out}  {sz} bytes  {sz//RECORD_SIZE} positions  {time.time()-start:.1f}s")
    print(f"Next: python3 trainer/distill.py --sdata {args.out} --out {args.out.replace('.bin','-distilled.bin')} --stockfish ~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2 --depth 12 --threads 8")
    print(f"      python3 trainer/train.py --sdata <distilled> --out nets/o2-lichess.o2nn --device cuda")

if __name__=="__main__":
    main()
