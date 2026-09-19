#!/usr/bin/env python3
"""
Owen 2 — Distill from Stockfish 18 AVX2 without copying
Teacher: ~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2 labels positions via UCI.
Student: Owen 1024 Threat net — learns eval numbers, not weights.

Usage:
  python3 trainer/distill.py --sdata data/sdata-combined-v1.bin --out data/sdata-distilled.bin --stockfish ~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2 --depth 12 --threads 8
  # then train:
  python3 trainer/train.py --sdata data/sdata-distilled.bin --out nets/o2-distilled.o2nn --device cuda

DeepSeek tricks for teacher: parallel UCI engines (8x) + batched FENs, no GPU needed for SF (CPU AVX2).
"""
import argparse, os, struct, subprocess, threading, queue, time

RECORD_SIZE_V1 = 69
RECORD_SIZE_V2 = 71
RECORD_SIZE_V3 = 73
RECORD_SIZE = 69
SF_DEFAULT = os.path.expanduser("~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2")

def detect_record_size(path):
    """Pick the record size whose records are content-valid (board bytes 0..12,
    stm in {0,1}). Purely size-based detection is ambiguous when a file size is
    divisible by several of {69,71,73} (69*73 has many common multiples)."""
    sz = os.path.getsize(path)
    import struct as _s
    def valid(R):
        if not sz or sz % R != 0:
            return False
        n = min(sz // R, 2000)
        bad = 0
        with open(path, "rb") as f:
            for _ in range(n):
                rec = f.read(R)
                if len(rec) < R:
                    break
                if not all(0 <= b <= 12 for b in rec[0:64]) or rec[64] not in (0, 1):
                    bad += 1
                    if bad > n // 20:
                        return False
        return bad <= n // 20
    for R in (RECORD_SIZE_V3, RECORD_SIZE_V2, RECORD_SIZE_V1):
        if valid(R):
            return R
    return RECORD_SIZE_V1

def unpack_distill_record(b):
    """Return (board_bytes, stm, ply, castling, ep). Handles v1 (69B), v2 (71B), v3 (73B)."""
    board = b[0:64]
    stm = b[64]
    ply = b[68]
    if len(b) >= 73:
        castling = b[69]
        ep = b[70]
    elif len(b) >= 71:
        castling = b[69]
        ep = b[70]
    else:
        castling, ep = 0, 64
    return board, stm, ply, castling, ep

def board_to_fen(board, stm, ply=0, castling=0, ep=64):
    """board 64 uint8 0..11/12 -> fen. stm 0 white 1 black.
    v2 passes real castling/EP; v1 legacy falls back to '-' (biased, prefer v2)."""
    piece_map = {0:'P',1:'N',2:'B',3:'R',4:'Q',5:'K',6:'p',7:'n',8:'b',9:'r',10:'q',11:'k'}
    rows=[]
    for r in range(7,-1,-1):
        empty=0; row=""
        for f in range(8):
            sq=r*8+f
            p=board[sq]
            if p==12:
                empty+=1
            else:
                if empty: row+=str(empty); empty=0
                row+=piece_map.get(int(p),'?')
        if empty: row+=str(empty)
        rows.append(row)
    stm_c = 'w' if stm==0 else 'b'
    cs = ""
    if castling & 1: cs += "K"
    if castling & 2: cs += "Q"
    if castling & 4: cs += "k"
    if castling & 8: cs += "q"
    if not cs: cs = "-"
    if ep is None or ep >= 64:
        eps = "-"
    else:
        eps = chr(ord('a') + (int(ep) % 8)) + str(int(ep) // 8 + 1)
    return f"{'/'.join(rows)} {stm_c} {cs} {eps} 0 {ply+1}"

def eval_with_sf(engine_path, fen, depth, hash_mb=16):
    """One UCI eval via subprocess per position — simple but parallelized by workers."""
    # persistent engine per worker would be faster; we do per-call for simplicity here
    # caller should use worker pool below
    pass

class SFWorker:
    """Persistent Stockfish UCI process for fast evals. Auto-restarts on crash."""
    def __init__(self, path, depth, hash_mb=32):
        self.path=path; self.depth=depth; self.hash_mb=hash_mb
        self.lock = threading.Lock()
        self._spawn()

    def _spawn(self):
        self.proc = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value 1")
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd+"\n")
        self.proc.stdin.flush()

    def _wait_for(self, token, timeout=10):
        deadline=time.time()+timeout
        while time.time() < deadline:
            # poll so we don't block forever if SF died
            line=self.proc.stdout.readline()
            if not line:
                # EOF — SF exited
                try: self.proc.wait(timeout=1)
                except: pass
                raise RuntimeError(f"SF died waiting for {token}")
            if token in line:
                return
        raise RuntimeError(f"SF timeout waiting for {token}")
    def eval_fen(self, fen):
        last_err=None
        for attempt in range(3):
            try:
                with self.lock:
                    return self._eval_fen_inner(fen)
            except RuntimeError as e:
                last_err=e
                # restart the process
                try: self.proc.kill()
                except: pass
                try: self.proc.wait(timeout=2)
                except: pass
                time.sleep(0.2 * (attempt+1))
                try:
                    self._spawn()
                except Exception as se:
                    last_err=se
                    continue
                continue
        raise RuntimeError(f"SF eval failed after 3 attempts: {fen[:60]} — {last_err}")

    def _eval_fen_inner(self, fen):
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        best=None; score=None
        deadline=time.time()+30  # per-position timeout (depth 9 should be <1s)
        while time.time() < deadline:
            line=self.proc.stdout.readline()
            if not line:
                raise RuntimeError("SF died during search")
            if line.startswith("info") and "score cp" in line:
                try:
                    idx=line.split().index("cp")
                    score=int(line.split()[idx+1])
                except: pass
            elif "score mate" in line:
                # mate score — clamp to window
                try:
                    idx=line.split().index("mate")
                    mate=int(line.split()[idx+1])
                    score= 10000 if mate>0 else -10000
                except: pass
            if line.startswith("bestmove"):
                if score is None:
                    if "mate" in line:
                        score=10000
                    else:
                        score=0
                break
        else:
            # timeout — kill and restart will happen on next call
            try: self.proc.kill()
            except: pass
            raise RuntimeError("SF timeout on position")
        return max(-15000, min(15000, score))
    def quit(self):
        try:
            self._send("quit")
            try: self.proc.wait(timeout=2)
            except: self.proc.kill()
        except: pass

def iter_records(path, record_size=None):
    if record_size is None:
        record_size = detect_record_size(path)
    n=os.path.getsize(path)//record_size
    with open(path,"rb") as f:
        for _ in range(n):
            b=f.read(record_size)
            if len(b)<record_size: break
            yield b

def distill(args):
    import concurrent.futures
    sdata=args.sdata
    out=args.out
    rs = detect_record_size(sdata)
    n=os.path.getsize(sdata)//rs
    tag = "v3" if rs==73 else ("v2" if rs==71 else "v1-legacy")
    print(f"Distill: {n} positions from {sdata} (record {rs}B {tag}) via {args.stockfish} depth {args.depth} threads {args.threads}", flush=True)

    out_exists=False
    already=0
    if os.path.exists(out):
        already=os.path.getsize(out)//rs
        if 0 < already < n:
            print(f" Resume: {already}/{n} already in {out} — will append rest", flush=True)
            out_exists=True

    if out_exists:
        skip=already
    else:
        skip=0
    to_do=n-skip
    print(f" To process: {to_do} records (skipped {skip})", flush=True)
    if to_do <= 0:
        print("Already complete.")
        return

    # Start workers
    workers=[SFWorker(args.stockfish, args.depth, hash_mb=args.hash) for _ in range(args.threads)]
    print(f" Started {len(workers)} SF workers", flush=True)

    def eval_one(args_tuple):
        idx, b = args_tuple
        board, stm, ply, castling, ep = unpack_distill_record(b)
        import numpy as np
        brd=np.frombuffer(board, dtype=np.uint8)
        fen=board_to_fen(brd, stm, ply, castling, ep)
        w = workers[idx % len(workers)]
        cp=w.eval_fen(fen)
        new = bytearray(b)
        struct.pack_into("<h", new, 65, int(cp))
        return idx, bytes(new)

    done_checkpoint=0; start=time.time(); failed=[]
    if out_exists:
        mode="ab"
    else:
        mode="wb"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # need os.stat-based counting after resume? use n-skip as total
    need_write = to_do

    # Streaming: keep only a bounded window of in-flight futures so memory
    # stays flat regardless of dataset size. Results are written in order.
    window = max(args.threads * 4, 64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex, \
         open(out, mode) as f:
        it = iter_records(sdata)
        for _ in range(skip):
            next(it, None)
        in_flight = {}   # fut -> (idx, b)
        pending = {}     # idx -> bytes  (completed but out of order, keyed for ordered write)
        next_in = skip   # next index to submit
        next_out = skip  # next index to write

        def submit_more():
            nonlocal next_in
            while len(in_flight) < window and next_in < n:
                b = next(it, None)
                if b is None:
                    next_in = n
                    break
                f_ = ex.submit(eval_one, (next_in, b))
                in_flight[f_] = (next_in, b)
                next_in += 1

        submit_more()
        written = 0
        while in_flight:
            fut = next(concurrent.futures.as_completed(in_flight))
            idx, b = in_flight[fut]
            try:
                _idx, nb = fut.result()
                pending[_idx] = nb
            except Exception as e:
                print(f" WARN eval {idx} failed: {e}", flush=True)
                failed.append(idx)
                nb = bytearray(b); struct.pack_into("<h", nb, 65, 0)
                pending[idx] = bytes(nb)
            del in_flight[fut]
            # drain ordered results to file
            while next_out in pending:
                f.write(pending.pop(next_out))
                next_out += 1
                written += 1
                done=written
                if done % 500 == 0 or done==(n-skip):
                    elapsed=time.time()-start
                    rate=done/elapsed if elapsed>0 else 0
                    eta=(need_write-done)/rate if rate>0 else 0
                    print(f" [{skip+done}/{n}] {rate:.1f} pos/s  eta {eta/60:.1f}m  failed {len(failed)}", flush=True)
            submit_more()

    for w in workers: w.quit()
    sz=os.path.getsize(out)
    print(f"Done -> {out}  {sz} bytes  {sz//rs} positions  failed {len(failed)}")
    if failed:
        print(f" {len(failed)} positions had fallback cp=0 (see WARNs above)")
    print(f"Now train: python3 trainer/train.py --sdata {out} --out nets/o2-distilled.o2nn --device cuda")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--sdata", required=True, help="input sdata (handcrafted or self-play)")
    ap.add_argument("--out", required=True, help="output distilled sdata")
    ap.add_argument("--stockfish", default=SF_DEFAULT)
    ap.add_argument("--depth", type=int, default=12, help="SF depth for labels (12 fast, 16 stronger)")
    ap.add_argument("--threads", type=int, default=8, help="parallel SF workers (1 thread each, AVX2)")
    ap.add_argument("--hash", type=int, default=32, help="Hash MB per worker")
    args=ap.parse_args()
    distill(args)
