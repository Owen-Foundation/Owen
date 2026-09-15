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
RECORD_SIZE = 69
SF_DEFAULT = os.path.expanduser("~/Videos/stockfish/stockfish-ubuntu-x86-64-avx2")

def detect_record_size(path):
    sz = os.path.getsize(path)
    if sz % RECORD_SIZE_V2 == 0 and sz % RECORD_SIZE_V1 != 0:
        return RECORD_SIZE_V2
    return RECORD_SIZE_V1

def unpack_distill_record(b):
    """Return (board_bytes, stm, ply, castling, ep). Handles v1 (69B) and v2 (71B)."""
    board = b[0:64]
    stm = b[64]
    ply = b[68]
    if len(b) >= 71:
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
    print(f"Distill: {n} positions from {sdata} (record {rs}B {'v2' if rs==71 else 'v1-legacy'}) via {args.stockfish} depth {args.depth} threads {args.threads}")

    out_exists=False
    already=0
    if os.path.exists(out):
        already=os.path.getsize(out)//rs
        if 0 < already < n:
            print(f" Resume: {already}/{n} already in {out} — will append rest")
            out_exists=True

    # Load needed records — stream if resuming: skip already-done prefix
    if out_exists:
        skip=already
    else:
        skip=0
    records=[]
    for i, b in enumerate(iter_records(sdata)):
        if i < skip: continue
        records.append(b)
    print(f" To process: {len(records)} records (skipped {skip})")
    if not records and out_exists:
        print("Already complete.")
        return

    # Start workers
    workers=[SFWorker(args.stockfish, args.depth, hash_mb=args.hash) for _ in range(args.threads)]
    print(f" Started {len(workers)} SF18 workers")

    out_records=[None]*len(records)

    def eval_one(i):
        b=records[i]
        board, stm, ply, castling, ep = unpack_distill_record(b)
        import numpy as np
        brd=np.frombuffer(board, dtype=np.uint8)
        fen=board_to_fen(brd, stm, ply, castling, ep)
        w = workers[i % len(workers)]
        cp=w.eval_fen(fen)
        new = bytearray(b)
        struct.pack_into("<h", new, 65, int(cp))
        return i, bytes(new)

    done=0; start=time.time()
    failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs={ex.submit(eval_one,i):i for i in range(len(records))}
        for fut in concurrent.futures.as_completed(futs):
            try:
                i, nb = fut.result()
            except Exception as e:
                idx=futs[fut]
                print(f" WARN eval {skip+idx} failed: {e}", flush=True)
                failed.append((skip+idx, str(e)))
                # write a neutral record so length stays correct
                b=records[idx]
                new=bytearray(b); struct.pack_into("<h", new, 65, 0)
                out_records[idx]=bytes(new)
                done+=1
                continue
            out_records[i]=nb
            done+=1
            if done % 500 == 0 or done==len(records):
                elapsed=time.time()-start
                rate=done/elapsed if elapsed>0 else 0
                eta=(len(records)-done)/rate if rate>0 else 0
                print(f" [{skip+done}/{n}] {rate:.1f} pos/s  eta {eta/60:.1f}m  failed {len(failed)}", flush=True)
                # checkpoint every 20k
                if done % 20000 == 0:
                    mode="ab" if out_exists or done>20000 else "wb"
                    if done==20000 and not out_exists:
                        mode="wb"
                    else:
                        mode="ab" if os.path.exists(out) and already>0 else ("ab" if done>20000 else "wb")
                    # we checkpoint differently: on first flush write all so far, then append
                    pass

    for w in workers: w.quit()

    mode="ab" if out_exists else "wb"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out,mode) as f:
        for r in out_records:
            f.write(r if r is not None else records[len(failed)]*0)
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
