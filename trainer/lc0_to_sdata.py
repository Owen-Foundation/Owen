#!/usr/bin/env python3
"""
Owen 2 — Lc0 training chunks (V3/V4/V5/V6) -> sdata v2 (71B).

Lc0 publishes open self-play data (ODbL, Database Contents License) at
  https://storage.lczero.org/files/training_data/
Stockfish itself trains on Lc0 data (SF19 rescored hundreds of billions of
positions with a Leela net). This script converts those chunks into Owen's
sdata v2 format so trainer/train.py can learn from them with no Stockfish
teacher involved.

Mapping (verified against lc0 src/neural/encoder.cc, trainingdata.cc,
trainingdata_v6.h and lczero-training chunkparser.py):
  - planes[104] = 8 history steps x 13 planes. planes[0..11] are the CURRENT
    position: ours {P,N,B,R,Q,K}, theirs {P,N,B,R,Q,K} + repetition flag.
    "ours" = side to move, squares absolute (no flip for current position).
    Stored bytes are ReverseBitsInBytes(canonical) for gzip; combined with
    MSB-first unpacking this cancels out: unpacked[plane*64 + s] == square s
    (a1 = bit 0, same as Owen's make_square).
  - side to move: classical formats -> stm byte (0 white, 1 black);
    canonical input_format>=3 -> bit 7 of invariance_info.
  - castling us_ooo/us_oo/them_ooo/them_oo are side-to-move-relative;
    mapped to absolute KQkq (nonzero = right present).
  - result: STM perspective. V6 result_q float; V3/V4/V5 int8 (-1/0/1).
    Owen result: 2 win / 1 draw / 0 loss (STM view).
  - eval label: best_q (fallback root_q, then orig_q) via Lc0's own
    cp formula: cp = 111.714640472 * tan(1.562068842 * q), clipped.
    V3 has no Q -> eval 0 (result-only, still trainable).
  - EP is not stored in classical chunks -> ep = 64 (none). Harmless:
    Owen's HalfKP features don't use EP; EP only mattered for SF-distill
    FEN reconstruction, which this path doesn't need.

Usage:
  # one tar (streamed, never fully extracted):
  python3 trainer/lc0_to_sdata.py --in data/training-run1--20230505-0917.tar \\
      --out data/lc0-1M.bin --max 1000000
  # single chunk / directory of chunks:
  python3 trainer/lc0_to_sdata.py --in chunk.gz --out data/lc0.bin --max 500000
  # then:
  python3 trainer/train.py --sdata data/lc0-1M.bin --out nets/o2-lc0.o2nn --device cuda

License note: Lc0 training data is ODbL. Training on it is what Stockfish
does; your resulting .o2nn weights are your artifact.
"""
import argparse
import gzip
import math
import os
import struct
import sys
import tarfile

import numpy as np

# (record_size, has_input_format, off_probs, off_planes, off_castle,
#  off_stm, off_rule50, off_aux, off_result_i8, off_root_q, off_best_q,
#  off_result_q, off_orig_q, off_visits, off_invariance)
_FMT = {
    3: dict(size=8276, has_fmt=False, probs=4, planes=7436, castle=8268,
             stm=8272, rule50=8273, aux=8274, ri8=8275,
             root_q=None, best_q=None, result_q=None, orig_q=None,
             visits=None, inv=None),
    4: dict(size=8292, has_fmt=False, probs=4, planes=7436, castle=8268,
             stm=8272, rule50=8273, aux=8274, ri8=8275,
             root_q=8276, best_q=8280, result_q=None, orig_q=None,
             visits=None, inv=None),
    5: dict(size=8308, has_fmt=True, probs=8, planes=7440, castle=8272,
             stm=8276, rule50=8277, aux=8278, ri8=8279,
             root_q=8280, best_q=8284, result_q=None, orig_q=None,
             visits=None, inv=8278),
    6: dict(size=8356, has_fmt=True, probs=8, planes=7440, castle=8272,
             stm=8276, rule50=8277, aux=8278, ri8=None,
             root_q=8280, best_q=8284, result_q=8308, orig_q=8328,
             visits=8340, inv=8278),
}

SDATA_V2 = 71
# Owen piece codes: 0..5 white P N B R Q K, 6..11 black, 12 empty
PT_ORDER = (0, 1, 2, 3, 4, 5)  # plane order per side is already P..K

LC0_CP_SCALE = 111.714640472
LC0_CP_K = 1.562068842


def q_to_cp(q):
    try:
        q = float(q)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(q):
        return 0
    q = max(-1.0, min(1.0, q))
    try:
        cp = LC0_CP_SCALE * math.tan(LC0_CP_K * q)
    except (OverflowError, ValueError):
        cp = 15000.0 if q >= 0 else -15000.0
    if not math.isfinite(cp):
        cp = 15000.0 if q >= 0 else -15000.0
    return int(max(-15000, min(15000, round(cp))))


def inv_transform_plane(bits8x8, transform):
    """Inverse of Lc0's canonical transform on an 8x8 [rank,file] bit plane.
    Forward order in encoder: flip(files) -> mirror(ranks) -> transpose.
    Inverse: transpose -> mirror -> flip."""
    b = bits8x8
    if transform & 4:  # transpose
        b = b.T.copy()
    if transform & 2:  # mirror (ranks)
        b = b[::-1, :].copy()
    if transform & 1:  # flip (files)
        b = b[:, ::-1].copy()
    return b


def decode_batch(buf, fmt, want_transform_check=True):
    """Decode one file's bytes -> (boards[N,64] uint8 Owen codes, stms, castlings,
    results(0/1/2), evals_cp, plys, stats). Boards use 12=empty."""
    n = len(buf) // fmt["size"]
    if n == 0:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8, count=n * fmt["size"]).reshape(n, fmt["size"])
    planes_b = arr[:, fmt["planes"]:fmt["planes"] + 832].reshape(n, 104, 8)
    # MSB-first unpack cancels Lc0's ReverseBitsInBytes: bit s == square s.
    bits = np.unpackbits(planes_b, axis=2).reshape(n, 104, 64)

    if fmt["has_fmt"]:
        input_fmt = np.frombuffer(buf, dtype=np.int32, count=n * (fmt["size"] // 4)).reshape(n, -1)[:, 1]
    else:
        input_fmt = np.ones(n, dtype=np.int64)

    if fmt["inv"] is not None:
        inv = arr[:, fmt["inv"]].astype(np.int64)
    else:
        inv = np.zeros(n, dtype=np.int64)
    canonical = input_fmt >= 3
    if fmt["has_fmt"]:
        stm = np.where(canonical, (inv >> 7) & 1, arr[:, fmt["stm"]].astype(np.int64))
    else:
        stm = arr[:, fmt["stm"]].astype(np.int64)

    us_ooo = arr[:, fmt["castle"]].astype(np.int64)
    us_oo = arr[:, fmt["castle"] + 1].astype(np.int64)
    them_ooo = arr[:, fmt["castle"] + 2].astype(np.int64)
    them_oo = arr[:, fmt["castle"] + 3].astype(np.int64)

    # result (STM view): V6 result_q else int8
    if fmt["result_q"] is not None:
        rq = np.frombuffer(buf, dtype=np.float32,
                           count=n * (fmt["size"] // 4)).reshape(n, -1)[:,
            fmt["result_q"] // 4].astype(np.float64)
        rq = np.nan_to_num(rq, nan=0.0, posinf=1.0, neginf=-1.0)
        res = np.where(rq >= 0.5, 2, np.where(rq <= -0.5, 0, 1))
    else:
        ri8 = arr[:, fmt["ri8"]].astype(np.int8).astype(np.int64)
        res = np.where(ri8 >= 1, 2, np.where(ri8 <= -1, 0, 1))

    # eval label: best_q -> root_q -> orig_q
    f32 = np.frombuffer(buf, dtype=np.float32,
                        count=n * (fmt["size"] // 4)).reshape(n, -1)
    q = np.full(n, np.nan)
    if fmt["best_q"] is not None:
        q = f32[:, fmt["best_q"] // 4].astype(np.float64)
    if fmt["root_q"] is not None:
        r = f32[:, fmt["root_q"] // 4].astype(np.float64)
        q = np.where(np.isfinite(q), q, r)
    if fmt["orig_q"] is not None:
        o = f32[:, fmt["orig_q"] // 4].astype(np.float64)
        q = np.where(np.isfinite(q), q, np.nan_to_num(o, nan=np.nan))
    q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
    q = np.clip(q, -1.0, 1.0)
    with np.errstate(over='ignore', invalid='ignore'):
        cp = LC0_CP_SCALE * np.tan(LC0_CP_K * q)
    cp = np.nan_to_num(cp, nan=0.0, posinf=15000.0, neginf=-15000.0)
    evals = np.clip(np.rint(cp), -15000, 15000).astype(np.int64)

    if fmt["size"] in (8276, 8292):
        plys = arr[:, fmt["aux"]].astype(np.int64)  # move_count
    else:
        plys = np.zeros(n, dtype=np.int64)

    return dict(bits=bits, stm=stm, input_fmt=input_fmt, inv=inv,
                canonical=canonical, us_ooo=us_ooo, us_oo=us_oo,
                them_ooo=them_ooo, them_oo=them_oo, res=res, evals=evals,
                plys=plys, n=n)


def build_records(dec, sample_every=1, min_visits=0, visits=None):
    """Vectorized-ish build of sdata v2 records. Returns (bytes, stats)."""
    n = dec["n"]
    idx = np.arange(n)
    if sample_every > 1:
        idx = idx[idx % sample_every == 0]
    if min_visits > 0 and visits is not None:
        idx = idx[visits[idx] >= min_visits]
    if len(idx) == 0:
        return b"", dict(kept=0, skipped=0)

    bits = dec["bits"][idx]  # [m,104,64]
    stm = dec["stm"][idx]
    m = len(idx)
    boards = np.full((m, 64), 12, dtype=np.uint8)

    # current-position planes 0..11 (ours P..K, theirs P..K)
    cur = bits[:, :12, :]  # [m,12,64]
    if np.any(dec["canonical"][idx]):
        # inverse canonical transform per record (rare path, loop is fine)
        for i in range(m):
            if dec["canonical"][idx[i]]:
                t = int(dec["inv"][idx[i]]) & 7
                if t:
                    sq = cur[i, :12].reshape(12, 8, 8)
                    inv = np.empty_like(sq)
                    for p in range(12):
                        inv[p] = inv_transform_plane(sq[p], t)
                    cur[i, :12] = inv.reshape(12, 64)

    # overlap / sanity: each square at most one piece; exactly one king each
    occ = cur.sum(axis=1)  # [m,64]
    ok = (occ.max(axis=1) <= 1)
    ok &= (cur[:, 5, :].sum(axis=1) == 1) & (cur[:, 11, :].sum(axis=1) == 1)
    tot = cur[:, :12, :].sum(axis=(1, 2))
    ok &= (tot <= 32) & (tot >= 2)
    ok &= (dec["stm"][idx] <= 1)
    keep = np.where(ok)[0]
    skipped = m - len(keep)
    if len(keep) == 0:
        return b"", dict(kept=0, skipped=int(skipped))

    cur = cur[keep]
    stm_k = dec["stm"][idx][keep].astype(np.int64)
    sq_idx = np.arange(64)
    for p in range(12):
        side = p // 6  # 0 ours, 1 theirs
        pt = p % 6
        color_is_white = np.where(side == 0, stm_k == 0, stm_k == 1)
        code_w = pt
        code_b = pt + 6
        mask = cur[:, p, :] > 0
        rows, cols = np.where(mask)
        boards[rows, cols] = np.where(
            color_is_white[rows], code_w, code_b).astype(np.uint8)

    white_stm = (stm_k == 0)
    K = np.where(white_stm, dec["us_oo"][idx][keep] != 0,
                 dec["them_oo"][idx][keep] != 0).astype(np.uint8)
    Q = np.where(white_stm, dec["us_ooo"][idx][keep] != 0,
                 dec["them_ooo"][idx][keep] != 0).astype(np.uint8)
    k = np.where(white_stm, dec["them_oo"][idx][keep] != 0,
                 dec["us_oo"][idx][keep] != 0).astype(np.uint8)
    q = np.where(white_stm, dec["them_ooo"][idx][keep] != 0,
                 dec["us_ooo"][idx][keep] != 0).astype(np.uint8)
    castling = K | (Q << 1) | (k << 2) | (q << 3)

    out = bytearray()
    res = dec["res"][idx][keep].astype(np.uint8)
    ev = dec["evals"][idx][keep].astype(np.int16)
    ply = np.clip(dec["plys"][idx][keep], 0, 255).astype(np.uint8)
    for i in range(len(keep)):
        out += boards[i].tobytes()
        out += struct.pack("<B", int(stm_k[i]))
        out += struct.pack("<h", int(ev[i]))
        out += struct.pack("<B", int(res[i]))
        out += struct.pack("<B", int(ply[i]))
        out += struct.pack("<B", int(castling[i]))
        out += struct.pack("<B", 64)  # ep unknown in chunks
    return bytes(out), dict(kept=len(keep), skipped=int(skipped))


def iter_chunk_files(path):
    """Yield (name, raw_bytes) for chunk data. Handles tar/tar.gz/gz/raw/dir."""
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                fp = os.path.join(root, f)
                if f.endswith(".gz"):
                    with gzip.open(fp, "rb") as g:
                        yield f, g.read()
                else:
                    with open(fp, "rb") as fh:
                        yield f, fh.read()
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                if not (name.startswith("training") or name.endswith(".gz")):
                    # still accept any file; skip obvious non-chunks
                    if "/." in member.name or name.startswith("."):
                        continue
                fo = tf.extractfile(member)
                if fo is None:
                    continue
                data = fo.read()
                if name.endswith(".gz"):
                    try:
                        data = gzip.decompress(data)
                    except gzip.BadGzipFile:
                        continue
                yield name, data
        return
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as g:
            yield os.path.basename(path), g.read()
    else:
        with open(path, "rb") as fh:
            yield os.path.basename(path), fh.read()


def detect_version(data):
    if len(data) < 4:
        return None
    (ver,) = struct.unpack("<I", data[:4])
    return ver if ver in _FMT else None


def convert(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total_kept = total_skip = total_files = 0
    version_seen = {}
    out_f = open(args.out, "wb")
    try:
        for name, data in iter_chunk_files(args.input):
            ver = detect_version(data)
            if ver is None:
                print(f"  skip {name}: unknown version", flush=True)
                continue
            fmt = _FMT[ver]
            if len(data) % fmt["size"] != 0:
                print(f"  skip {name}: size {len(data)} not multiple of {fmt['size']}",
                      flush=True)
                continue
            version_seen[ver] = version_seen.get(ver, 0) + 1
            dec = decode_batch(data, fmt)
            if dec is None:
                continue
            visits = None
            if fmt["visits"] is not None and args.min_visits > 0:
                f32n = len(data) // fmt["size"]
                visits = np.frombuffer(data, dtype=np.uint32,
                                       count=f32n * (fmt["size"] // 4)).reshape(
                    f32n, -1)[:, fmt["visits"] // 4]
            recs, st = build_records(dec, args.sample_every, args.min_visits,
                                     visits)
            out_f.write(recs)
            total_kept += st["kept"]
            total_skip += st["skipped"]
            total_files += 1
            if total_files % 5 == 0 or total_kept >= args.max:
                print(f"  [{total_files} files] kept {total_kept} skipped-invalid "
                      f"{total_skip} versions {version_seen}", flush=True)
            if total_kept >= args.max:
                break
    finally:
        out_f.close()
    # trim to --max records if overshot
    if total_kept > args.max:
        with open(args.out, "r+b") as f:
            f.truncate(args.max * SDATA_V2)
        total_kept = args.max
    print(f"Done: {args.out}  {total_kept} positions "
          f"({total_kept * SDATA_V2} bytes) from {total_files} chunks, "
          f"skipped-invalid {total_skip}, versions {version_seen}")
    print(f"Next: python3 trainer/train.py --sdata {args.out} "
          f"--out nets/o2-lc0.o2nn --device cuda --epochs 80")


def main():
    ap = argparse.ArgumentParser(description="Lc0 chunks -> Owen sdata v2")
    ap.add_argument("--in", dest="input", required=True,
                    help=".tar/.tar.gz/.gz chunk, raw chunk, or directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=2000000)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--min-visits", type=int, default=0,
                    help="V6 only: drop records with fewer search visits")
    args = ap.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
