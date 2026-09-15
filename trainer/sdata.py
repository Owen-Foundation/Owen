"""
sdata layout:
  v3 (current, 73 bytes per record, pragma pack 1): v2 + move
    move      uint16 (from 6b | to 6b | promo 3b (0 none,1 N,2 B,3 R,4 Q) | ep 1b)
    (policy label: the move played from this position; 0 = none/unknown)
  v2 (71 bytes per record, pragma pack 1):
    board[64] uint8  (0..11 piece, 12 empty)
    stm       uint8  (0 white, 1 black)
    eval      int16  (centipawns)
    result    uint8  (0 loss, 1 draw, 2 win) from stm view
    ply       uint8
    castling  uint8  (K=1 Q=2 k=4 q=8)
    ep        uint8  (0..63 square, 64 none)
  v1 (legacy, 69 bytes): same minus castling/ep (readers default 0/64).
We use numpy to read efficiently.
"""
import struct, numpy as np, os

RECORD_FMT = "<64B B h B B" # little endian
RECORD_SIZE = struct.calcsize(RECORD_FMT) # 68+? 64+1+2+1+1=69? actually 69
# C++ SDataRecord packed is 64+1+2+1+1 = 69 bytes (pragma pack 1) [v1 legacy]
# v2 adds castling(1B) + ep(1B) = 71 bytes. v3 adds move(2B) = 73 bytes.
RECORD_SIZE_V1 = 69
RECORD_SIZE_V2 = 71
RECORD_SIZE_V3 = 73
RECORD_SIZE = 69

# policy index: 4352 = 4096 from-to + 256 promo ((promo-1)*64+to)
NPOL = 4352
def policy_index(move16):
    """move16: from|to<<6|promo<<12|ep<<15. Returns 0..4351 or -1 if none."""
    if not move16: return -1
    fr = move16 & 63; to = (move16 >> 6) & 63; promo = (move16 >> 12) & 7
    if promo: return 4096 + (promo - 1) * 64 + to
    return fr * 64 + to

def detect_record_size(path):
    sz = os.path.getsize(path)
    # Prefer newest version whose size divides evenly (v3 -> v2 -> v1).
    if sz > 0 and sz % RECORD_SIZE_V3 == 0:
        return RECORD_SIZE_V3
    if sz > 0 and sz % RECORD_SIZE_V2 == 0 and sz % RECORD_SIZE_V1 != 0:
        return RECORD_SIZE_V2
    return RECORD_SIZE_V1

def unpack_record(b):
    """Unpack a 69B (v1), 71B (v2) or 73B (v3) record -> dict."""
    if len(b) == RECORD_SIZE_V3:
        board = np.frombuffer(b[0:64], dtype=np.uint8)
        stm = b[64]
        ev = struct.unpack_from("<h", b, 65)[0]
        result = b[67]
        ply = b[68]
        castling = b[69]
        ep = b[70]
        move16 = struct.unpack_from("<H", b, 71)[0]
    elif len(b) == RECORD_SIZE_V2:
        board = np.frombuffer(b[0:64], dtype=np.uint8)
        stm = b[64]
        ev = struct.unpack_from("<h", b, 65)[0]
        result = b[67]
        ply = b[68]
        castling = b[69]
        ep = b[70]
        move16 = 0
    else:
        board = np.frombuffer(b[0:64], dtype=np.uint8)
        stm = b[64]
        ev = struct.unpack_from("<h", b, 65)[0]
        result = b[67]
        ply = b[68]
        castling = 0
        ep = 64
        move16 = 0
    return {"board": board, "stm": stm, "eval": ev, "result": result, "ply": ply,
            "castling": castling, "ep": ep, "move16": move16}

def iter_records(path, limit=None):
    sz = os.path.getsize(path)
    rs = detect_record_size(path)
    n = sz // rs
    if limit: n = min(n, limit)
    with open(path, "rb") as f:
        for i in range(n):
            b = f.read(rs)
            if len(b) < rs: break
            d = unpack_record(b)
            yield {"board": d["board"], "stm": d["stm"], "eval": d["eval"], "result": d["result"], "ply": d["ply"],
                   "move16": d.get("move16", 0)}

# Feature conversion: HalfKP index list for a position
# Mirrors src/nnue/features.cpp

def feature_indices(board, stm, wk_sq=None, bk_sq=None):
    """board: 64 array of piece codes 0..11, 12 empty. Returns list of indices for perspective stm."""
    # find kings
    if wk_sq is None:
        wks = np.where(board==5)[0]
        wk_sq = int(wks[0]) if len(wks) else 4
    if bk_sq is None:
        bks = np.where(board==11)[0]
        bk_sq = int(bks[0]) if len(bks) else 60
    king_sq = wk_sq if stm==0 else bk_sq
    # orient
    def orient(s, persp):
        if persp==1: return s ^ 56
        return s
    k = orient(king_sq, stm)
    out=[]
    for s, p in enumerate(board):
        if p==12: continue
        if p==5 or p==11: continue # king not a feature
        # 10-way mapping
        if p < 6: pc10 = p  # 0..4
        else: pc10 = (p-6)+5
        ps = orient(s, stm)
        if pc10>=10: continue
        idx = pc10*4096 + k*64 + ps
        out.append(idx)
    return out
