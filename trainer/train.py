#!/usr/bin/env python3
"""
Owen 2 v2 — SFNNv10-class trainer sdata -> .o2nn ver=2
H=1024, INPUT=81920 (HalfKP + Threat). L1=16, L2=32.
DeepSeek-style GPU-shortage tricks for 4GB RTX 2050:
  1) Mixed precision FP16 autocast + GradScaler (2x speed, half VRAM)
  2) Gradient accumulation (micro-batch 512 fits 4GB, effective 8192)
  3) 8-bit AdamW via bitsandbytes if present (75% optimizer RAM saved)
  4) Mmap dataset (no per-sample open/seek, zero-copy)
  5) TF32 + cudnn.benchmark + channels_last (Tensor Core)
  6) torch.compile (inductor) if available
No Stockfish code.
"""
import argparse, os, struct, math, random
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sdata import RECORD_SIZE, RECORD_SIZE_V1, RECORD_SIZE_V2, RECORD_SIZE_V3, NPOL, policy_index

INPUT_SIZE = 81920
HALFKP = 40960
H = 1024
L1 = 16
L2 = 32

def feature_indices_v2(board, stm):
    wk = int(np.where(board==5)[0][0]) if np.any(board==5) else 4
    bk = int(np.where(board==11)[0][0]) if np.any(board==11) else 60
    king = wk if stm==0 else bk
    def orient(s, persp): return s ^ 56 if persp==1 else s
    k = orient(king, stm)
    occ = set(int(i) for i in range(64) if board[i]!=12)
    attacked = set()
    for s, p in enumerate(board):
        p = int(p)  # numpy uint8 scalar -> python int (numpy>=2 strictness)
        if p==12: continue
        col = 0 if p < 6 else 1
        if col == stm: continue
        pt = p % 6
        r, f = divmod(s, 8)
        if pt==0:
            nr = r + (1 if col==0 else -1)
            for df in (-1,1):
                nf=f+df
                if 0<=nr<8 and 0<=nf<8: attacked.add(nr*8+nf)
        elif pt==1:
            for dr,df in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
                nr, nf = r+dr, f+df
                if 0<=nr<8 and 0<=nf<8: attacked.add(nr*8+nf)
        elif pt==5:
            for dr in (-1,0,1):
                for df in (-1,0,1):
                    if dr==0 and df==0: continue
                    nr, nf = r+dr, f+df
                    if 0<=nr<8 and 0<=nf<8: attacked.add(nr*8+nf)
        elif pt in (2,3,4):
            dirs = []
            if pt in (2,4): dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]
            if pt in (3,4): dirs += [(1,0),(-1,0),(0,1),(0,-1)]
            for dr,df in dirs:
                nr, nf = r+dr, f+df
                while 0<=nr<8 and 0<=nf<8:
                    ns = nr*8+nf
                    attacked.add(ns)
                    if ns in occ: break
                    nr+=dr; nf+=df
    out=[]
    for s, p in enumerate(board):
        p = int(p)  # numpy uint8 scalar -> python int (numpy>=2 raises on overflow ops)
        if p==12: continue
        if p==5 or p==11: continue
        pc10 = p if p<6 else (p-6)+5
        if pc10>=10: continue
        ps = orient(s, stm)
        base = pc10*4096 + k*64 + ps
        out.append(base)
        if s in attacked:
            out.append(HALFKP + base)
    return out

class OwenNetV2(nn.Module):
    def __init__(self, with_policy=False):
        super().__init__()
        self.with_policy = with_policy
        self.ft = nn.Embedding(INPUT_SIZE, H)
        self.ft_bias = nn.Parameter(torch.zeros(H))
        self.l1 = nn.Linear(H, L1)
        self.l2 = nn.Linear(L1, L2)
        self.out = nn.Linear(L2, 1)
        if with_policy:
            self.pol = nn.Linear(L2, NPOL)
        nn.init.normal_(self.ft.weight, std=0.02)
        nn.init.zeros_(self.ft_bias)
        nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        nn.init.zeros_(self.l1.bias)
        nn.init.kaiming_uniform_(self.l2.weight, nonlinearity='relu')
        nn.init.zeros_(self.l2.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
        if with_policy:
            nn.init.kaiming_uniform_(self.pol.weight, nonlinearity='linear')
            nn.init.zeros_(self.pol.bias)
    def forward(self, idx):
        mask = (idx != -1).float()
        emb = self.ft(idx.clamp(min=0)) * mask.unsqueeze(-1)
        acc = emb.sum(dim=1) + self.ft_bias
        h0 = torch.clamp(acc / 64.0, 0, 127) / 127.0
        h1 = torch.clamp(self.l1(h0), 0, 1)
        h2 = torch.clamp(self.l2(h1), 0, 1)
        out = self.out(h2).squeeze(-1)
        if self.with_policy:
            return out * 1000.0, self.pol(h2)
        return out * 1000.0, None
    def export_o2nn(self, path):
        fw = (self.ft.weight.detach().cpu().numpy() * 64).round().clip(-32768,32767).astype(np.int16)
        fb = (self.ft_bias.detach().cpu().numpy() * 64).round().clip(-32768,32767).astype(np.int16)
        l1w = (self.l1.weight.detach().cpu().numpy() * 64).round().clip(-128,127).astype(np.int8)
        l1b = (self.l1.bias.detach().cpu().numpy() * 64).round().clip(-32768,32767).astype(np.int16)
        l2w = (self.l2.weight.detach().cpu().numpy() * 64).round().clip(-128,127).astype(np.int8)
        l2b = (self.l2.bias.detach().cpu().numpy() * 64).round().clip(-32768,32767).astype(np.int16)
        ow  = (self.out.weight.detach().cpu().numpy() * 64).round().clip(-128,127).astype(np.int8).flatten()
        ob  = int(np.round(float(self.out.bias.detach().cpu().numpy()) * 64))
        ob = max(-32768, min(32767, ob))
        ver = 3 if self.with_policy else 2
        with open(path,"wb") as f:
            f.write(b"O2NN")
            f.write(struct.pack("<II", ver, H))
            f.write(fw.tobytes())
            f.write(fb.tobytes())
            f.write(l1w.tobytes())
            f.write(l1b.tobytes())
            f.write(l2w.tobytes())
            f.write(l2b.tobytes())
            f.write(ow.tobytes())
            f.write(struct.pack("<h", ob))
            if self.with_policy:
                pw = (self.pol.weight.detach().cpu().numpy() * 64).round().clip(-128,127).astype(np.int8)
                pb = (self.pol.bias.detach().cpu().numpy() * 64).round().clip(-32768,32767).astype(np.int16)
                f.write(pw.tobytes())
                f.write(pb.tobytes())
        print(f"Exported  {path}  ({os.path.getsize(path)} bytes)")

    def load_o2nn(self, path):
        data = open(path,"rb").read()
        off = 0
        def take(n):
            nonlocal off
            x = data[off:off+n]; off += n
            return x
        magic = take(4)
        assert magic == b"O2NN", f"bad magic {magic}"
        ver, Hr = struct.unpack("<II", take(8))
        assert Hr == H, f"hidden {Hr} != {H}"
        fw = np.frombuffer(take(INPUT_SIZE*H*2), dtype=np.int16).reshape(INPUT_SIZE, H)
        fb = np.frombuffer(take(H*2), dtype=np.int16)
        l1w = np.frombuffer(take(L1*H), dtype=np.int8).reshape(L1, H)
        l1b = np.frombuffer(take(L1*2), dtype=np.int16)
        l2w = np.frombuffer(take(L2*L1), dtype=np.int8).reshape(L2, L1)
        l2b = np.frombuffer(take(L2*2), dtype=np.int16)
        ow  = np.frombuffer(take(L2), dtype=np.int8).flatten()
        ob  = np.frombuffer(take(2), dtype=np.int16)[0]
        self.ft.weight.data.copy_(torch.from_numpy(fw.astype(np.float32)/64.0))
        self.ft_bias.data.copy_(torch.from_numpy(fb.astype(np.float32)/64.0))
        self.l1.weight.data.copy_(torch.from_numpy(l1w.astype(np.float32)/64.0))
        self.l1.bias.data.copy_(torch.from_numpy(l1b.astype(np.float32)/64.0))
        self.l2.weight.data.copy_(torch.from_numpy(l2w.astype(np.float32)/64.0))
        self.l2.bias.data.copy_(torch.from_numpy(l2b.astype(np.float32)/64.0))
        self.out.weight.data.copy_(torch.from_numpy(ow.astype(np.float32)/64.0))
        self.out.bias.data.copy_(torch.tensor(float(ob)/64.0))
        if self.with_policy and off < len(data):
            rest = np.frombuffer(data[off:], dtype=np.uint8)
            nb = NPOL*2
            pw = rest[:len(rest)-nb].view(np.int8).reshape(NPOL, L2)
            pb = rest[-nb:].view(np.int16).reshape(NPOL)
            self.pol.weight.data.copy_(torch.from_numpy(pw.astype(np.float32)/64.0))
            self.pol.bias.data.copy_(torch.from_numpy(pb.astype(np.float32)/64.0))
        print(f"Loaded {path}  ver={ver}  policy={'yes' if (self.with_policy and off<len(data)) else 'no'}")

# DeepSeek trick #4: mmap dataset — read whole file once via memmap, no open/seek per sample
class SDataDataset(Dataset):
    def __init__(self, path, max_active=48):
        self.path=path
        self.max_active=max_active
        sz = os.path.getsize(path)
        # v3 (73B) adds the played move (policy label); v2 (71B) has
        # castling+ep trailer; v1 legacy is 69B.
        if sz % RECORD_SIZE_V3 == 0 and sz > 0:
            self.rs = RECORD_SIZE_V3
        elif sz % RECORD_SIZE_V2 == 0 and sz % RECORD_SIZE_V1 != 0:
            self.rs = RECORD_SIZE_V2
        else:
            self.rs = RECORD_SIZE_V1
        self.n = sz // self.rs
        print(f"Dataset v2 mmap: {self.n} positions from {path} (H={H} threat, record {self.rs}B)")
        # mmap as raw bytes for zero-copy access
        self.data = np.memmap(path, dtype=np.uint8, mode='r')
        # verify size
        assert self.data.size >= self.n * self.rs
    def __len__(self): return self.n
    def __getitem__(self, i):
        off = i * self.rs
        # slice without copy where possible, then copy small record to parse
        b = self.data[off:off+self.rs]
        # numpy slice is still memmap view; convert to bytes via tobytes for struct
        # faster: use memoryview
        board = b[0:64].copy()  # 64B
        stm = int(b[64])
        ev = struct.unpack_from("<h", b, 65)[0]
        result = int(b[67])
        result_cp = {0:-600, 1:0, 2:600}[result]
        target = 0.6*result_cp + 0.4*float(np.clip(ev,-1500,1500))
        pol = -1
        if self.rs == RECORD_SIZE_V3:
            move16 = struct.unpack_from("<H", b, 71)[0]
            pol = policy_index(move16)
        feats = feature_indices_v2(board, stm)
        if len(feats) > self.max_active: feats = random.sample(feats, self.max_active)
        arr = np.full(self.max_active, -1, dtype=np.int64)
        arr[:len(feats)] = feats
        return torch.from_numpy(arr), torch.tensor(float(target), dtype=torch.float32), torch.tensor(pol, dtype=torch.int64)

def train(args):
    device = args.device
    if device=="auto":
        if torch.cuda.is_available(): device="cuda"
        elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): device="mps"
        else: device="cpu"
    print(f"Device: {device}  H={H} L1={L1} L2={L2} micro_batch={args.batch} accum={args.accum} eff={args.batch*args.accum} amp={args.amp}")

    # DeepSeek tricks: enable Tensor Core
    if device=="cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    ds_paths = [p.strip() for p in args.sdata.split(",") if p.strip()]
    datasets = [SDataDataset(p, max_active=args.max_active) for p in ds_paths]
    ds = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    if len(datasets) > 1:
        print(f"Mixed {len(datasets)} datasets: {[len(d) for d in datasets]}")
    n_train = int(len(ds)*0.95)
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train], generator=g)

    # num_workers 2 + pin_memory + prefetch = overlap CPU feature gen with GPU
    # keep 0 on low RAM systems to avoid fork OOM; auto-detect
    nw = 2 if os.cpu_count() and os.cpu_count() >= 8 else 0
    if args.workers is not None: nw = args.workers
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=(device=="cuda"), prefetch_factor=2 if nw>0 else None, persistent_workers=(nw>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch*4, shuffle=False, num_workers=nw, pin_memory=(device=="cuda"), prefetch_factor=2 if nw>0 else None, persistent_workers=(nw>0))

    net = OwenNetV2(with_policy=args.policy).to(device)
    if args.init:
        net.load_o2nn(args.init)
    # torch.compile = DeepSeek fused kernels (inductor)
    if args.compile and hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skip: {e}")

    # 8-bit AdamW if bitsandbytes present — saves 75% optimizer VRAM (DeepSeek CPU offload style)
    opt = None
    try:
        if args.eight_bit and device=="cuda":
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(net.parameters(), lr=args.lr, weight_decay=1e-4)
            print("Optimizer: AdamW8bit (bitsandbytes) — 75% VRAM saved")
    except Exception as e:
        print(f"8-bit adam not available: {e}")

    if opt is None:
        # fused AdamW on CUDA is faster and uses less memory than for-loop
        try:
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, fused=(device=="cuda"))
        except TypeError:
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    # DeepSeek #1: AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device=="cuda"))
    best_val=float("inf")

    for epoch in range(1, args.epochs+1):
        net.train()
        total=0
        ptotal=0; pcount=0
        opt.zero_grad(set_to_none=True)
        for step, (idx, target, pol) in enumerate(train_loader, 1):
            idx, target = idx.to(device, non_blocking=True), target.to(device, non_blocking=True)
            # autocast FP16 for forward — Tensor Core on RTX 2050
            with torch.cuda.amp.autocast(enabled=(args.amp and device=="cuda")):
                pred, plogits = net(idx)
                loss = loss_fn(pred, target)
                if args.policy and plogits is not None:
                    pol = pol.to(device, non_blocking=True)
                    m = (pol >= 0)
                    if m.any():
                        ploss = ce_fn(plogits[m], pol[m])
                        loss = loss + args.pol_w * ploss
                        ptotal += ploss.item() * int(m.sum()); pcount += int(m.sum())
                loss = loss / args.accum  # scale for accumulation

            scaler.scale(loss).backward()

            if step % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total+=loss.item()*args.accum*len(target)

            if step % 500 == 0:
                print(f"  step {step} loss {loss.item()*args.accum:.1f}")

        # handle leftover grads
        if len(train_loader) % args.accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sched.step()
        net.eval()
        vloss=0; vplos=0; vpct=0; vtop1=0; vtop1n=0
        with torch.no_grad():
            for idx, target, pol in val_loader:
                idx, target = idx.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device=="cuda")):
                    pred, plogits = net(idx)
                vloss += loss_fn(pred, target).item()*len(target)
                if args.policy and plogits is not None:
                    pol = pol.to(device, non_blocking=True)
                    m = (pol >= 0)
                    if m.any():
                        vplos += ce_fn(plogits[m], pol[m]).item()*int(m.sum()); vpct += int(m.sum())
                        vtop1 += (plogits[m].argmax(dim=1) == pol[m]).sum().item(); vtop1n += int(m.sum())
        vloss/=max(1,len(val_ds))
        tloss=total/max(1,len(train_ds))
        extra=""
        combined=vloss
        if args.policy and vpct:
            vplos/=vpct; extra=f" pval {vplos:.3f} top1 {vtop1/max(1,vtop1n)*100:.1f}%"
            if pcount: extra+=f" ptrain {ptotal/pcount:.3f}"
            combined=vloss + args.pol_w*vplos
        print(f"epoch {epoch:3d} train {tloss:.1f} val {vloss:.1f}{extra} lr {sched.get_last_lr()[0]:.2e}")
        if combined < best_val:
            best_val=combined
            # unwrap compile wrapper for export
            raw = net._orig_mod if hasattr(net, "_orig_mod") else net
            raw.export_o2nn(args.out)
        # free cache each epoch to avoid 4GB fragmentation
        if device=="cuda": torch.cuda.empty_cache()
    print("Done. Best val", best_val)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--sdata", required=True,
                    help="sdata file, or comma-separated list to mix "
                         "(e.g. match games + base pool, each auto-detects v1/v2)")
    ap.add_argument("--out", default="nets/o2-v1.o2nn")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512, help="micro-batch that fits 4GB (default 512)")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps (effective batch = batch*accum, default 4096)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-active", type=int, default=48)
    ap.add_argument("--device", default="auto", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="FP16 mixed precision (DeepSeek FP8 style)")
    ap.add_argument("--eight-bit", action=argparse.BooleanOptionalAction, default=False, help="8-bit AdamW via bitsandbytes")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--policy", action=argparse.BooleanOptionalAction, default=False,
                    help="train a policy head (needs v3 sdata with moves); exports ver=3 net")
    ap.add_argument("--pol-w", type=float, default=1000.0, help="policy CE weight in joint loss (value MSE is ~1e5 scale, CE ~8)")
    ap.add_argument("--init", default=None, help="warm-start from this .o2nn (ver 2 or 3)")
    args=ap.parse_args()
    if args.gpu: args.device="cuda"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train(args)
