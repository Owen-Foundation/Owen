#!/usr/bin/env python3
"""Owen 2 real-time learning daemon.

Streams finished StableAlpha games straight into training:
  writer   : new game PGN -> sdata -> SF19-distilled -> data/live/fresh.bin  (seconds)
  refresher: fresh pool >= threshold & GPU free -> short tag train vs mixed base
             -> validate vs deployed net -> atomic swap nets/active.o2nn

Deployed net = nets/active.o2nn (lichess-bot config NNUEFile points at it).
Each lichess-bot game spawns a fresh engine process, so an atomic file swap
is picked up by the next game automatically. No restart needed.

Usage:
  python3 scripts/real_time_learn.py --once          # one writer pass + one refresh pass
  python3 scripts/real_time_learn.py                 # daemon loop
"""
import argparse, json, logging, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "data" / "live"
FRESH = LIVE / "fresh.bin"
MINI_BASE = LIVE / "mini-base.bin"
STATE = LIVE / "state.json"
LOG = LIVE / "real_time.log"

SF_BIN = os.environ.get("SF_BIN", "/home/hemesh/Documents/stockfish/stockfish-linux-x86-64-universal")
ENGINE = ROOT / "build" / "owen2"
ATTACHED_NET = ROOT / "nets" / "active.o2nn"
BOOK = ROOT / "tools" / "book.epd"


def v_size(sz: int) -> int:
    if sz <= 0:
        return 71
    return 71 if (sz % 71 == 0 and sz % 69 != 0) else 69


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"seen_pgns": [], "round": 0, "champion": None}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(STATE)


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_step(args, check=True, env=None):
    e = dict(os.environ)
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:128")
    if env:
        e.update(env)
    r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, env=e)
    if check and r.returncode != 0:
        log(f"STEP FAILED {args[0]} rc={r.returncode}: {r.stderr[-800:]}")
        raise RuntimeError("step failed")
    return r


def gpu_free_log(tag, exec_check=True):
    import subprocess as sp
    try:
        out = sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return
    finally:
        pass
    log(f"GPU after {tag}: {out}")
    if exec_check:
        sp.run(["sync"], check=False)  # flush frees nothing; keep block honest


def reap_gpu(tag):
    """Let the CUDA context of a finished training process get torn down."""
    import time as _t
    gpu_free_log("pre-reap")
    for _ in range(30):
        _t.sleep(1)
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        try:
            if int(used) < 200:
                break  # context released; VRAM back to ~zero
        except ValueError:
            break
    gpu_free_log(tag)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_deployed_net() -> bool:
    """Boot the engine exactly like lichess-bot (cwd=ROOT) and confirm it loads
    nets/active.o2nn. Returns True only if the debug log shows a successful load."""
    import subprocess as sp
    try:
        p = sp.run([str(ENGINE)], input="uci\nisready\nsetoption name NNUEFile value nets/active.o2nn\nisready\nquit\n",
                   capture_output=True, text=True, cwd=ROOT, timeout=60)
        blob = p.stdout + p.stderr
        return "NNUE loaded: nets/active.o2nn" in blob
    except Exception as e:
        log(f"probe failed: {e}")
        return False


def verify_deployed(st):
    import datetime
    if not ATTACHED_NET.exists():
        return
    h = sha256(ATTACHED_NET)
    ok = probe_deployed_net()
    st["deployed_sha256"] = h
    st["deployed_probe_ok"] = ok
    st["deployed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    log(f"DEPLOY VERIFIED sha256={h[:12]} NNUE load={ok} (active.o2nn, bot picks up next game)")


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    return os.path.getsize(path) // v_size(os.path.getsize(path))


def undistilled_games(pgn_dir: Path, st) -> list:
    if not pgn_dir.exists():
        return []
    seen = set(st["seen_pgns"])
    out = []
    for p in sorted(pgn_dir.glob("*.pgn"), key=lambda x: x.stat().st_mtime):
        if p.name not in seen:
            out.append(p)
    return out


def build_mini_base(full: Path, target_records: int):
    if MINI_BASE.exists():
        return
    n = os.path.getsize(full)
    rs = v_size(n)
    total = n // rs
    stride = max(1, total // target_records)
    with open(full, "rb") as f:
        data = f.read()
    out = b"".join(data[i * rs:(i + 1) * rs] for i in range(0, total, stride))
    LIVE.mkdir(parents=True, exist_ok=True)
    with open(MINI_BASE, "wb") as f:
        f.write(out)
    log(f"mini-base built: {os.path.getsize(MINI_BASE)//rs} records (stride {stride})")


def writer_pass(pgn_dir: Path, st):
    games = undistilled_games(pgn_dir, st)
    if not games:
        return
    for g in games:
        try:
            raw = LIVE / "tmp_raw.bin"
            run_step([sys.executable, str(ROOT / "trainer/pgn_to_sdata.py"),
                      "--pgn", str(g), "--out", str(raw),
                      "--elo-min", "0", "--min-ply", "8", "--sample-every", "1",
                      "--threads", "4", "--chunk-games", "100", "--max", "100000"])
            if not raw.exists() or os.path.getsize(raw) == 0:
                st["seen_pgns"].append(g.name)
                st["seen_pgns"] = st["seen_pgns"][-5000:]
                log(f"no positions from {g.name}")
                continue
            dist = LIVE / "tmp_dist.bin"
            run_step([sys.executable, str(ROOT / "trainer/distill.py"),
                      "--sdata", str(raw), "--out", str(dist),
                      "--stockfish", SF_BIN, "--depth", "12", "--threads", "4",
                      "--hash", "64"])
            with open(dist, "rb") as f, open(FRESH, "ab") as out:
                shutil.copyfileobj(f, out)
            n = os.path.getsize(dist) // 71
            st["seen_pgns"].append(g.name)
            st["seen_pgns"] = st["seen_pgns"][-5000:]
            log(f"ingested {g.name}: +{n} pos (fresh pool now {count_records(FRESH)})")
            raw.unlink(missing_ok=True)
            dist.unlink(missing_ok=True)
        except Exception as e:
            log(f"game failed {g.name}: {e}")
            st["seen_pgns"].append(g.name)


def trainer_busy() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True).stdout
        return bool(out.strip())
    except Exception:
        return False


def refresher_pass(st, epochs: int, min_fresh: int):
    if count_records(FRESH) < min_fresh:
        return
    if not ATTACHED_NET.exists():
        return
    if count_records(MINI_BASE) == 0:
        log("no mini-base yet; skipping refresh")
        return
    if trainer_busy():
        log("train already running; deferring refresh")
        return
    st["round"] += 1
    r = st["round"]
    net = ROOT / f"nets/o2-live-r{r}.o2nn"
    champ = ROOT / (st["champion"] or "nets/active.o2nn")
    log(f"REFRESH round {r}: {count_records(FRESH)} fresh pos, train {epochs} epochs")
    sdata = f"{MINI_BASE},{FRESH}" + f",{FRESH}" * 8
    run_step([sys.executable, str(ROOT / "trainer/train.py"),
              "--sdata", sdata, "--out", str(net),
              "--device", "cuda", "--epochs", str(epochs), "--batch", "512",
              "--no-compile", "--workers", "2"])
    reap_gpu(f"train round {r}")
    if not net.exists():
        log("training produced no net; aborting refresh")
        return
    val_out = LIVE / f"val-r{r}.pgn"
    rc = run_step([sys.executable, str(ROOT / "tools/run_match.py"),
                   "--white-cmd", str(ENGINE), "--white-name", "new", "--white-opt", f"Threads=1 NNUEFile={net}",
                   "--black-cmd", str(ENGINE), "--black-name", "prev", "--black-opt", f"Threads=1 NNUEFile={champ}",
                   "--book", str(BOOK), "--games", "8", "--movetime", "0.3",
                   "--concurrency", "6", "--out", str(val_out)], check=False)
    raw = rc.stdout or ""
    import re
    m = re.search(r"Score (\d+(?:\.\d+)?)%", raw)
    score = float(m.group(1)) if m else 0.0
    log(f"refresh validation: {score}% ({val_out.name})")
    consumed = LIVE / f"consumed-r{r}.bin"
    FRESH.replace(consumed)
    if score >= 45.0:
        tmp = ATTACHED_NET.with_suffix(".o2nn.tmp")
        shutil.copyfile(net, tmp)
        tmp.replace(ATTACHED_NET)
        st["champion"] = str(net.relative_to(ROOT))
        verify_deployed(st)
        log(f"PROMOTED round {r}: {net.name} deployed as active.o2nn (score {score}%)")
    else:
        (ROOT / "nets" / "rejected").mkdir(exist_ok=True)
        net.replace(ROOT / "nets" / "rejected" / net.name)
        log(f"REJECTED round {r}: {score}% < 45%, pool archived as {consumed.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--pgn-dir", default=str(ROOT / "data" / "bot-games"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-fresh", type=int, default=1000)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--seed-net", default=None, help="copy this net to active.o2nn if missing")
    a = ap.parse_args()

    LIVE.mkdir(parents=True, exist_ok=True)
    lock = LIVE / ".lock"
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            log("another instance running; exiting")
            return
        except (ProcessLookupError, ValueError):
            lock.unlink(missing_ok=True)  # stale lock from a rebooted machine
    lock.write_text(str(os.getpid()))
    try:
        st = load_state()
        if a.seed_net and not ATTACHED_NET.exists():
            seed = ROOT / a.seed_net if not Path(a.seed_net).is_absolute() else Path(a.seed_net)
            if seed.exists():
                shutil.copyfile(seed, ATTACHED_NET)
                st["champion"] = str(seed.relative_to(ROOT))
                log(f"seeded active.o2nn from {seed.name}")
        full = ROOT / "data" / "sdata-final-distilled.bin"
        if full.exists():
            build_mini_base(full, 200000)
        verify_deployed(st)

        pgn_dir = Path(a.pgn_dir)
        if a.once:
            writer_pass(pgn_dir, st)
            save_state(st)
            refresher_pass(st, a.epochs, a.min_fresh)
            save_state(st)
            log("--once pass complete")
            return

        log(f"daemon up: poll {a.poll}s, min-fresh {a.min_fresh}, epochs {a.epochs}")
        while True:
            try:
                writer_pass(pgn_dir, st)
                save_state(st)
                refresher_pass(st, a.epochs, a.min_fresh)
                save_state(st)
            except Exception as e:
                log(f"loop error: {e}")
            time.sleep(a.poll)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()