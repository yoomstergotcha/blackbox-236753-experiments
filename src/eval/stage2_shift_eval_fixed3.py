"""배포 규칙(stride 3 if n//3>=30 else 1, 꼬리 3 제외)으로 10/30fps × 짧은/긴 클립 이동에서 hit±0.3s 평가. OOF-정확(fold별 모델).
배포 11모델(ego 전용 split) vs 새 ego+non-ego 모델(split 재계산) vs 혼합. 실행: python -m src.eval.stage2_shift_eval_fixed3 --new output/exp_s2_all_001_noapp ...
"""
import argparse, importlib, sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann
inf = importlib.import_module("inference_v26")
TAIL = 3

def load_fold(pattern, dev):
    out = []
    for k in range(5):
        ms = []
        for p in sorted(Path().glob(pattern.format(k=k))):
            ck = torch.load(p, map_location="cpu", weights_only=False); m = inf._l2_build(ck["config"]); m.load_state_dict(ck["model"]); ms.append(m.to(dev).eval())
        out.append(ms)
    return out

@torch.inference_mode()
def curve(ms, x, dev):
    return np.mean([torch.sigmoid(m(torch.from_numpy(x)[None].to(dev))[0]).float().cpu().numpy() for m in ms], 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", nargs="*", default=[]); ap.add_argument("--w", nargs="*", type=float, default=[0.5]); ap.add_argument("--cache", default="output/stage2_ms_cache"); a = ap.parse_args(); C = Path(a.cache)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); idx = pd.read_csv(C / "index.csv", dtype={"vid": str}); cache = {r.vid: dict(np.load(C / f"{r.vid}.npz")) for r in idx.itertuples()}
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.onset >= 0]; FE = Path("output/stage2_ccd_feats"); ann = ann[[(FE / f"{v}.npz").is_file() for v in ann.vid]]
    pool = ann[~ann.vid.isin({"000001", "000002", "000003", "000004", "000005"})]
    fold_new = dict(zip(pool.vid, [ {s: i % 5 for i, s in enumerate(np.random.default_rng(20260825).permutation(sorted(set(pool.src))))}[s] for s in pool.src]))
    old = load_fold("output/s2_ens_v26/*_fold{k}.pt", dev); new = [load_fold(f"{d}/fold{{k}}.pt", dev) for d in a.new]
    rng = np.random.default_rng(7)
    def seg(vid, rate, s, L):
        parts, tot = [], 0
        while tot < L:
            v = idx.vid[int(rng.integers(len(idx)))]
            if v == vid: continue
            o = int(cache[v]["onset10"]) * (int(rate) // 10) // s; p = cache[v][f"{rate}_s{s}"][: max(o - 5 // s - 1, 3)]; parts.append(p); tot += len(p)
        return np.concatenate(parts)[:L] if L > 0 else cache[vid][f"{rate}_s1"][:0]
    groups = {"deployed-11": lambda r, x: curve(old[int(r.fold)], x, dev)}
    for i, d in enumerate(a.new): groups[Path(d).name] = (lambda i: lambda r, x: curve(new[i][fold_new[r.vid]], x, dev))(i)
    if new:
        groups[f"new{len(new)}-mean"] = lambda r, x: np.mean([curve(new[i][fold_new[r.vid]], x, dev) for i in range(len(new))], 0)
        for w in a.w: groups[f"mix w={w}"] = (lambda w: lambda r, x: (1 - w) * curve(old[int(r.fold)], x, dev) + w * np.mean([curve(new[i][fold_new[r.vid]], x, dev) for i in range(len(new))], 0))(w)
    res = {g: {} for g in groups}
    for rate in ("10", "30"):
        f = int(rate) // 10; tol = 3 * f
        for long in (False, True):
            hits = {g: [] for g in groups}
            for r in idx.itertuples():
                pre, post = (int(rng.integers(0, 101)) * f, int(rng.integers(0, 101)) * f) if long else (0, 0)
                x1 = cache[r.vid][f"{rate}_s1"]; n = len(x1) + pre + post; s = 3 if n // 3 >= 30 else 1
                x = cache[r.vid][f"{rate}_s{s}"]; p_, q_ = seg(r.vid, rate, s, pre // s), seg(r.vid, rate, s, post // s); x = np.concatenate([p_, x, q_]); truth = int(r.onset) * f + len(p_) * s
                for g, fn in groups.items():
                    c = fn(r, x); hits[g].append(abs(int(np.argmax(c[: max(len(c) - TAIL, 1)])) * s - truth) <= tol)
            for g in groups: res[g][f"{rate}fps {'long' if long else 'short'}"] = round(float(np.mean(hits[g])), 3)
    for g, v in res.items(): print(f"{g:22s} {v} | min {min(v.values()):.3f}")

if __name__ == "__main__":
    main()
