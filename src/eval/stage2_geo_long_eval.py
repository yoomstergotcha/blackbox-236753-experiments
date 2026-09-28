"""EXP-S2-GEO-001 긴 클립 이동 검증(특징 공간): ego OOF 클립 앞뒤에 다른 클립의 사고 전 구간(물리 19-d + 기하 24-d)을 0~100프레임 이어붙여
배포 11(19-d) vs 기하 모델(43-d) vs 혼합의 hit±3을 fold-정확(OOF)으로 측정. 10fps stride 1 = 30fps stride 3 등가.
실행: python -m src.eval.stage2_geo_long_eval --dirs output/exp_s2_geo_s_001_s20260825 ... --feats output/stage2_ccd_feats_geo_s
"""
import argparse, importlib, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann, load_feats
inf = importlib.import_module("inference_v26")
TAIL, TOL = 3, 3


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
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--feats", default="output/stage2_ccd_feats_geo_s"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--w", nargs="*", type=float, default=[0.3, 0.5]); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); FE = Path(a.feats)
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.ego & (ann.onset >= 0)]; ann = ann[[(FE / f"{v}.npz").is_file() for v in ann.vid]]
    pool = ann[~ann.vid.isin({"000001", "000002", "000003", "000004", "000005"})].reset_index(drop=True)
    fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(20260825).permutation(sorted(set(pool.src))))}
    pool["fold"] = [fold_of[s] for s in pool.src]; X = {v: load_feats(FE, v, False, ("geo",)) for v in pool.vid}; onset = dict(zip(pool.vid, pool.onset.astype(int)))
    old = load_fold("output/s2_ens_v26/*_fold{k}.pt", dev); new = [load_fold(f"{d}/fold{{k}}.pt", dev) for d in a.dirs]
    rng = np.random.default_rng(11); vids = list(pool.vid); sub = rng.choice(len(vids), size=min(a.n, len(vids)), replace=False)
    def seg(vid, L):
        parts, tot = [], 0
        while tot < L:
            v = vids[int(rng.integers(len(vids)))]
            if v == vid: continue
            p = X[v][: max(onset[v] - 6, 3)]; parts.append(p); tot += len(p)
        return np.concatenate(parts)[:L] if L > 0 else X[vid][:0]
    res = {}
    for long in (False, True):
        hits = {"deployed-11": [], "geo": []}; hits.update({f"mix w={w}": [] for w in a.w})
        for i in sub:
            v = vids[i]; k = int(pool.fold[i]); pre, post = (int(rng.integers(0, 101)), int(rng.integers(0, 101))) if long else (0, 0)
            p_, q_ = seg(v, pre), seg(v, post); x = np.concatenate([p_, X[v], q_]).astype(np.float32); truth = onset[v] + len(p_)
            c_old = curve(old[k], x[:, :19], dev); c_new = np.mean([curve(nw[k], x, dev) for nw in new], 0)
            def dec(c): return int(np.argmax(c[: max(len(c) - TAIL, 1)]))
            hits["deployed-11"].append(abs(dec(c_old) - truth) <= TOL); hits["geo"].append(abs(dec(c_new) - truth) <= TOL)
            for w in a.w: hits[f"mix w={w}"].append(abs(dec((1 - w) * c_old + w * c_new) - truth) <= TOL)
        for g, h in hits.items(): res.setdefault(g, {})["long" if long else "short"] = round(float(np.mean(h)), 3)
    for g, v in res.items(): print(f"{g:14s} {v}")


if __name__ == "__main__":
    main()
