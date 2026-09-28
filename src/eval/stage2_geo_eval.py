"""EXP-S2-GEO-001 평가: ego 전용 split(배포 11모델과 동일)에서 geo 모델 seed 앙상블의 strict OOF와 배포 11과의 그룹 가중 혼합.
실행: python -m src.eval.stage2_geo_eval --dirs output/exp_s2_geo_001_s20260825 ... [--feats output/stage2_ccd_feats_geo --extra geo]
"""
import argparse, importlib, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann, load_feats
inf = importlib.import_module("inference_v26")
TAIL, TOL = 3, 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--feats", default="output/stage2_ccd_feats_geo"); ap.add_argument("--extra", nargs="*", default=["geo"]); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.ego & (ann.onset >= 0)]; FE = Path(a.feats); ann = ann[[(FE / f"{v}.npz").is_file() for v in ann.vid]]
    pool = ann[~ann.vid.isin({"000001", "000002", "000003", "000004", "000005"})].reset_index(drop=True)
    fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(20260825).permutation(sorted(set(pool.src))))}
    fold = np.array([fold_of[s] for s in pool.src]); onset = pool.onset.to_numpy().astype(int); vids = list(pool.vid)
    X = torch.from_numpy(np.stack([load_feats(FE, v, False, tuple(a.extra)) for v in vids])).to(dev); n = X.shape[1]
    oofE = np.load("output/s2_ens11_oof_scores.npz"); old = np.stack([oofE[v] for v in vids])
    new = np.zeros((len(a.dirs), len(vids), n), np.float32)
    with torch.inference_mode():
        for di, d in enumerate(a.dirs):
            for k in range(5):
                ck = torch.load(Path(d) / f"fold{k}.pt", map_location="cpu", weights_only=False); m = inf._l2_build(ck["config"]); m.load_state_dict(ck["model"]); m.to(dev).eval()
                sel = fold == k; new[di, sel] = torch.sigmoid(m(X[sel])).float().cpu().numpy()
    def hit(E): return (np.abs(E[:, : n - TAIL].argmax(1) - onset) <= TOL)
    h_old = hit(old); print(f"deployed 11-ens OOF hit {h_old.mean():.3f}")
    zp = np.load("output/s2_ens11_oof_permodel.npz"); S = zp["S"]; pv = {v: i for i, v in enumerate(zp["vids"])}; sel = [pv[v] for v in vids]
    print(f"baseline 3-seed noapp (m01-03) mean   hit {hit(S[:3][:, sel].mean(0)).mean():.3f} | 5 GRU (m01-05) {hit(S[:5][:, sel].mean(0)).mean():.3f}")
    for di, d in enumerate(a.dirs): print(f"{Path(d).name:32s} hit {hit(new[di]).mean():.3f}")
    N = new.mean(0); hN = hit(N); print(f"geo {len(a.dirs)}-seed mean               hit {hN.mean():.3f} | rescued {int(((~h_old) & hN).sum())} broken {int((h_old & ~hN).sum())}")
    for w in (0.3, 0.5, 0.7):
        hm = hit((1 - w) * old + w * N); print(f"mix w={w}                            hit {hm.mean():.3f} | rescued {int(((~h_old) & hm).sum())} broken {int((h_old & ~hm).sum())}")
    err = N[:, : n - TAIL].argmax(1) - onset; print("geo mean early(<-3) %.3f late(>3) %.3f" % ((err < -TOL).mean(), (err > TOL).mean()))


if __name__ == "__main__":
    main()
