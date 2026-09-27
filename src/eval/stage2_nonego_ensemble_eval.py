"""ego+non-ego 학습 모델(seed 앙상블)과 배포 11모델(ego 전용)의 OOF 조합 평가. ego/non-ego 그룹별 hit±3.
실행: python -m src.eval.stage2_nonego_ensemble_eval --dirs output/exp_s2_all_001_noapp output/exp_s2_all_001_noapp_s1 ...
"""
import argparse, importlib, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann, load_feats
inf = importlib.import_module("inference_v26")
TAIL, TOL = 3, 3

def scores(models, X, dev):
    with torch.inference_mode():
        return np.mean([torch.sigmoid(m(X)).float().cpu().numpy() for m in models], 0)

def load(p, dev):
    ck = torch.load(p, map_location="cpu", weights_only=False); m = inf._l2_build(ck["config"]); m.load_state_dict(ck["model"]); return m.to(dev).eval(), ck["config"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--feats", default="output/stage2_ccd_feats"); ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--save", default="")
    a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.onset >= 0]; FE = Path(a.feats)
    ann = ann[[(FE / f"{v}.npz").is_file() for v in ann.vid]]; public = {"000001", "000002", "000003", "000004", "000005"}
    pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
    fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(20260825).permutation(sorted(set(pool.src))))}
    fold = np.array([fold_of[s] for s in pool.src]); onset = pool.onset.to_numpy().astype(int); ego = pool.ego.to_numpy(); vids = list(pool.vid)
    X = torch.from_numpy(np.stack([load_feats(FE, v, False, tuple(a.extra)) for v in vids])).to(dev); n = X.shape[1]
    X19 = torch.from_numpy(np.stack([load_feats(Path("output/stage2_ccd_feats"), v, False) for v in vids])).to(dev)
    new = np.zeros((len(a.dirs), len(vids), n), np.float32)
    for di, d in enumerate(a.dirs):
        for k in range(5):
            m, _ = load(Path(d) / f"fold{k}.pt", dev); sel = fold == k; new[di, sel] = scores([m], X[sel], dev)
    old = np.zeros((len(vids), n), np.float32); oofE = np.load("output/s2_ens11_oof_scores.npz")
    olds = [load(p, dev)[0] for p in sorted(Path("output/s2_ens_v26").glob("*.pt"))]
    for i, v in enumerate(vids):
        old[i] = oofE[v] if v in oofE.files else scores(olds, X19[i: i + 1], dev)[0]
    def rep(name, E):
        pred = E[:, : n - TAIL].argmax(1); hit = np.abs(pred - onset) <= TOL
        print(f"{name:34s} ego {hit[ego].mean():.3f} | non-ego {hit[~ego].mean():.3f} | all {hit.mean():.3f} | mix53/47 {0.53 * hit[ego].mean() + 0.47 * hit[~ego].mean():.3f}")
    rep("deployed 11-ens (ego-only)", old)
    for di, d in enumerate(a.dirs): rep(Path(d).name, new[di])
    N = new.mean(0); rep(f"new {len(a.dirs)}-seed mean", N)
    for w in (0.3, 0.5, 0.7): rep(f"old*(1-w)+new*w, w={w}", (1 - w) * old + w * N)
    rep("max(old,new)", np.maximum(old, N))
    if a.save: np.savez_compressed(a.save, new=N, old=old, vids=np.array(vids), onset=onset, ego=ego, fold=fold)

if __name__ == "__main__":
    main()
