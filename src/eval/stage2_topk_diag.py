"""11모델 union 앙상블(배포 v26~v36 collision)의 진짜 OOF 점수: argmax 오답 클립에서 정답이 top-k 후보에 있는지 진단."""
import sys, importlib, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann, load_feats
inf = importlib.import_module("inference_v26")
FE, CK, TAIL, TOL = Path("output/stage2_ccd_feats"), Path("output/s2_ens_v26"), 3, 3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.ego & (ann.onset >= 0)]
ann = ann[[(FE / f"{v}.npz").is_file() for v in ann.vid]]
public = {"000001", "000002", "000003", "000004", "000005"}
pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(20260825).permutation(sorted(set(pool.src))))}
fold = np.array([fold_of[s] for s in pool.src]); onset = pool.onset.to_numpy().astype(int); vids = list(pool.vid)
X = np.stack([load_feats(FE, v, False) for v in vids]); n = X.shape[1]; print("clips", len(vids), "frames", n, "dim", X.shape[2])
Xt = torch.from_numpy(X).to(dev)
names = sorted({p.name.rsplit("_fold", 1)[0] for p in CK.glob("*_fold0.pt")}); print("models", names)
S = np.zeros((len(names), len(vids), n), np.float32); tab = []
with torch.inference_mode():
    for mi, nm in enumerate(names):
        for k in range(5):
            ck = torch.load(CK / f"{nm}_fold{k}.pt", map_location="cpu", weights_only=False)
            m = inf._l2_build(ck["config"]); m.load_state_dict(ck["model"]); m.to(dev).eval()
            sc = torch.sigmoid(m(Xt)).float().cpu().numpy()
            pred = sc[:, : n - TAIL].argmax(1); hit = np.abs(pred - onset) <= TOL
            tab.append((nm, k, hit[fold != k].mean(), hit[fold == k].mean()))
            S[mi, fold == k] = sc[fold == k]
t = pd.DataFrame(tab, columns=["model", "fold", "in_sample", "oof"]); print(t.groupby("model")[["in_sample", "oof"]].mean().round(3))
E = S.mean(0); np.savez_compressed("output/s2_ens11_oof_scores.npz", **{v: E[i] for i, v in enumerate(vids)})
np.savez_compressed("output/s2_ens11_oof_permodel.npz", S=S, vids=np.array(vids), onset=onset, fold=fold, names=np.array(names))

def cands(e, sep=TOL):
    e = e[: n - TAIL]; order = np.argsort(-e); out = []
    for t_ in order:
        if all(abs(t_ - c) > sep for c in out): out.append(int(t_))
        if len(out) == 8: break
    return out

pred = E[:, : n - TAIL].argmax(1); err = pred - onset; hit = np.abs(err) <= TOL
print(f"\n11-ens OOF argmax(tail3) hit±3 {hit.mean():.3f} | early(<-3) {(err < -TOL).mean():.3f} late(>3) {(err > TOL).mean():.3f}")
print("shift δ:", {d: round(float((np.abs(np.clip(pred + d, 0, n - 1) - onset) <= TOL).mean()), 3) for d in (-3, -2, -1, 1, 2, 3)})
print("tail1 :", round(float((np.abs(E[:, : n - 1].argmax(1) - onset) <= TOL).mean()), 3), "| tail0:", round(float((np.abs(E.argmax(1) - onset) <= TOL).mean()), 3))
C = [cands(E[i]) for i in range(len(vids))]
rank = np.array([next((j for j, c in enumerate(cs) if abs(c - o) <= TOL), 99) for cs, o in zip(C, onset)])
print("oracle top-k hit:", {k: round(float((rank < k).mean()), 3) for k in (1, 2, 3, 4, 5, 8)})
miss = ~hit; print(f"\nmisses {miss.sum()} | GT rank among misses: top2 {(rank[miss] == 1).mean():.3f} top3 {(rank[miss] == 2).mean():.3f} top4-8 {((rank[miss] >= 3) & (rank[miss] < 8)).mean():.3f} none {(rank[miss] == 99).mean():.3f}")
gtwin = np.array([E[i, max(o - TOL, 0): o + TOL + 1].max() for i, o in enumerate(onset)]); p1 = np.array([E[i, c[0]] for i, c in enumerate(C)]); p2 = np.array([E[i, c[1]] if len(c) > 1 else 0 for i, c in enumerate(C)])
print("GT-window max score: hits med %.2f | misses med %.2f (p25 %.2f p75 %.2f)" % (np.median(gtwin[hit]), np.median(gtwin[miss]), *np.percentile(gtwin[miss], [25, 75])))
print("p2/p1 ratio: hits med %.2f | misses(GT=top2) med %.2f | misses(other) med %.2f" % (np.median(p2[hit] / p1[hit]), np.median((p2 / p1)[miss & (rank == 1)]), np.median((p2 / p1)[miss & (rank != 1)])))
r2 = miss & (rank == 1); gt2 = np.array([c[1] if len(c) > 1 else -1 for c in C])
print(f"GT=top2 cases {r2.sum()}: GT cand earlier than top1 {np.mean(gt2[r2] < pred[r2]):.2f} | top1 err med {np.median(err[r2]):.0f} | |err| bins 4-6/7-10/11-20/>20:", np.histogram(np.abs(err[miss]), [4, 7, 11, 21, 99])[0].tolist(), "(all misses)")
for a in (0.5, 0.6, 0.7, 0.8, 0.9):
    fp = np.array([min([c for c in cs if E[i, c] >= a * E[i, cs[0]]]) for i, cs in enumerate(C)]); print(f"earliest cand ≥{a}·p1: hit {(np.abs(fp - onset) <= TOL).mean():.3f}")
# 앙상블 크기 민감도: 5모델(구 앙상블) vs 11
E5 = S[:5].mean(0); print("5-model (m01-05) OOF hit:", round(float((np.abs(E5[:, : n - TAIL].argmax(1) - onset) <= TOL).mean()), 3))
pd.DataFrame(dict(vid=vids, fold=fold, onset=onset, pred=pred, err=err, hit=hit, rank=rank, p1=p1, p2=p2, gtwin=gtwin, c2=gt2)).to_csv("output/s2_ens11_oof_diag.csv", index=False)
