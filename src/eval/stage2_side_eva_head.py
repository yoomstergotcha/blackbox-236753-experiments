"""EXP-S2-SIDEEVA-001 — 수동 라벨(기존 53/63 + v3 200)로 side/evasion 지도학습 헤드 검증.
특징: VLM 로짓(side_score, eva_img, eva_vid; output/s2_vlm_scores_all.csv, GT onset 기준) + YOLO 트랙 cue(피해차량 횡위치, 자유공간 점유, 박스 수).
모델: 표준화 로지스틱 회귀(numpy, L2) — 5-fold(클립) 교차검증, 기준 = 배포 VLM 임계(side thr 1.375, evasion z-합 thr −1.23).
실행: python -m src.eval.stage2_side_eva_head
"""
import glob, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from src.eval.stage2_track_utils import tracks_of, victim, bottom_center, evasion_geom
from src.train.train_stage2_collision import load_ann
_ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); onset = {v: int(o) for v, o in zip(_ann.vid, _ann.onset) if o >= 0}

M1, S1, M2, S2 = -1.1012, 0.1585, 0.0337, 0.1171; EVA_THR, SIDE_THR = -1.23, 1.375

lab = pd.concat([pd.read_csv("output/ccd_manual_labels_all.csv", dtype={"vid": str})] + [pd.read_csv(f, dtype={"vid": str}) for f in sorted(glob.glob("output/ccd_manual_labels_v*_part*.csv"))], ignore_index=True)
lab = lab.drop_duplicates("vid", keep="first"); lab = lab[lab.vid.isin(onset)]
sc = pd.read_csv("output/s2_vlm_scores_all.csv", dtype={"vid": str}).drop_duplicates("vid").set_index("vid")
lab = lab[lab.vid.isin(sc.index)].reset_index(drop=True); print("labeled clips with VLM scores:", len(lab), "| side", lab.manual_side.notna().sum(), "| evasion", lab.manual_evasion.notna().sum())


def track_cues(v):
    c = onset[v]; trs = tracks_of(v); vic = victim(trs, c, "area")
    xs = [bottom_center(vic, f) for f in range(max(c - 8, 0), c)] if vic is not None else []; xs = [p[0] for p in xs if p]
    lat = (np.mean(xs) - 0.5) if xs else 0.0; has = 1.0 if xs else 0.0
    occ = evasion_geom(trs, vic, c); nb = sum(1 for tr in trs if c in tr)
    area = max((((b[2] - b[0]) * (b[3] - b[1])) for tr in trs for f, (b, _) in tr.items() if f == c), default=0.0)
    return lat, has, occ, nb, area


rows = []
for v in lab.vid:
    s = sc.loc[v]; lat, has, occ, nb, area = track_cues(v)
    rows.append(dict(vid=v, side_score=s.side_score, eva_z=((s.eva_img - M1) / S1 + (s.eva_vid - M2) / S2), eva_img=s.eva_img, eva_vid=s.eva_vid, lat=lat, has=has, occ=occ, nb=nb, area=area))
F = pd.DataFrame(rows).set_index("vid"); lab = lab.set_index("vid")


def macro_f1(y, p, classes):
    f = []
    for c in classes:
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


def logreg_cv(X, y, folds=5, l2=1.0, seed=0, iters=500, lr=0.1):
    n = len(y); rng = np.random.default_rng(seed); order = rng.permutation(n); fold = np.zeros(n, int); fold[order] = np.arange(n) % folds; p = np.zeros(n)
    for k in range(folds):
        tr, te = fold != k, fold == k; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xt = np.c_[(X - mu) / sd, np.ones(n)]; w = np.zeros(Xt.shape[1])
        for _ in range(iters):
            z = Xt[tr] @ w; g = Xt[tr].T @ (1 / (1 + np.exp(-z)) - y[tr]) / tr.sum() + l2 * np.r_[w[:-1], 0] / tr.sum(); w -= lr * g
        p[te] = 1 / (1 + np.exp(-(Xt[te] @ w)))
    return p


# ---- side ----
s = lab[lab.manual_side.notna()]; ys = (s.manual_side == "LEFT").astype(int).to_numpy(); Fs = F.loc[s.index]
base = np.where(Fs.side_score > SIDE_THR, 1, 0); print(f"\nSIDE n={len(s)} | VLM thr1.375: acc {(base == ys).mean():.3f} macroF1 {macro_f1(ys, base, (0, 1)):.3f} | VLM median-thr: {macro_f1(ys, (Fs.side_score > Fs.side_score.median()).astype(int).to_numpy(), (0, 1)):.3f} | track sign: {macro_f1(ys[Fs.has > 0], (Fs.lat[Fs.has > 0] < 0).astype(int).to_numpy(), (0, 1)):.3f} (n={int(Fs.has.sum())})")
for name, cols in (("vlm", ["side_score"]), ("vlm+track", ["side_score", "lat", "has"]), ("vlm+track+det", ["side_score", "lat", "has", "occ", "nb", "area"])):
    res = []
    for seed in range(5):
        p = logreg_cv(Fs[cols].to_numpy(float), ys, seed=seed); res.append((macro_f1(ys, (p > 0.5).astype(int), (0, 1)), ((p > 0.5).astype(int) == ys).mean()))
    print(f"  LR[{name:14s}] 5-fold×5seed macroF1 {np.mean([r[0] for r in res]):.3f}±{np.std([r[0] for r in res]):.3f} acc {np.mean([r[1] for r in res]):.3f}")
# ---- evasion ----
e = lab[lab.manual_evasion.notna()]; ye = e.manual_evasion.astype(int).to_numpy(); Fe = F.loc[e.index]
base = (Fe.eva_z > EVA_THR).astype(int).to_numpy(); print(f"\nEVASION n={len(e)} (label1 {ye.mean():.2f}) | VLM z-sum thr−1.23: acc {(base == ye).mean():.3f} macroF1 {macro_f1(ye, base, (0, 1)):.3f} pred1 {base.mean():.2f}")
for thr in np.percentile(Fe.eva_z, [20, 30, 40]): pe = (Fe.eva_z > thr).astype(int).to_numpy(); print(f"  VLM z-sum thr {thr:.2f}: macroF1 {macro_f1(ye, pe, (0, 1)):.3f} pred1 {pe.mean():.2f}")
for name, cols in (("vlm2", ["eva_img", "eva_vid"]), ("vlm2+det", ["eva_img", "eva_vid", "occ", "nb", "area"]), ("vlm2+det+side", ["eva_img", "eva_vid", "occ", "nb", "area", "lat", "has"])):
    res = []
    for seed in range(5):
        p = logreg_cv(Fe[cols].to_numpy(float), ye, seed=seed); pe = (p > 0.5).astype(int); res.append((macro_f1(ye, pe, (0, 1)), (pe == ye).mean(), pe.mean()))
    print(f"  LR[{name:14s}] 5-fold×5seed macroF1 {np.mean([r[0] for r in res]):.3f}±{np.std([r[0] for r in res]):.3f} acc {np.mean([r[1] for r in res]):.3f} pred1 {np.mean([r[2] for r in res]):.2f}")
F.join(lab[["manual_side", "manual_evasion"]]).to_csv("output/s2_side_eva_features.csv")
