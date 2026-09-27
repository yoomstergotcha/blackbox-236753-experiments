"""EXP-S2-RERANK-001 feasibility: 11-ens OOF 상위 K 후보를 listwise MLP로 재순위. 특징군 ablation(curve/phys/app/det), fold OOF."""
import sys, numpy as np, torch
from pathlib import Path
from torch import nn
z = np.load("output/s2_ens11_oof_permodel.npz"); S, vids, onset, fold = z["S"], list(z["vids"]), z["onset"], z["fold"]
E = S.mean(0); M, NV, n = S.shape; TOL, TAIL, K = 3, 3, int(sys.argv[1]) if len(sys.argv) > 1 else 3
FE, DET = Path("output/stage2_ccd_feats"), Path("output/stage2_ccd_det")
def cands(e):
    e = e[: n - TAIL]; out = []
    for t in np.argsort(-e):
        if all(abs(t - c) > TOL for c in out): out.append(int(t))
        if len(out) == K: break
    return out
def W(a, lo, hi):
    lo, hi = max(lo, 0), min(hi, n)
    if hi <= lo: lo, hi = min(max(lo, 0), n - 1), min(max(lo, 0), n - 1) + 1
    seg = a[lo:hi]; return np.concatenate([seg.mean(0), seg.max(0)]) if seg.ndim == 2 else np.array([seg.mean(), seg.max()])
def det_series(vid):
    area, ybot, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    if not (DET / f"{vid}.npz").is_file():
        MISSING.append(vid); return area, ybot, cnt
    d = np.load(DET / f"{vid}.npz")["det"]
    for fr, x1, y1, x2, y2, cf, cl in d:
        i = int(fr)
        if cl in (2, 3, 5, 7) and cf >= 0.3 and i < n:
            a = (x2 - x1) * (y2 - y1); cnt[i] += 1
            if a > area[i]: area[i], ybot[i] = a, y2
    return area, ybot, cnt
G = {"curve": [], "phys": [], "app": [], "det": []}; Y, C = [], []; MISSING = []
for i, v in enumerate(vids):
    zz = np.load(FE / f"{v}.npz"); phys = np.concatenate([zz["signals"], zz["motion"]], 1).astype(np.float32); app = zz["app"].astype(np.float32)
    nov = np.r_[0, np.linalg.norm(np.diff(app, axis=0), axis=1)]; area, ybot, cnt = det_series(v); e = E[i]; am = S[:, i, : n - TAIL].argmax(1); cs = cands(e); c1 = cs[0]
    g = {k: [] for k in G}; y, cc = [], []
    for r, c in enumerate(cs):
        g["curve"].append([e[c], e[c] / e[c1], r, c / n, (c - c1) / n, *W(e, c - 3, c + 4), *W(e, c - 10, c - 3), *W(e, c + 4, c + 11), e[c] - e[max(c - 3, 0)], e[c] - e[min(c + 3, n - 1)], (np.abs(am - c) <= TOL).mean(), S[:, i, c].std(), (e[: n - TAIL] >= 0.5 * e[c1]).sum() / n])
        g["phys"].append(np.concatenate([W(phys, c - 3, c + 4), W(phys, c - 9, c - 3), W(phys, c + 4, c + 10)]))
        pb, pa = app[max(c - 9, 0): max(c - 3, 1)].mean(0), (app[c + 4: c + 10].mean(0) if c + 4 < n else app[-1])
        g["app"].append([*W(nov, c - 3, c + 4), *W(nov, c - 9, c - 3), *W(nov, c + 4, c + 10), float(pb @ pa / (np.linalg.norm(pb) * np.linalg.norm(pa) + 1e-6))])
        g["det"].append([*W(area, c - 3, c + 4), area[c] - area[max(c - 5, 0)], *W(area, c - 9, c - 3), *W(area, c + 4, c + 10), *W(ybot, c - 3, c + 4), *W(cnt, c - 3, c + 4)])
        y.append(abs(c - onset[i]) <= TOL); cc.append(c)
    for k in G: G[k].append(np.array(g[k], np.float32))
    Y.append(y); C.append(cc)
G = {k: np.stack(v) for k, v in G.items()}; Y = np.array(Y, np.float32); C = np.array(C)
print("det missing", len(MISSING)); print("K", K, "| feature dims", {k: v.shape[2] for k, v in G.items()}, "| top1 hit %.3f | oracle %.3f | clips w/o positive %d" % (Y[:, 0].mean(), Y.max(1).mean(), (Y.max(1) == 0).sum()))
votes = np.stack([[(np.abs(S[:, i, : n - TAIL].argmax(1) - c) <= TOL).sum() for c in C[i]] for i in range(NV)]); vp = np.array([np.argmax(votes[i] + 1e-3 * (np.arange(K) == 0)) for i in range(NV)])
print("majority-vote pick hit %.3f" % Y[np.arange(NV), vp].mean())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run(groups, seeds=(0, 1, 2, 3, 4), epochs=80, hid=64):
    X = np.concatenate([G[k] for k in groups], 2); out = np.zeros((NV, K))
    for k in range(5):
        tr = (fold != k) & (Y.max(1) > 0); te = fold == k
        mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
        Xt = torch.from_numpy((X - mu) / sd).float().to(dev); Yt = torch.from_numpy(Y / Y.sum(1, keepdims=True).clip(1)).to(dev)
        for s in seeds:
            torch.manual_seed(s); m = nn.Sequential(nn.Linear(X.shape[2], hid), nn.GELU(), nn.Dropout(0.3), nn.Linear(hid, 1)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            for ep in range(epochs):
                m.train(); lg = m(Xt[tr]).squeeze(-1); loss = -(Yt[tr] * torch.log_softmax(lg, 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): out[te] += m(Xt[te]).squeeze(-1).cpu().numpy() / len(seeds)
    return out
res = {}
for groups in (["curve"], ["curve", "phys"], ["curve", "app"], ["curve", "det"], ["curve", "phys", "app", "det"], ["phys"], ["det"]):
    L = run(groups); pick = L.argmax(1); hit = Y[np.arange(NV), pick]; base = Y[:, 0]
    gated = {}
    for tau in (0.5, 1.0, 2.0):
        p = np.where(L[np.arange(NV), pick] - L[:, 0] > tau, pick, 0); gated[tau] = round(float(Y[np.arange(NV), p].mean()), 3)
    res["+".join(groups)] = hit.mean()
    print(f"{'+'.join(groups):20s} hit {hit.mean():.3f} | fixed {int(((base == 0) & (hit == 1)).sum())} broke {int(((base == 1) & (hit == 0)).sum())} | gated τ {gated}", flush=True)
