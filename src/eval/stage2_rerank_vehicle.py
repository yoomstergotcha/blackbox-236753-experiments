"""EXP-S2-RERANK-VEHICLE-001 — object-centric(트랙·가림·접촉 cue) 후보 재순위 검증.
11모델 strict OOF 상위 3후보(분리 >3프레임)에 대해 YOLO 박스 캐시(output/stage2_ccd_det)로 greedy IoU 트랙을 만들고
후보 시점 t 주변의 피해차량 후보(가장 큰/가까운 트랙) 상호작용 특징을 뽑아 listwise MLP·pairwise hinge로 재순위.
출력: hit, rescued/broken, margin 게이트, 라벨 74클립의 joint utility(0.35·collision + 0.35·entry(c−6)).
실행: python -m src.eval.stage2_rerank_vehicle
"""
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch import nn

z = np.load("output/s2_ens11_oof_permodel.npz"); S, vids, onset, fold = z["S"], list(z["vids"]), z["onset"], z["fold"]
E = S.mean(0); M, NV, n = S.shape; TOL, TAIL, K = 3, 3, 3
DET = Path("output/stage2_ccd_det"); FE = Path("output/stage2_ccd_feats"); VEH = {2, 3, 5, 7}
ent = pd.read_csv("output/ccd_entry_labels_v2.csv", dtype={"vid": str}); entry_off = dict(zip(ent.vid, ent.entry_offset))


def cands(e):
    e = e[: n - TAIL]; out = []
    for t in np.argsort(-e):
        if all(abs(t - c) > TOL for c in out): out.append(int(t))
        if len(out) == K: break
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (u + 1e-9)


def tracks_of(vid):
    """greedy IoU(≥0.3) 프레임 간 매칭, 1프레임 gap 허용. 반환: [{f: (box, conf)}...]"""
    per = [[] for _ in range(n)]
    if (DET / f"{vid}.npz").is_file():
        for fr, x1, y1, x2, y2, cf, cl in np.load(DET / f"{vid}.npz")["det"]:
            if int(cl) in VEH and cf >= 0.3 and 0 <= int(fr) < n: per[int(fr)].append(((x1, y1, x2, y2), float(cf)))
    tracks, active = [], []  # active: (track_idx, last_frame)
    for f in range(n):
        used = set(); new_active = []
        for ti, last in active:
            if f - last > 2: continue
            best, bj = 0.3, -1
            for j, (b, c) in enumerate(per[f]):
                if j in used: continue
                v = iou(tracks[ti][last][0], b)
                if v > best: best, bj = v, j
            if bj >= 0: tracks[ti][f] = per[f][bj]; used.add(bj); new_active.append((ti, f))
            else: new_active.append((ti, last))
        for j, (b, c) in enumerate(per[f]):
            if j not in used: tracks.append({f: (b, c)}); new_active.append((len(tracks) - 1, f))
        active = [(ti, last) for ti, last in new_active if f - last <= 2]
    return [t for t in tracks if len(t) >= 2]


def area(b): return (b[2] - b[0]) * (b[3] - b[1])


def track_feats(tracks, t):
    """후보 t의 피해차량 후보 = [t−5, t] 구간에 살아있는 트랙 중 t(또는 그 이전 마지막 프레임) 면적 최대. 16-d."""
    alive = []
    for tr in tracks:
        fs = [f for f in tr if t - 5 <= f <= t]
        if fs: alive.append((tr, max(fs)))
    f0 = np.zeros(16, np.float32)
    f0[15] = len(alive)
    if not alive: return f0
    tr, fl = max(alive, key=lambda x: area(x[0][x[1]][0]))
    b, c = tr[fl]; fs_all = sorted(tr); first, last = fs_all[0], fs_all[-1]
    def at(f):
        fs = [g for g in fs_all if g <= f]; return tr[fs[-1]] if fs else None
    prev = at(t - 5); pb = prev[0] if prev else b
    f0[0] = area(b); f0[1] = (area(b) - area(pb)) / (area(pb) + 1e-6); f0[2] = b[3]; f0[3] = abs((b[0] + b[2]) / 2 - 0.5)
    f0[4] = ((b[0] + b[2]) / 2) - ((pb[0] + pb[2]) / 2); f0[5] = t - first; f0[6] = float(last <= t + 2)  # 트랙 단절
    f0[7] = float(last >= t + 3)  # 계속 살아있음
    pc = np.mean([tr[g][1] for g in fs_all if t - 5 <= g <= t - 1]) if any(t - 5 <= g <= t - 1 for g in fs_all) else c
    post = [tr[g][1] for g in fs_all if t <= g <= t + 2]; f0[8] = pc - (np.mean(post) if post else 0.0)  # conf 급락
    f0[9] = t - fl  # 마지막 관측과의 거리(가림/소실)
    # 쌍 IoU(겹침) at t vs t−5
    def max_iou_at(f):
        bs = [tr2[g][0] for tr2 in tracks for g in [max([h for h in tr2 if h <= f], default=None)] if g is not None and f - g <= 2]
        m = 0.0
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)): m = max(m, iou(bs[i], bs[j]))
        return m, len(bs)
    m_t, n_t = max_iou_at(t); m_p, _ = max_iou_at(t - 5); f0[10] = m_t; f0[11] = m_t - m_p
    f0[12] = sum(1 for tr2 in tracks if t <= max(tr2) <= t + 3)  # 소실 트랙 수
    f0[13] = n_t; f0[14] = max((area(tr2[g][0]) for tr2 in tracks for g in tr2 if g == t), default=0.0)
    return f0


G = {"curve": [], "track": [], "app": []}; Y, C = [], []
for i, v in enumerate(vids):
    e = E[i]; am = S[:, i, : n - TAIL].argmax(1); cs = cands(e); c1 = cs[0]; trs = tracks_of(v)
    app = np.load(FE / f"{v}.npz")["app"].astype(np.float32); nov = np.r_[0, np.linalg.norm(np.diff(app, axis=0), axis=1)]
    g = {k: [] for k in G}; y, cc = [], []
    for r, c in enumerate(cs):
        w = e[max(c - 3, 0): c + 4]; pre = e[max(c - 10, 0): max(c - 3, 1)]; post = e[c + 4: c + 11] if c + 4 < n else e[-1:]
        g["curve"].append([e[c], e[c] / e[c1], r, c / n, (c - c1) / n, w.mean(), pre.mean(), post.mean(), e[c] - e[max(c - 3, 0)], e[c] - e[min(c + 3, n - 1)], (np.abs(am - c) <= TOL).mean(), S[:, i, c].std(), (e[: n - TAIL] >= 0.5 * e[c1]).sum() / n, (w >= 0.5 * e[c]).sum()])
        g["track"].append(track_feats(trs, c))
        g["app"].append([nov[max(c - 3, 0): c + 4].mean(), nov[max(c - 3, 0): c + 4].max(), nov[max(c - 9, 0): max(c - 3, 1)].mean(), nov[c + 4: c + 10].mean() if c + 4 < n else 0.0])
        y.append(abs(c - onset[i]) <= TOL); cc.append(c)
    for k in G: G[k].append(np.array(g[k], np.float32))
    Y.append(y); C.append(cc)
G = {k: np.stack(v) for k, v in G.items()}; Y = np.array(Y, np.float32); C = np.array(C)
base = Y[:, 0]; print("K", K, "| dims", {k: v.shape[2] for k, v in G.items()}, "| top1 %.3f | oracle %.3f | no-positive %d" % (base.mean(), Y.max(1).mean(), (Y.max(1) == 0).sum()))
# Group B 특징 비교(정답이 2~3위 vs 적중)
rank = np.array([next((j for j in range(K) if Y[i, j]), 99) for i in range(NV)])
A, B = rank == 0, (rank > 0) & (rank < 99)
names = ["area", "growth", "bottom_y", "center_dist", "lat_vel", "track_age", "track_break", "alive_after", "conf_drop", "since_seen", "pair_iou", "iou_jump", "lost_tracks", "n_boxes", "max_area_t", "n_alive"]
print("Group A(top1 정답) vs B(정답 2~3위) — top1 후보의 트랙 특징 중앙값:")
for j, nm in enumerate(names): print(f"  {nm:12s} A {np.median(G['track'][A, 0, j]):.3f} | B {np.median(G['track'][B, 0, j]):.3f}")
gtB = np.array([G["track"][i, rank[i]] for i in np.where(B)[0]]); t1B = G["track"][B, 0]
print("Group B: 정답 후보 vs top1 후보(같은 클립) 중앙값 — break %.2f/%.2f conf_drop %.3f/%.3f growth %.3f/%.3f iou_jump %.3f/%.3f lost %.2f/%.2f" % (np.median(gtB[:, 6]), np.median(t1B[:, 6]), np.median(gtB[:, 8]), np.median(t1B[:, 8]), np.median(gtB[:, 1]), np.median(t1B[:, 1]), np.median(gtB[:, 11]), np.median(t1B[:, 11]), np.median(gtB[:, 12]), np.median(t1B[:, 12])))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lab_idx = np.array([i for i, v in enumerate(vids) if v in entry_off]); ent_gt = np.array([onset[i] + entry_off[vids[i]] for i in lab_idx])


def run(groups, loss="listwise", seeds=(0, 1, 2, 3, 4), epochs=80, hid=64):
    X = np.concatenate([G[k] for k in groups], 2); out = np.zeros((NV, K))
    for k in range(5):
        tr = (fold != k) & (Y.max(1) > 0); te = fold == k
        mu = X[tr].reshape(-1, X.shape[2]).mean(0); sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
        Xt = torch.from_numpy((X - mu) / sd).float().to(dev); Yt = torch.from_numpy(Y).to(dev)
        for s in seeds:
            torch.manual_seed(s); m = nn.Sequential(nn.Linear(X.shape[2], hid), nn.GELU(), nn.Dropout(0.3), nn.Linear(hid, 1)).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            for _ in range(epochs):
                m.train(); lg = m(Xt[tr]).squeeze(-1); yt = Yt[tr]
                if loss == "listwise": l = -((yt / yt.sum(1, keepdim=True)) * torch.log_softmax(lg, 1)).sum(1).mean()
                else:  # pairwise hinge: 양성 − 음성 ≥ 1
                    pos = (lg * yt).sum(1) / yt.sum(1); neg = torch.where(yt > 0, torch.full_like(lg, -1e4), lg).max(1).values; l = torch.clamp(1.0 - pos + neg, min=0).mean()
                opt.zero_grad(); l.backward(); opt.step()
            m.eval()
            with torch.no_grad(): out[te] += m(Xt[te]).squeeze(-1).cpu().numpy() / len(seeds)
    return out


def report(name, L):
    pick = L.argmax(1); hit = Y[np.arange(NV), pick]
    res = f"{name:26s} hit {hit.mean():.3f} rescued {int(((base == 0) & (hit == 1)).sum()):3d} broken {int(((base == 1) & (hit == 0)).sum()):3d}"
    for tau in (0.5, 1.0, 2.0):
        p = np.where(L[np.arange(NV), pick] - L[:, 0] > tau, pick, 0); h = Y[np.arange(NV), p]
        res += f" | τ{tau}: {h.mean():.3f} (+{int(((base == 0) & (h == 1)).sum())}/−{int(((base == 1) & (h == 0)).sum())})"
    # joint utility on labeled clips (entry = c − 6)
    cj = C[lab_idx, pick[lab_idx]]; cb = C[lab_idx, 0]
    ju = lambda cc: 0.35 * (np.abs(cc - onset[lab_idx]) <= TOL).mean() + 0.35 * (np.abs(np.maximum(cc - 6, 0) - ent_gt) <= TOL).mean()
    print(res + f" | joint(n={len(lab_idx)}) base {ju(cb):.3f} → {ju(cj):.3f}", flush=True)


for groups, loss in ((["curve"], "listwise"), (["track"], "listwise"), (["curve", "track"], "listwise"), (["curve", "track"], "pairwise"), (["curve", "track", "app"], "listwise")):
    report("+".join(groups) + f" [{loss}]", run(groups, loss))
