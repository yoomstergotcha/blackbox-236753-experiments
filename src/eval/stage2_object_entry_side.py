"""EXP-S2-TRACK-ENTRY-001 — 객체 중심 프로토타입: 피해차량 트랙 역추적 → ego corridor 진입 시점(entry), 트랙 횡위치 → 진입 방향(side),
충돌 직전 자유 공간 → 회피 공간(evasion). YOLO 박스 캐시(output/stage2_ccd_det, 10fps)로 greedy IoU 트랙 구성.
평가: 진입 라벨 74(±3, 상수 −6과 비교; c = GT onset 조건부 / c = 배포 11모델 OOF 예측), 방향 라벨(acc/macro-F1), 회피 라벨(acc/F1).
실행: python -m src.eval.stage2_object_entry_side
"""
import numpy as np, pandas as pd
from pathlib import Path

DET = Path("output/stage2_ccd_det"); VEH = {2, 3, 5, 7}; TOL = 3; TAIL = 3
z = np.load("output/s2_ens11_oof_permodel.npz"); vids = list(z["vids"]); onset = dict(zip(vids, z["onset"].astype(int))); E = dict(zip(vids, z["S"].mean(0))); n = z["S"].shape[2]
pred_c = {v: int(np.argmax(E[v][: n - TAIL])) for v in vids}
ent = pd.read_csv("output/ccd_entry_labels_v2.csv", dtype={"vid": str}); entry_gt = {v: onset[v] + int(o) for v, o in zip(ent.vid, ent.entry_offset) if v in onset}
man = pd.read_csv("output/ccd_manual_labels_all.csv", dtype={"vid": str}); side_gt = {v: s for v, s in zip(man.vid, man.manual_side) if isinstance(s, str) and v in onset}
eva_gt = {v: int(e) for v, e in zip(man.vid, man.manual_evasion) if e == e and v in onset}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (u + 1e-9)


def area(b): return (b[2] - b[0]) * (b[3] - b[1])


def tracks_of(vid, conf_min=0.3):
    per = [[] for _ in range(n)]
    if (DET / f"{vid}.npz").is_file():
        for fr, x1, y1, x2, y2, cf, cl in np.load(DET / f"{vid}.npz")["det"]:
            if int(cl) in VEH and cf >= conf_min and 0 <= int(fr) < n: per[int(fr)].append(((x1, y1, x2, y2), float(cf)))
    tracks, active = [], []
    for f in range(n):
        used = set(); new_active = []
        for ti, last in active:
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


def victim(tracks, c, mode="area"):
    alive = []
    for tr in tracks:
        fs = [f for f in tr if c - 5 <= f <= c]
        if fs: alive.append((tr, max(fs)))
    if not alive: return None
    if mode == "area": return max(alive, key=lambda x: area(x[0][x[1]][0]))[0]
    return min(alive, key=lambda x: abs((x[0][x[1]][0][0] + x[0][x[1]][0][2]) / 2 - 0.5) + (1 - x[0][x[1]][0][3]))[0]  # 화면 하단 중앙에 가장 가까운 트랙


def bottom_center(tr, f):
    """관측 없으면 ±2프레임 안에서 선형 보간."""
    if f in tr: b = tr[f][0]; return (b[0] + b[2]) / 2, b[3]
    lo = [g for g in tr if g < f]; hi = [g for g in tr if g > f]
    if lo and hi and hi[0] - lo[-1] <= 3:
        a, b = tr[lo[-1]][0], tr[hi[0]][0]; w = (f - lo[-1]) / (hi[0] - lo[-1])
        return ((a[0] + a[2]) / 2 * (1 - w) + (b[0] + b[2]) / 2 * w, a[3] * (1 - w) + b[3] * w)
    return None


def entry_geom(tr, c, k=0.6, y0=0.5, min_len=4):
    """c에서 역방향으로 피해차량 하단 중심이 corridor(|x−0.5| ≤ k·(y−y0)) 밖이었던 첫 프레임을 찾고, 그 다음 프레임을 entry로."""
    if tr is None: return None
    obs = sorted(g for g in tr if g <= c)
    if len(obs) < min_len: return None
    inside_prev = None
    for f in range(c, max(obs[0] - 1, -1), -1):
        p = bottom_center(tr, f)
        if p is None: continue
        inside = abs(p[0] - 0.5) <= k * max(p[1] - y0, 0.0)
        if not inside and inside_prev is not None: return inside_prev
        if inside: inside_prev = f
    return None  # 트랙 시작부터 안에 있었음(진입이 영상 시작 전)


def side_geom(tr, c, e):
    if tr is None: return None
    xs = [bottom_center(tr, f) for f in range(max(e - 3, 0), max(e, 1))]; xs = [p[0] for p in xs if p]
    if not xs: xs = [bottom_center(tr, f) for f in range(max(c - 8, 0), c)]; xs = [p[0] for p in xs if p]
    if not xs: return None
    return "LEFT" if np.mean(xs) < 0.5 else "RIGHT"


def evasion_geom(tracks, vic, c, band=0.6):
    """c−2 프레임에서 피해차량 제외 차량 박스가 화면 하단 밴드(y>band)를 좌/우 절반에서 얼마나 차지하는지 → 덜 차지한 쪽 점유율."""
    f = max(c - 2, 0); occ = [0.0, 0.0]
    for tr in tracks:
        if tr is vic or f not in tr: continue
        b = tr[f][0]
        if b[3] < band: continue
        cx = (b[0] + b[2]) / 2; occ[0 if cx < 0.5 else 1] += min(b[2], 1) - max(b[0], 0)
    return min(occ)


hit = lambda p, g: p is not None and abs(p - g) <= TOL
rows = []
for cond in ("gt", "pred"):
    C = {v: (onset[v] if cond == "gt" else pred_c[v]) for v in vids}
    for mode in ("area", "center"):
        for k in (0.3, 0.45, 0.6, 0.8, 1.0):
            hc, hg, hgate, ngeo = [], [], [], 0
            for v, g in entry_gt.items():
                c = C[v]; trs = tracks_of(v); vic = victim(trs, c, mode); eg = entry_geom(vic, c, k=k)
                hc.append(hit(max(c - 6, 0), g)); hg.append(hit(eg if eg is not None else max(c - 6, 0), g)); ngeo += eg is not None
                hgate.append(hit(eg if (eg is not None and c - eg <= 15) else max(c - 6, 0), g))
            rows.append(dict(cond=cond, victim=mode, k=k, const6=np.mean(hc), geom=np.mean(hg), geom_gate15=np.mean(hgate), n_geom=ngeo, n=len(hc)))
d = pd.DataFrame(rows); print("== entry (74 라벨, hit±3) ==\n", d.round(3).to_string(index=False))
# side
for cond in ("gt", "pred"):
    for mode in ("area", "center"):
        ok, tot, conf = 0, 0, {}
        for v, s in side_gt.items():
            c = onset[v] if cond == "gt" else pred_c[v]; trs = tracks_of(v); vic = victim(trs, c, mode); eg = entry_geom(vic, c, k=0.6); e = eg if eg is not None else max(c - 6, 0)
            p = side_geom(vic, c, e)
            if p is None: continue
            tot += 1; ok += p == s; conf[(s, p)] = conf.get((s, p), 0) + 1
        f1s = []
        for cls in ("LEFT", "RIGHT"):
            tp = conf.get((cls, cls), 0); fp = sum(v for (g, p), v in conf.items() if p == cls and g != cls); fn = sum(v for (g, p), v in conf.items() if g == cls and p != cls)
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        print(f"side [{cond}, victim={mode}] acc {ok / max(tot, 1):.3f} macroF1 {np.mean(f1s):.3f} (n={tot}/{len(side_gt)}) conf {conf}")
# evasion
for cond in ("gt", "pred"):
    vals = []
    for v, g in eva_gt.items():
        c = onset[v] if cond == "gt" else pred_c[v]; trs = tracks_of(v); vic = victim(trs, c, "area"); vals.append((evasion_geom(trs, vic, c), g))
    for thr in (0.05, 0.1, 0.2, 0.3):
        p = np.array([1 if o <= thr else 0 for o, _ in vals]); g = np.array([gg for _, gg in vals])
        f1 = []
        for cls in (0, 1):
            tp = ((p == cls) & (g == cls)).sum(); fp = ((p == cls) & (g != cls)).sum(); fn = ((p != cls) & (g == cls)).sum(); f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        print(f"evasion [{cond}] thr {thr}: acc {(p == g).mean():.3f} macroF1 {np.mean(f1):.3f} pred1 {p.mean():.2f} (n={len(g)}, label1 {g.mean():.2f})")
