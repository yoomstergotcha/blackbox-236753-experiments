"""YOLO 박스 캐시(output/stage2_ccd_det, 10fps CCD) → greedy IoU 트랙, 피해차량 선택, 하단 중심, 자유 공간 점유 (stage2_object_entry_side의 공용 함수)."""
import numpy as np
from pathlib import Path

DET = Path("output/stage2_ccd_det"); VEH = {2, 3, 5, 7}; N = 50


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (u + 1e-9)


def area(b): return (b[2] - b[0]) * (b[3] - b[1])


def tracks_of(vid, conf_min=0.3, n=N):
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
    return min(alive, key=lambda x: abs((x[0][x[1]][0][0] + x[0][x[1]][0][2]) / 2 - 0.5) + (1 - x[0][x[1]][0][3]))[0]


def bottom_center(tr, f):
    if f in tr: b = tr[f][0]; return (b[0] + b[2]) / 2, b[3]
    lo = [g for g in tr if g < f]; hi = [g for g in tr if g > f]
    if lo and hi and hi[0] - lo[-1] <= 3:
        a, b = tr[lo[-1]][0], tr[hi[0]][0]; w = (f - lo[-1]) / (hi[0] - lo[-1])
        return ((a[0] + a[2]) / 2 * (1 - w) + (b[0] + b[2]) / 2 * w, a[3] * (1 - w) + b[3] * w)
    return None


def evasion_geom(tracks, vic, c, band=0.6):
    f = max(c - 2, 0); occ = [0.0, 0.0]
    for tr in tracks:
        if tr is vic or f not in tr: continue
        b = tr[f][0]
        if b[3] < band: continue
        cx = (b[0] + b[2]) / 2; occ[0 if cx < 0.5 else 1] += min(b[2], 1) - max(b[0], 0)
    return min(occ)
