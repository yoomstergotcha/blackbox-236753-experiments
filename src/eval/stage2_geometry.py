"""객체 트랙 위의 ego 접촉 기하 표현 (EXP-S2-GEO-001). 순수 numpy — 추론 스니펫에 그대로 삽입된다.

입력: per[t] = [(box(x1,y1,x2,y2 정규화 0~1), conf), ...] (샘플링된 프레임열, 검출 클래스는 차량만)
출력: (n, GEO_DIM) float32. 비율·변화율은 '샘플 간' 단위라 stride를 10fps 등가로 맞추면 fps에 무관.
"""
import numpy as np

GEO_DIM = 24


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def build_tracks(per, iou_min=0.3, gap=2):
    """greedy IoU 트랙. 반환: [{frame: (box, conf)}...] (길이 ≥ 2)"""
    tracks, active = [], []
    for f, bs in enumerate(per):
        used, new_active = set(), []
        for ti, last in active:
            best, bj = iou_min, -1
            for j, (b, c) in enumerate(bs):
                if j in used:
                    continue
                v = _iou(tracks[ti][last][0], b)
                if v > best:
                    best, bj = v, j
            if bj >= 0:
                tracks[ti][f] = bs[bj]
                used.add(bj)
                new_active.append((ti, f))
            else:
                new_active.append((ti, last))
        for j, (b, c) in enumerate(bs):
            if j not in used:
                tracks.append({f: (b, c)})
                new_active.append((len(tracks) - 1, f))
        active = [(ti, last) for ti, last in new_active if f - last <= gap]
    return [t for t in tracks if len(t) >= 2]


def geometry_features(per):
    """프레임별 24-d: 주 트랙(하단 y>0.35 중 최대 면적) 기하 15 + 집계 8 + 존재 1."""
    n = len(per)
    tracks = build_tracks(per)
    # 트랙별 프레임 상태(보간 없이 관측만) → 프레임별 [(track_id, box, conf, age)]
    at = [[] for _ in range(n)]
    for ti, tr in enumerate(tracks):
        first = min(tr)
        for f, (b, c) in tr.items():
            at[f].append((ti, b, c, f - first))
    prev_box = {}  # track_id -> (frame, box)
    out = np.zeros((n, GEO_DIM), np.float32)
    for f in range(n):
        items = at[f]
        if not items:
            out[f, 12] = 1.0  # TTC 근사: 객체 없음 = 임박 아님
            continue
        boxes = [b for _, b, _, _ in items]
        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        cand = [i for i, b in enumerate(boxes) if b[3] > 0.35]
        pi = int(max(cand, key=lambda i: areas[i])) if cand else int(np.argmax(areas))
        ti, b, c, age = items[pi]
        w, h, a, y2, cx = b[2] - b[0], b[3] - b[1], areas[pi], b[3], (b[0] + b[2]) / 2
        d = float(np.hypot(cx - 0.5, 1.0 - y2))
        corr = 1.0 if abs(cx - 0.5) <= 0.35 * max(y2 - 0.5, 0.0) + 0.05 else 0.0
        pb = prev_box.get(ti)
        if pb is not None and f - pb[0] <= 3:
            dt = f - pb[0]; ob = pb[1]
            ow = ob[2] - ob[0]; oa = (ob[2] - ob[0]) * (ob[3] - ob[1])
            dw = (w - ow) / dt; da = (a - oa) / (oa + 1e-6) / dt; dy2 = (y2 - ob[3]) / dt; dcx = (cx - (ob[0] + ob[2]) / 2) / dt
        else:
            dw = da = dy2 = dcx = 0.0
        ttc = float(np.clip(w / max(dw, 1e-3), 0, 50)) / 50.0 if dw > 0 else 1.0  # 폭 증가율 기반 TTC 근사(작을수록 임박)
        out[f, :15] = [a, w, h, y2, cx - 0.5, abs(cx - 0.5), d, corr, dw, da, dy2, dcx, ttc, min(age, 30) / 30.0, c]
        # 집계
        ds = [float(np.hypot((bb[0] + bb[2]) / 2 - 0.5, 1.0 - bb[3])) for bb in boxes]
        das = []
        for tj, bb, cc, ag in items:
            pj = prev_box.get(tj)
            if pj is not None and f - pj[0] <= 3:
                oa = (pj[1][2] - pj[1][0]) * (pj[1][3] - pj[1][1]); das.append(((bb[2] - bb[0]) * (bb[3] - bb[1]) - oa) / (oa + 1e-6) / (f - pj[0]))
        miou = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                miou = max(miou, _iou(boxes[i], boxes[j]))
        out[f, 15:23] = [len(boxes), float(areas.sum()), float(areas.max()), min(ds), max(das) if das else 0.0, sum(1 for bb in boxes if abs((bb[0] + bb[2]) / 2 - 0.5) <= 0.35 * max(bb[3] - 0.5, 0.0) + 0.05), miou, float(np.mean([bb[3] for bb in boxes]))]
        out[f, 23] = 1.0
        for tj, bb, cc, ag in items:
            prev_box[tj] = (f, bb)
    return out


def per_frame_from_det(det, n, conf_min=0.3, classes=(2, 3, 5, 7), stride=1):
    """캐시 det (M,7) [frame,x1,y1,x2,y2,conf,cls] → per[t] (stride 샘플링: 프레임 t*stride)"""
    per = [[] for _ in range(n)]
    for fr, x1, y1, x2, y2, cf, cl in det:
        f = int(fr)
        if int(cl) in classes and cf >= conf_min and f % stride == 0 and 0 <= f // stride < n:
            per[f // stride].append(((float(x1), float(y1), float(x2), float(y2)), float(cf)))
    return per
