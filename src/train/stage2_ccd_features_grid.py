"""EXP-S2-LEARN-007 — CCD 프레임 특징에 공간 격자 흐름 특징(grid)을 추가한다.

grid (N, 48): 160x120 Farneback 흐름(i→i+1)을 4열x3행 셀로 나눠 셀별 [mean u, mean v, mean |flow|] 36 + 셀별 프레임 차 평균 12.
마지막 프레임은 직전 값 복사. 입력 <feats>/<vid>.npz(기존 키 유지) → <out>/<vid>.npz(+grid).

실행: python -m src.train.stage2_ccd_features_grid --feats output/stage2_ccd_feats --out output/stage2_ccd_feats_v2
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

GC, GR = 4, 3


def grid_features(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    diff = np.abs(prev.astype(np.float32) - nxt.astype(np.float32)) / 255.0
    h, w = mag.shape
    out = []
    for r in range(GR):
        for c in range(GC):
            ys, xs = slice(r * h // GR, (r + 1) * h // GR), slice(c * w // GC, (c + 1) * w // GC)
            out += [flow[ys, xs, 0].mean(), flow[ys, xs, 1].mean(), mag[ys, xs].mean()]
    for r in range(GR):
        for c in range(GC):
            out.append(diff[r * h // GR : (r + 1) * h // GR, c * w // GC : (c + 1) * w // GC].mean())
    return np.array(out, dtype=np.float32)


def _job(args):
    video, src, dst = args
    if dst.is_file():
        return
    cap = cv2.VideoCapture(str(video))
    g = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA))
    cap.release()
    n = len(g)
    grid = np.zeros((n, 3 * GC * GR + GC * GR), np.float32)
    for i in range(n - 1):
        grid[i] = grid_features(g[i], g[i + 1])
    if n > 1:
        grid[n - 1] = grid[n - 2]
    z = dict(np.load(src))
    z["grid"] = grid
    np.savez(dst, **z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("data/external/ccd"))
    ap.add_argument("--feats", type=Path, default=Path("output/stage2_ccd_feats"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_ccd_feats_v2"))
    ap.add_argument("--vids", type=Path, default=None, help="처리할 vid 목록 파일(없으면 전체)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vids = [l.strip() for l in open(args.vids)] if args.vids else [p.stem for p in sorted(args.feats.glob("*.npz"))]
    jobs = [(args.videos / f"{v}.mp4", args.feats / f"{v}.npz", args.out / f"{v}.npz") for v in vids]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=8), 1):
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print(f"완료 {len(jobs)} -> {args.out}")


if __name__ == "__main__":
    main()
