"""EXP-S2-INV-001 — Stage2 불변성 검증용 다중 스케일 특징 캐시.

CCD OOF 클립(fold당 n개)마다 (a) 원본 10fps, (b) ×3 선형 보간(합성 30fps) 프레임열을 만들고, stride 1~6로 솎은 뒤
19-d 물리 특징(전역 이동 3 + robust z 3 + Farneback 13)을 계산해 저장한다. 라벨/모델과 무관하므로 한 번만 계산.
출력: <out>/<vid>.npz  키 = f"{rate}_s{stride}" (rate ∈ {10,30}), 'onset10'
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from src.train.stage2_ccd_features import robust_z, signals
from src.train.stage3_motion import _radial_grid, flow_features

STRIDES = (1, 2, 3, 4, 6)


def _decode(path: Path):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return fr


def _interp3(fr):
    out = []
    for i in range(len(fr) - 1):
        a, b = fr[i].astype(np.float32), fr[i + 1].astype(np.float32)
        out += [fr[i], np.clip((2 * a + b) / 3, 0, 255).astype(np.uint8), np.clip((a + 2 * b) / 3, 0, 255).astype(np.uint8)]
    out.append(fr[-1])
    return out


def feats19(gray_frames, rx, ry):
    g320 = [cv2.resize(x, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32) for x in gray_frames]
    g160 = [cv2.resize(x, (160, 120), interpolation=cv2.INTER_AREA) for x in gray_frames]
    n = len(g320)
    mo = np.zeros((n, 13), np.float32)
    for i in range(n - 1):
        mo[i] = flow_features(g160[i], g160[i + 1], rx, ry)
    if n > 1:
        mo[n - 1] = mo[n - 2]
    return np.concatenate([signals(g320), mo], 1)


def _job(args):
    vid, onset, videos, out = args
    dst = out / f"{vid}.npz"
    if dst.is_file():
        return
    rx, ry = _radial_grid(120, 160)
    fr = _decode(videos / f"{vid}.mp4")
    data = {"onset10": np.int64(onset)}
    for rate, frames in (("10", fr), ("30", _interp3(fr))):
        for s in STRIDES:
            sub = frames[::s]
            if len(sub) < 8:
                continue
            data[f"{rate}_s{s}"] = feats19(sub, rx, ry)
    np.savez(dst, **data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("data/external/ccd"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_ms_cache"))
    ap.add_argument("--per-fold", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from src.train.train_stage2_collision import load_ann

    ann = load_ann(Path("data/external/ccd/Crash-1500.txt"))
    ann = ann[ann.ego & (ann.onset >= 0)]
    public = {"000001", "000002", "000003", "000004", "000005"}
    pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
    rng = np.random.default_rng(20260825)
    srcs = pool.src.to_numpy()
    fold_of = {s: i % 5 for i, s in enumerate(rng.permutation(sorted(set(srcs))))}
    pool["fold"] = [fold_of[s] for s in srcs]
    sub = pool.groupby("fold").head(args.per_fold).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    sub[["vid", "onset", "fold"]].to_csv(args.out / "index.csv", index=False)
    jobs = [(r.vid, int(r.onset), args.videos, args.out) for r in sub.itertuples()]
    with Pool(args.workers) as p:
        for i, _ in enumerate(p.imap_unordered(_job, jobs), 1):
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", len(jobs), "->", args.out)


if __name__ == "__main__":
    main()
