"""EXP-S3-MOTION-001 — 프레임 간 dense optical flow(Farneback, 160x120 gray)로 ego-motion 요약 특징을 뽑아 캐시.

동기: Stage3 라벨(가감속/조향)은 본질적으로 카메라 자체의 움직임인데, frozen ImageNet ResNet18
프레임 특징의 평균은 '장면이 무엇인가'만 담고 '어떻게 움직이는가'는 거의 담지 못한다
(EXP-S3-CLASS-001 accel_f1 0.57이 병목). 광류의 팽창률(divergence)은 전진 속도, 수평 평균 흐름은
요(yaw)/조향, 수직 평균 흐름은 제동 시 피칭의 직접 단서다.

저장: (N, 2, K) float32 — 축1은 stride 1(i→i+1)과 stride 2(i→i+2).
stride 2는 비공개 Stage3 영상이 10fps(comma2k19/공개 OPEN은 20fps)인 것을 시뮬레이션하기 위한 것:
20fps 영상에서 짝수 프레임만 취해 stride-2 흐름을 쓰면 10fps 영상의 stride-1 흐름과 같은 시간 간격(0.1s)이 된다.
(컨테이너 CAP_PROP_FPS는 공개 OPEN에서 479.78로 엉터리라 fps를 읽어 쓸 수 없다 — 학습을 두 fps에 모두 노출시킨다.)

실행:
    python -m src.train.stage3_motion --video-paths output/comma2k19_multi/video_paths.csv --out output/stage3_motion
    python -m src.train.stage3_motion --video-paths output/open_video_paths.csv --out output/stage3_motion_open
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FLOW_W, FLOW_H = 160, 120
STRIDES = (1, 2)
MOTION_NAMES = [
    "mag_mean",  # 평균 흐름 크기(px/step) — 속도 proxy
    "u_mean",  # 수평 평균 — 요/조향
    "v_mean",  # 수직 평균 — 피칭(제동/가속)
    "div",  # 방사 성분 평균(팽창률) — 전진 속도
    "mag_std",
    "u_left",  # 좌반부 수평 — 회전 vs 병진 구분
    "u_right",
    "v_top",
    "v_bottom",
    "mag_bottom",  # 근거리(도로) 흐름 — 속도에 민감
    "mag_top",
    "static_frac",  # |flow|<0.15px 비율 — 정지 단서
    "absdiff",  # 프레임 차 평균/255 — 가장 싼 움직임 척도
]
MOTION_DIM = len(MOTION_NAMES)


def _radial_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xs - (w - 1) / 2) / (w / 2)
    yn = (ys - (h - 1) / 2) / (h / 2)
    r = np.sqrt(xn * xn + yn * yn) + 1e-6
    return xn / r, yn / r


def decode_gray_small(path: Path) -> np.ndarray:
    """(N, FLOW_H, FLOW_W) uint8. 추론 snippet은 같은 리사이즈(INTER_AREA)를 써야 한다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA))
    cap.release()
    if not out:
        raise RuntimeError(f"디코딩된 프레임이 없습니다: {path}")
    return np.stack(out)


def flow_features(prev: np.ndarray, nxt: np.ndarray, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    h, w = mag.shape
    return np.array(
        [
            mag.mean(), u.mean(), v.mean(), float((u * rx + v * ry).mean()), mag.std(),
            u[:, : w // 2].mean(), u[:, w // 2 :].mean(), v[: h // 2].mean(), v[h // 2 :].mean(),
            mag[h // 2 :].mean(), mag[: h // 2].mean(), float((mag < 0.15).mean()),
            float(np.abs(prev.astype(np.float32) - nxt.astype(np.float32)).mean() / 255.0),
        ],
        dtype=np.float32,
    )


def motion_features(gray: np.ndarray, strides: tuple[int, ...] = STRIDES) -> np.ndarray:
    """(N, len(strides), MOTION_DIM). 마지막 s개 프레임은 마지막 유효값을 복제."""
    n = len(gray)
    rx, ry = _radial_grid(gray.shape[1], gray.shape[2])
    out = np.zeros((n, len(strides), MOTION_DIM), dtype=np.float32)
    for si, s in enumerate(strides):
        last = max(n - s - 1, 0)
        for i in range(max(n - s, 0)):
            out[i, si] = flow_features(gray[i], gray[i + s], rx, ry)
        for i in range(max(n - s, 0), n):
            out[i, si] = out[last, si]
    return out


def _process(job: tuple[str, str, str]) -> tuple[str, int, float]:
    segment_id, video_path, out_dir = job
    cv2.setNumThreads(1)
    t0 = time.time()
    out_path = Path(out_dir) / (segment_id.replace("/", "__") + ".npy")
    gray = decode_gray_small(Path(video_path))
    np.save(out_path, motion_features(gray))
    return segment_id, len(gray), time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-paths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vp = pd.read_csv(args.video_paths)
    jobs = [(r.segment_id, r.video_path, str(args.out)) for r in vp.itertuples(index=False) if not (args.out / (r.segment_id.replace("/", "__") + ".npy")).is_file()]
    print(f"{len(vp)} segments, {len(jobs)} to do, workers={args.workers}", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (sid, n, dt) in enumerate(pool.imap_unordered(_process, jobs), 1):
            if k % 10 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {sid} frames={n} {dt:.1f}s/seg, elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"완료 -> {args.out}")


if __name__ == "__main__":
    main()


LOGRATIO_IDX = [MOTION_NAMES.index(n) for n in ("mag_mean", "div", "mag_bottom", "mag_top")]
DIFF_IDX = [MOTION_NAMES.index("static_frac")]
DEFAULT_HORIZONS = (16, 32, 64, 128)


def motion_clip_features(m: np.ndarray, frame_idx: np.ndarray, n_frames: int = 16, horizons: tuple[int, ...] = ()) -> np.ndarray:
    """m: (N, K) 프레임별 motion 특징(한 stride). 각 frame_idx에 대해 ResNet 특징과 같은 16프레임 윈도우
    [i-8, i+8)에서 [mean(K), std(K), 뒤8평균-앞8평균(K)] concat -> 3K.
    horizons가 있으면 각 H(프레임)에 대해 [i, i+H) 평균 / [i-H, i) 평균의 log-ratio(mag_mean, div, mag_bottom, mag_top;
    깊이 스케일에 불변인 '속도 변화율' proxy)와 static_frac 차이를 덧붙인다 -> 3K + 5*len(horizons).
    EXP-S3-MOTION-001 사전 분석: div log-ratio와 CAN 가속도의 상관이 H=16(0.8s) 0.15 -> H=64(3.2s) 0.34로 지평이 길수록 커진다.
    경계(앞/뒤 윈도우가 비는 곳)는 0."""
    n, k = m.shape
    h = n_frames // 2
    out = np.empty((len(frame_idx), 3 * k), dtype=np.float32)
    m32 = m.astype(np.float32)
    for r, i in enumerate(frame_idx):
        idx = np.clip(int(i) - h + np.arange(n_frames), 0, n - 1)
        w = m32[idx]
        out[r, :k] = w.mean(0)
        out[r, k : 2 * k] = w.std(0)
        out[r, 2 * k :] = w[h:].mean(0) - w[:h].mean(0)
    if not horizons:
        return out
    c = np.concatenate([np.zeros((1, k), np.float64), np.cumsum(m32.astype(np.float64), 0)])
    i = np.clip(frame_idx.astype(int), 0, n - 1)
    extras = [out]
    for H in horizons:
        lo, hi = np.clip(i - H, 0, n), np.clip(i + H, 0, n)
        nb, na = np.maximum(i - lo, 1)[:, None], np.maximum(hi - i, 1)[:, None]
        before, after = (c[i] - c[lo]) / nb, (c[hi] - c[i]) / na
        valid = ((i - lo) > 0) & ((hi - i) > 0)
        lr = np.log(np.maximum(after[:, LOGRATIO_IDX], 0) + 1e-3) - np.log(np.maximum(before[:, LOGRATIO_IDX], 0) + 1e-3)
        df = after[:, DIFF_IDX] - before[:, DIFF_IDX]
        ex = np.concatenate([lr, df], 1).astype(np.float32)
        ex[~valid] = 0.0
        extras.append(ex)
    return np.concatenate(extras, 1)
