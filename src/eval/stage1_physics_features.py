"""EXP-S1-PHYS-001 — 재촬영의 물리 흔적 특징(코덱/메타데이터 불사용) 계산 및 그룹별 분포 비교.

영상당 최대 40프레임(균등 샘플, 320x180 gray)에서:
  static_frac : 시간 표준편차 < 1.0 인 픽셀 비율 (원본의 타임스탬프/로고/테두리는 완전 정지, 재촬영은 손떨림·노이즈로 0에 가까움)
  jitter      : 연속 프레임 전역 이동(phase correlation)의 2차 차분 RMS (손떨림 고주파 성분)
  hf_ratio    : 2D 스펙트럼 고주파(반경 상위 40%) 에너지 비율 평균 (블러/화질 손실)
  moire       : 중고주파 대역 스펙트럼 peak / median (주기적 격자 패턴)
  border_dark : 바깥 4% 테두리 픽셀 중 시간 평균 < 24 인 비율 (베젤/검은 테두리)
  lum_flicker : 프레임 평균 밝기의 2차 차분 RMS (화면 주사/노출 깜빡임)
출력: output/s1_physics_feats.csv (group, path, 특징들) + 그룹별 중앙값/AUC 요약.
"""
from __future__ import annotations

import argparse
import glob
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

W, H, NF = 320, 180, 40


def physics_features(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = set(np.linspace(0, max(n - 1, 0), min(NF, max(n, 1))).astype(int).tolist())
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in idx:
            frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32))
        i += 1
        if len(frames) >= NF:
            break
    cap.release()
    if len(frames) < 4:
        return {"static_frac": np.nan}
    g = np.stack(frames)
    static_frac = float((g.std(0) < 1.0).mean())
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    shifts = np.array([cv2.phaseCorrelate(g[i - 1], g[i], win)[0] for i in range(1, len(g))])
    jitter = float(np.sqrt((np.diff(shifts, 2, axis=0) ** 2).sum(1).mean())) if len(shifts) > 3 else 0.0
    ys, xs = np.mgrid[0:H, 0:W]
    r = np.sqrt(((ys - H / 2) / (H / 2)) ** 2 + ((xs - W / 2) / (W / 2)) ** 2)
    hf, mo = [], []
    for f in g[:: max(1, len(g) // 8)][:8]:
        spec = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean())))
        tot = spec.sum() + 1e-6
        hf.append(float(spec[r > 0.6].sum() / tot))
        band = spec[(r > 0.25) & (r < 0.9)]
        mo.append(float(band.max() / (np.median(band) + 1e-6)))
    m = g.mean(0)
    bw, bh = int(W * 0.04), int(H * 0.04)
    border = np.concatenate([m[:bh].ravel(), m[-bh:].ravel(), m[:, :bw].ravel(), m[:, -bw:].ravel()])
    lum = g.mean((1, 2))
    return {
        "static_frac": static_frac,
        "jitter": jitter,
        "hf_ratio": float(np.mean(hf)),
        "moire": float(np.median(mo)),
        "border_dark": float((border < 24).mean()),
        "lum_flicker": float(np.sqrt(np.mean(np.diff(lum, 2) ** 2))) if len(lum) > 3 else 0.0,
    }


def _job(a):
    group, path = a
    try:
        return {"group": group, "path": path, **physics_features(path)}
    except Exception as exc:  # noqa: BLE001
        return {"group": group, "path": path, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("output/s1_physics_feats.csv"))
    args = ap.parse_args()
    random.seed(5)
    groups = {
        "ccd_orig": random.sample(sorted(glob.glob("data/external/ccd/0*.mp4")), args.n),
        "comma_orig": random.sample(sorted(glob.glob("output/stage1_synth_videos_v8/*/orig.mp4")), args.n),
        "synth_capture": random.sample(sorted(glob.glob("output/stage1_synth_videos_v8/*/rerec_capture.mp4")), args.n),
        "synth_subtle": random.sample(sorted(glob.glob("output/stage1_synth_videos_v8/*/rerec_subtle.mp4")), args.n),
        "public_O": sorted(glob.glob("sample_evaluation_data/stage1/videos/SAMPLE_S1_O_*.mp4")),
        "public_R": sorted(glob.glob("sample_evaluation_data/stage1/videos/SAMPLE_S1_R_*.mp4")),
    }
    jobs = [(g, p) for g, ps in groups.items() for p in ps]
    with Pool(args.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)
    feats = ["static_frac", "jitter", "hf_ratio", "moire", "border_dark", "lum_flicker"]
    print(d.groupby("group")[feats].median().round(4).to_string())
    from sklearn.metrics import roc_auc_score

    orig = d[d.group.isin(["ccd_orig", "comma_orig"])]
    for g in ["synth_capture", "synth_subtle", "public_R"]:
        neg = d[d.group == g]
        print(g, {f: round(roc_auc_score(np.r_[np.zeros(len(orig)), np.ones(len(neg))], np.r_[orig[f], neg[f]]), 3) for f in feats})


if __name__ == "__main__":
    main()
