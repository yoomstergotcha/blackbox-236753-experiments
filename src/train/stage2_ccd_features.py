"""EXP-S2-LEARN-001 — CCD 클립(10fps, 50프레임)에서 Stage2 학습용 프레임 단위 특징을 캐시한다.

특징(프레임 i):
  - 전역 이동 3신호: shift, vert, diff (predict_stage2_heuristic._s2_signals와 동일) + 각 robust z
  - Farneback 흐름 요약 13-d (stage3_motion.flow_features, 160x120 gray, stride 1)
  - ResNet18(ImageNet) 512-d (짧은 변 224 리사이즈+중앙 crop)
출력: <out>/<vid>.npz  (signals (50,6) float32, motion (50,13) float32, app (50,512) float16)
라벨은 Crash-1500.txt(onset = 첫 사고 프레임, ego)와 수동 라벨 CSV에서 학습 스크립트가 읽는다.

실행:
    python -m src.train.stage2_ccd_features --videos data/external/ccd --out output/stage2_ccd_feats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.train.stage3_motion import _radial_grid, flow_features

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def decode(path: Path):
    cap = cv2.VideoCapture(str(path))
    rgb, g320, g160 = [], [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g320.append(cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32))
        g160.append(cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA))
        r = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = r.shape[:2]
        s = 224 / min(h, w)
        nh, nw = max(224, round(h * s)), max(224, round(w * s))
        r = cv2.resize(r, (nw, nh), interpolation=cv2.INTER_AREA)
        y, x = (nh - 224) // 2, (nw - 224) // 2
        rgb.append(r[y : y + 224, x : x + 224])
    cap.release()
    return np.stack(rgb), g320, np.stack(g160)


def signals(g320):
    n = len(g320)
    win = cv2.createHanningWindow((320, 180), cv2.CV_32F)
    shift, vert, diff = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        (dx, dy), _ = cv2.phaseCorrelate(g320[i - 1], g320[i], win)
        shift[i], vert[i] = float(np.hypot(dx, dy)), float(abs(dy))
        diff[i] = float(np.abs(g320[i] - g320[i - 1]).mean())
    return np.stack([shift, vert, diff, robust_z(shift), robust_z(vert), robust_z(diff)], 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("data/external/ccd"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_ccd_feats"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    mean, std = _MEAN.to(device), _STD.to(device)
    paths = sorted(args.videos.glob("*.mp4"))
    rx, ry = _radial_grid(120, 160)
    done = 0
    with torch.inference_mode():
        for p in paths:
            out = args.out / (p.stem + ".npz")
            if out.is_file():
                done += 1
                continue
            rgb, g320, g160 = decode(p)
            n = len(rgb)
            motion = np.zeros((n, 13), np.float32)
            for i in range(n - 1):
                motion[i] = flow_features(g160[i], g160[i + 1], rx, ry)
            if n > 1:
                motion[n - 1] = motion[n - 2]
            x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255.0
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                app = backbone((x - mean) / std).float().cpu().numpy().astype(np.float16)
            np.savez(out, signals=signals(g320), motion=motion, app=app)
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(paths)}", flush=True)
    print(f"완료 {done}/{len(paths)} -> {args.out}")


if __name__ == "__main__":
    main()
