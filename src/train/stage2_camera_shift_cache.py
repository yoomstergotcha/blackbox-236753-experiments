"""EXP-S2-CAMSHIFT-001 — 카메라 도메인 이동 검증용 특징 캐시: CCD OOF 클립에 영상 변환을 적용한 뒤 19-d 특징 계산.

변환: gamma(0.6/1.6), jpeg(q=25), lowres(640x360→원복), zoom(중앙 70% 크롭→원복, 화각 변화), noise(σ=12), blur(σ=1.5), hflip.
출력: <out>/<vid>.npz  키 = 변환명 → (N,19)
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.train.stage2_multiscale_cache import _decode, feats19
from src.train.stage3_motion import _radial_grid


def transform(frames, name, rng):
    out = []
    for g in frames:  # gray uint8 (H,W)
        h, w = g.shape
        if name == "orig":
            x = g
        elif name.startswith("gamma"):
            gam = float(name[5:]); x = (255.0 * (g / 255.0) ** gam).astype(np.uint8)
        elif name == "jpeg25":
            ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 25]); x = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        elif name == "lowres":
            x = cv2.resize(cv2.resize(g, (640, 360), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        elif name == "zoom":
            ch, cw = int(h * 0.7), int(w * 0.7); y0, x0 = (h - ch) // 2, (w - cw) // 2; x = cv2.resize(g[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        elif name == "noise":
            x = np.clip(g.astype(np.float32) + rng.normal(0, 12, g.shape), 0, 255).astype(np.uint8)
        elif name == "blur":
            x = cv2.GaussianBlur(g, (0, 0), 1.5)
        elif name == "hflip":
            x = g[:, ::-1]
        else:
            raise ValueError(name)
        out.append(np.ascontiguousarray(x))
    return out


NAMES = ["orig", "gamma0.6", "gamma1.6", "jpeg25", "lowres", "zoom", "noise", "blur", "hflip"]


def _job(args):
    vid, out = args
    dst = out / f"{vid}.npz"
    if dst.is_file():
        return
    rx, ry = _radial_grid(120, 160)
    fr = _decode(Path("data/external/ccd") / f"{vid}.mp4")
    rng = np.random.default_rng(int(vid))
    np.savez(dst, **{n: feats19(transform(fr, n, rng), rx, ry) for n in NAMES})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("output/stage2_ms_cache/index.csv"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_camshift_cache"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(args.index, dtype={"vid": str}).head(args.n)
    with Pool(args.workers) as p:
        for i, _ in enumerate(p.imap_unordered(_job, [(v, args.out) for v in idx.vid], chunksize=4), 1):
            if i % 50 == 0:
                print(f"{i}/{len(idx)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
