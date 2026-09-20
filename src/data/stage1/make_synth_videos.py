"""EXP-S1-SYNTH-007 — '재촬영 합성 → 저비트율 mpeg4(FMP4) 재인코딩' 파이프라인으로 합성 영상 생성.

LB 실측 해석(제출 7: 코덱 규칙 실패, S1 0.393): 비공개 Stage1 파일은 두 클래스 모두 원본과 같은 mpeg4(FMP4) 계열로
인코딩돼 있을 가능성이 높다. 그동안의 합성(v1~v4)은 프레임에 흔적을 입힌 뒤 *다시 인코딩하지 않고* 학습해
저비트율 mpeg4가 지워 버리는 미세 흔적(모아레·그레인)에 의존했다. 여기서는 영상 단위로
  원본 클립(mpeg4 ~0.4-1.2Mbps, CCD 원본 규격 1280x720 10fps 5초)
  → 프레임별 재촬영 합성(클립당 파라미터 고정, v4 apply_capture 또는 미세 apply_rerecord)
  → mpeg4(80%) / libx264(20%) 저비트율 재인코딩
을 거쳐 이중 압축까지 포함한 RERECORDED 영상을 만들고, 같은 소스의 ORIGINAL 영상과 쌍으로 학습한다.

소스: 공개 Stage1 원본 5개(CCD, 그대로 ORIGINAL) + comma2k19 RAV4/Civic 세그먼트(5초 창, 16:9 중앙 크롭 → 1280x720 10fps).
출력: <out>/<clip_id>/{orig.mp4, rerec_capture.mp4, rerec_subtle.mp4} + <out>/clips.csv
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from src.data.stage1.synth_rerecord import apply_capture, apply_rerecord, sample_capture_params, sample_params

W, H, FPS, N = 1280, 720, 10, 50


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def to_ccd_format(frames: list[np.ndarray]) -> list[np.ndarray]:
    """16:9 중앙 크롭 후 1280x720."""
    out = []
    for f in frames:
        h, w = f.shape[:2]
        th = int(round(w * 9 / 16))
        if th <= h:
            y = (h - th) // 2
            f = f[y : y + th]
        else:
            tw = int(round(h * 16 / 9))
            x = (w - tw) // 2
            f = f[:, x : x + tw]
        out.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    return out


def encode(frames: list[np.ndarray], path: Path, codec: str, bitrate: int) -> None:
    out = av.open(str(path), "w")
    st = out.add_stream(codec, rate=FPS)
    st.width, st.height, st.pix_fmt, st.bit_rate = W, H, "yuv420p", int(bitrate)
    if codec == "mpeg4":
        st.options = {"g": "12"}
    for f in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


def comma_clip(video_path: Path, rng: np.random.Generator) -> list[np.ndarray]:
    frames = decode(video_path)  # 20fps 1200프레임
    if len(frames) < 2 * N:
        raise ValueError(f"too short: {video_path}")
    start = int(rng.integers(0, len(frames) - 2 * N + 1))
    return to_ccd_format(frames[start : start + 2 * N : 2])


def rerecord(frames: list[np.ndarray], mode: str, rng: np.random.Generator) -> list[np.ndarray]:
    if mode == "capture":
        p = sample_capture_params(rng)
        return [apply_capture(f, p, rng) for f in frames]
    p = sample_params(rng)
    return [apply_rerecord(f, p, rng) for f in frames]


def _job(args_tuple):
    clip_id, src, kind, out_dir, h264_ratio, seed = args_tuple
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    rows = []
    try:
        d = Path(out_dir) / clip_id
        d.mkdir(parents=True, exist_ok=True)
        if kind == "official":
            (d / "orig.mp4").write_bytes(Path(src).read_bytes())  # 공개 원본은 CCD 인코딩 그대로
        else:
            encode(comma_clip(Path(src), rng), d / "orig.mp4", "mpeg4", rng.uniform(400_000, 1_200_000))
        rows.append({"clip_id": clip_id, "kind": kind, "label": "ORIGINAL", "variant": "orig", "path": str(d / "orig.mp4")})
        base = decode(d / "orig.mp4")  # 실제 인코딩을 거친 프레임에서 시작(이중 압축 재현)
        for mode in ("capture", "subtle"):
            codec = "libx264" if rng.random() < h264_ratio else "mpeg4"
            br = rng.uniform(400_000, 1_500_000) if codec == "mpeg4" else rng.uniform(800_000, 1_800_000)
            encode(rerecord(base, mode, rng), d / f"rerec_{mode}.mp4", codec, br)
            rows.append({"clip_id": clip_id, "kind": kind, "label": "RERECORDED", "variant": f"{mode}/{codec}", "path": str(d / f"rerec_{mode}.mp4")})
    except Exception as e:  # noqa: BLE001
        print("skip", clip_id, repr(e)[:80], flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("output/stage1_synth_videos"))
    ap.add_argument("--comma-paths", type=Path, default=Path("output/comma2k19_multi_v5/video_paths.csv"))
    ap.add_argument("--n-rav4", type=int, default=150)
    ap.add_argument("--n-civic", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--h264-ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    jobs = []
    for vp in sorted(Path("data/stage1/original").glob("*.mp4")):
        jobs.append((f"official_{vp.stem}", str(vp), "official", args.out, args.h264_ratio, int(rng.integers(1 << 30))))
    vpaths = pd.read_csv(args.comma_paths)
    is_rav = vpaths.segment_id.str.startswith("b0c9d2329ad1606b")
    rav = vpaths[is_rav].sample(n=min(args.n_rav4, int(is_rav.sum())), random_state=args.seed)
    civ = vpaths[vpaths.segment_id.str.startswith("99c94dc769b5d96e")]
    civ = civ.sample(n=min(args.n_civic, len(civ)), random_state=args.seed) if len(civ) else civ
    for kind, df in (("rav4", rav), ("civic", civ)):
        for r in df.itertuples(index=False):
            jobs.append((f"{kind}_{r.segment_id.replace('/', '__')}", r.video_path, kind, args.out, args.h264_ratio, int(rng.integers(1 << 30))))
    print(f"{len(jobs)} clips, workers={args.workers}", flush=True)
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.extend(res)
            if k % 25 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    with open(args.out / "clips.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "kind", "label", "variant", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} videos -> {args.out / 'clips.csv'}")


if __name__ == "__main__":
    main()
