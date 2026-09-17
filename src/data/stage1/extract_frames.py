"""Stage1 학습·검증용 프레임 추출 — 각 소스 영상에서 N개 프레임을 균등 샘플링해 PNG(무손실)로
저장하고 manifest CSV를 만든다. ORIGINAL 소스만 추출한다 — RERECORDED는 학습 시 synth_rerecord로
프레임에서 즉석 합성한다. 예외: 공식 data/stage1/rerecorded 5건은 실제(DACON식) 재녹화 예시라
검증 전용으로 그대로 추출한다.

manifest 컬럼: source_id, group, domain, label, frame_path
  group: leakage 방지용 split 키(같은 원본에서 나온 프레임/파생본은 항상 같은 split)
  domain: comma2k19 / open_stage3 / official_stage1
  label: ORIGINAL 또는 RERECORDED(공식 검증 5건만)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        raise RuntimeError(f"디코딩 실패: {video_path}")
    idx = np.linspace(0, len(frames) - 1, num=min(n, len(frames))).round().astype(int)
    return [frames[i] for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("output/stage1_frames"))
    parser.add_argument("--comma-root", type=Path, default=Path("data/external/comma2k19_chunk1_subset"))
    parser.add_argument("--comma-frames", type=int, default=10)
    parser.add_argument("--open-frames", type=int, default=20)
    parser.add_argument("--official-frames", type=int, default=10)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    def dump(frames, source_id, group, domain, label):
        d = args.out / source_id.replace("/", "__").replace("|", "_")
        d.mkdir(parents=True, exist_ok=True)
        for i, bgr in enumerate(frames):
            p = d / f"{i:03d}.png"
            cv2.imwrite(str(p), bgr)
            rows.append({"source_id": source_id, "group": group, "domain": domain, "label": label, "frame_path": str(p)})

    for vp in sorted(args.comma_root.rglob("video.hevc")):
        seg_dir = vp.parent
        route = seg_dir.parent.name
        source_id = f"{route}/{seg_dir.name}"
        dump(sample_frames(vp, args.comma_frames), source_id, route, "comma2k19", "ORIGINAL")

    example = Path("data/external/comma2k19_example/segment/video.hevc")
    if example.is_file():
        dump(sample_frames(example, args.comma_frames), "comma_example/40", "b0c9d2329ad1606b_2018-08-02--08-34-47", "comma2k19", "ORIGINAL")

    for vp in sorted(Path("data/stage3/videos").glob("*.mp4")):
        dump(sample_frames(vp, args.open_frames), f"open/{vp.stem}", f"open_{vp.stem}", "open_stage3", "ORIGINAL")

    for kind, label in [("original", "ORIGINAL"), ("rerecorded", "RERECORDED")]:
        for vp in sorted((Path("data/stage1") / kind).glob("*.mp4")):
            dump(sample_frames(vp, args.official_frames), f"official_{kind}/{vp.stem}", f"official_{vp.stem}", "official_stage1", label)

    manifest = args.out / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "group", "domain", "label", "frame_path"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} frames -> {manifest}")


if __name__ == "__main__":
    main()
