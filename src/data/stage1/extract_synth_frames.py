"""make_synth_videos.py 출력(clips.csv)의 각 영상에서 프레임을 균등 샘플링해 Stage1 manifest를 만든다.

manifest 컬럼은 extract_frames.py와 동일(source_id, group, domain, label, frame_path):
  source_id = <clip_id>/<variant>, group = <clip_id>(같은 소스는 같은 split), domain = synth_video_<kind>, label = ORIGINAL|RERECORDED.
기존 공식 5쌍 행(output/stage1_frames/manifest.csv, domain official_stage1)을 뒤에 붙인다.

실행:
    python -m src.data.stage1.extract_synth_frames --clips output/stage1_synth_videos/clips.csv --out output/stage1_frames_v5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd


def sample_frames(video_path: Path, n: int):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        return []
    idx = sorted({int(round(i)) for i in [k * (len(frames) - 1) / max(n - 1, 1) for k in range(n)]})
    return [frames[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--official-manifest", type=Path, default=Path("output/stage1_frames/manifest.csv"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(args.clips)
    rows = []
    for r in clips.itertuples(index=False):
        variant = r.variant.split("/")[0]
        d = args.out / r.clip_id / variant
        d.mkdir(parents=True, exist_ok=True)
        for i, bgr in enumerate(sample_frames(Path(r.path), args.frames)):
            p = d / f"{i:03d}.png"
            if not p.is_file():
                cv2.imwrite(str(p), bgr)
            rows.append({"source_id": f"{r.clip_id}/{variant}", "group": r.clip_id, "domain": f"synth_video_{r.kind}", "label": r.label, "frame_path": str(p)})
    n_synth = len(rows)
    if args.official_manifest.is_file():
        off = pd.read_csv(args.official_manifest)
        off = off[off.domain == "official_stage1"]
        rows += off.to_dict("records")
    with open(args.out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "group", "domain", "label", "frame_path"])
        w.writeheader()
        w.writerows(rows)
    print(f"synth frames {n_synth} + official {len(rows) - n_synth} -> {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
