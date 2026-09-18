"""추출된 comma2k19 세그먼트 폴더 여러 개(각 `<route>/<segment번호>/video.hevc` 구조)에서
10Hz 라벨 테이블을 한꺼번에 만든다. `run_subset.py`의 단일 세그먼트 로직을 그대로
재사용한다 — 로직 중복 없음.

사용 예:
    python -m src.data.comma2k19.build_multi_segment_labels \
        --root data/external/comma2k19_chunk1_subset --out output/comma2k19_chunk1_subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .labeling import LabelConfig
from .run_subset import build_label_table


def find_segments(root: Path) -> list[Path]:
    """root 아래에서 video.hevc가 있는 폴더를 전부 찾는다 (=하나의 segment)."""
    return sorted(p.parent for p in root.rglob("video.hevc"))


def route_id(segment_dir: Path) -> str:
    return segment_dir.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="추출된 다중 세그먼트 -> 10Hz 라벨 테이블")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="라벨 임계값 yaml (기본 configs/stage3/comma2k19_label.yaml)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = LabelConfig.load(args.config)
    segments = find_segments(args.root)
    print(f"{len(segments)}개 segment 발견")

    tables = []
    video_paths = []
    failures = []
    for seg_dir in segments:
        segment_id = f"{route_id(seg_dir)}/{seg_dir.name}"
        try:
            table, _ = build_label_table(seg_dir, config=config)
            table.insert(0, "segment_id", segment_id)
            table.insert(1, "route_id", route_id(seg_dir))
            tables.append(table)
            video_paths.append({"segment_id": segment_id, "video_path": str(seg_dir / "video.hevc")})
        except Exception as exc:  # noqa: BLE001
            failures.append({"segment_id": segment_id, "error": repr(exc)})

    labels = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    labels.to_csv(args.out / "labels_10hz.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(video_paths).to_csv(args.out / "video_paths.csv", index=False, encoding="utf-8-sig")

    n_segments = labels["segment_id"].nunique() if not labels.empty else 0
    n_routes = labels["route_id"].nunique() if not labels.empty else 0
    print(f"세그먼트 {n_segments}개 (route {n_routes}개), 총 {len(labels)} sample -> {args.out / 'labels_10hz.csv'}")
    if failures:
        print("실패:")
        for f in failures:
            print(" ", f)


if __name__ == "__main__":
    main()
