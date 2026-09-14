"""작은 comma2k19 서브셋(세그먼트 1개 이상)에 대해 다음을 한 번에 만든다:

  1) timestamp 기준으로 동기화된 10Hz accel/steer 라벨 테이블 (CSV)
  2) 클래스 분포 리포트 (CSV)
  3) 라벨 방향(특히 steer_label 부호) 육안 검증용 오버레이 프레임 (JPEG)

실행 예 (repo 안에 실제로 받아둔 공식 데모 세그먼트 기준):
    python -m src.data.comma2k19.run_subset \
        --segment data/external/comma2k19_example/segment \
        --out output/comma2k19_subset

⚠️ 이 스크립트가 만든 라벨은 아직 model 학습에 쓰면 안 된다. overlay 프레임을 직접 보고
labeling.STEER_SIGN(좌/우 부호)이 맞는지 확인한 뒤 학습으로 넘어갈 것.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .labeling import DEFAULT_CONFIG_PATH, LabelConfig, accel_from_speed, accel_label, steer_label
from .report import class_distribution_report
from .sync import Comma2k19Segment, build_10hz_grid, interpolate, nearest_frame_indices


def build_label_table(
    segment_dir: Path, hz: float = 10.0, config: LabelConfig | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    config = config or LabelConfig.load()
    segment = Comma2k19Segment.load(segment_dir)
    grid = build_10hz_grid(segment, hz)
    speed, steer = interpolate(segment, grid)
    accel = accel_from_speed(grid, speed, config)
    frame_idx = nearest_frame_indices(segment, grid)

    df = pd.DataFrame(
        {
            "sample_index": np.arange(len(grid)),
            "time_seconds": grid - grid[0],
            "frame_index": frame_idx,
            "speed_mps": speed,
            "accel_mps2": accel,
            "steering_angle_deg": steer,
            "accel_label": accel_label(speed, accel, config),
            "steer_label": steer_label(steer, config),
        }
    )
    return df, frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="comma2k19 세그먼트 -> Stage3 라벨 변환 검증")
    parser.add_argument("--segment", type=Path, required=True, help="세그먼트 폴더 (video.hevc가 있는 위치)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="라벨 파라미터 yaml")
    parser.add_argument("--overlay-every", type=int, default=10, help="오버레이 이미지를 몇 샘플마다 저장할지")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    config = LabelConfig.load(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    df, frame_idx = build_label_table(args.segment, config=config)
    df.to_csv(args.out / "labels_10hz.csv", index=False, encoding="utf-8-sig")

    report = class_distribution_report(df)
    report.to_csv(args.out / "class_distribution.csv", index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))

    if not args.no_overlay:
        from .overlay import render_overlay_frames

        saved = render_overlay_frames(
            args.segment / "video.hevc", df, frame_idx, args.out / "overlay", every=args.overlay_every
        )
        print(f"\n오버레이 프레임 {len(saved)}장 저장: {args.out / 'overlay'}")

    print(f"\n라벨 테이블: {args.out / 'labels_10hz.csv'} ({len(df)}행)")
    print(
        "steer_sign은 gyro_z 교차검증으로 확인됨(EXPERIMENT_DESIGN.md §9). "
        "단 이 세그먼트엔 실제 회전이 없어 육안 검증은 아직 못 함 - 회전이 있는 "
        "segment를 구하면 overlay로 재확인할 것."
    )


if __name__ == "__main__":
    main()
