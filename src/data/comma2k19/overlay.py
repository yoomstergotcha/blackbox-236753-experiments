"""라벨 방향(특히 steer_label의 좌/우 부호, labeling.STEER_SIGN)을 실제 영상과 대조해
눈으로 검증하기 위한 프레임 오버레이 생성기.

cv2가 이 환경에서 video.hevc(HEVC 코덱)를 디코딩하지 못하면(코덱 미포함 빌드),
opencv를 ffmpeg 지원 포함으로 다시 설치하거나 ffmpeg CLI로 프레임을 추출하는 방식으로
바꿔야 한다. 이 함수는 cv2.VideoCapture만 시도한다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def render_overlay_frames(
    video_path: Path,
    grid_df: pd.DataFrame,
    frame_indices: np.ndarray,
    out_dir: Path,
    every: int = 1,
) -> list[Path]:
    """grid_df의 각 행(sample_index, speed, steering_angle, accel_label, steer_label)을
    frame_indices[i]에 해당하는 실제 프레임 위에 텍스트로 찍어 저장한다.

    every=N이면 N개 샘플마다 하나씩만 저장한다(전체를 다 저장하면 이미지 수가 많아짐).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다(코덱 미지원 가능): {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    current = -1
    frame = None
    rows = grid_df.reset_index(drop=True)

    try:
        for i in range(0, len(rows), every):
            target = int(frame_indices[i])
            if target < current:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                current = target - 1
            while current < target:
                ok, frame = cap.read()
                if not ok:
                    break
                current += 1
            if frame is None:
                continue

            row = rows.iloc[i]
            canvas = frame.copy()
            lines = [
                f"sample_index={int(row.sample_index)} t={row.time_seconds:.2f}s frame={target}",
                f"speed={row.speed_mps:.2f}m/s accel={row.accel_mps2:+.2f} -> {row.accel_label}",
                f"steering_angle={row.steering_angle_deg:+.2f}deg -> {row.steer_label}",
            ]
            for j, line in enumerate(lines):
                cv2.putText(
                    canvas, line, (10, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    canvas, line, (10, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA
                )

            out_path = out_dir / f"sample_{int(row.sample_index):05d}.jpg"
            cv2.imwrite(str(out_path), canvas)
            saved.append(out_path)
    finally:
        cap.release()

    return saved
