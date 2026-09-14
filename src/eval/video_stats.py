"""Stage 2 채점에 필요한 영상별 fps·최대 프레임 번호 수집.

metrics.py의 순수 채점 로직은 cv2 없이도 동작해야 하므로, cv2를 쓰는 코드는
이 모듈에만 두고 지연 import한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


def build_fps_and_frame_count(
    video_paths: Mapping[str, str | Path],
) -> tuple[dict[str, float], dict[str, int]]:
    """{video_id: path} -> ({video_id: fps}, {video_id: last_frame_index})."""
    import cv2

    fps_map: dict[str, float] = {}
    max_frame_map: dict[str, int] = {}
    for video_id, path in video_paths.items():
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        fps_map[video_id] = float(fps) if fps and fps > 0 else 0.0
        max_frame_map[video_id] = max(0, frame_count - 1)
    return fps_map, max_frame_map
