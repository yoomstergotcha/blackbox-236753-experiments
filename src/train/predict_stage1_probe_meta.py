"""EXP-S1-PROBE-002 — Stage1 구조 프로브: 파일 메타데이터(해상도·fps)만으로 판정한다.

가설: 비공개 ORIGINAL은 공개 예시(CCD 규격 1280x720, 10fps)와 같고, 재촬영본은 촬영 기기 규격(1080p/세로/30fps 등)이 남아 있다
(공식 안내의 '해상도/aspect ratio 변화' 단서). 규칙: (w,h)!=(1280,720) 또는 round(fps)!=10 → RERECORDED, 그 외 ORIGINAL.
해석: LB S1 ≈0.405(전부 ORIGINAL과 동일)면 비공개는 두 클래스 모두 규격이 정규화됨 / 높으면 규격 차이가 라벨과 일치 /
0.405 미만이면 ORIGINAL 쪽 규격이 다양함. 파일 간 통계 없음(고정 상수 규칙).
"""
from pathlib import Path

import cv2
import pandas as pd

_P_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".3gp"}


def _p_meta(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def predict_stage1(data_dir, model_dir):
    video_dir = Path(data_dir) / "videos"
    if not video_dir.is_dir():
        video_dir = Path(data_dir)
    rows = []
    for path in sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in _P_VIDEO_EXT):
        meta = _p_meta(path)
        answer = "ORIGINAL"
        if meta is not None:
            w, h, fps = meta
            if (w, h) != (1280, 720) or round(fps) != 10:
                answer = "RERECORDED"
        rows.append({"ID": path.stem, "answer": answer})
    return pd.DataFrame(rows, columns=["ID", "answer"])
