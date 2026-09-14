"""Stage 3 RGB clip dataset — 라벨 CSV(`src.data.comma2k19.run_subset` 출력) + video.hevc에서
16-frame 클립을 만든다.

첫 baseline(`EXP-S3-BASE-001`) 전용. 베이스라인 노트북
(`[Baseline_Train]_3Stage_학습.ipynb`)의 Stage3MViT 클립 구성 방식(16프레임, 224x224,
mean/std 0.45/0.225)을 그대로 따른다 — 이미 검증된 전처리를 그대로 재사용해 "한 번에
한 축만 바꾼다" 원칙을 지킨다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ACCEL_TO_IDX = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
STEER_TO_IDX = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}

_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]


class Stage3ClipDataset(Dataset):
    """단일 영상(`video_path`)과 그 영상에 대한 라벨 CSV로부터 클립을 만든다.

    전체 프레임을 `__init__`에서 한 번만 디코딩·리사이즈해 캐시한다 — 세그먼트 1개(<=1200
    프레임) 규모에서는 메모리에 다 올려도 무리가 없고, __getitem__마다 video.hevc를 다시
    여는 것보다 훨씬 빠르다.
    """

    def __init__(self, labels_csv: Path, video_path: Path, n_frames: int = 16, size: int = 224):
        self.table = pd.read_csv(labels_csv)
        self.video_path = Path(video_path)
        self.n_frames = n_frames
        self.size = size
        self.frames = self._decode_all_frames()

    def _decode_all_frames(self) -> np.ndarray:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {self.video_path}")
        out = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = self.size / min(h, w)
            nh, nw = max(self.size, round(h * scale)), max(self.size, round(w * scale))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            y, x = (nh - self.size) // 2, (nw - self.size) // 2
            out.append(rgb[y : y + self.size, x : x + self.size])
        cap.release()
        if not out:
            raise RuntimeError(f"디코딩된 프레임이 없습니다: {self.video_path}")
        return np.stack(out)  # (total_frames, size, size, 3) uint8

    def __len__(self) -> int:
        return len(self.table)

    def _clip(self, center_frame: int) -> torch.Tensor:
        total = len(self.frames)
        idx = np.clip(center_frame - self.n_frames // 2 + np.arange(self.n_frames), 0, total - 1)
        clip = torch.from_numpy(self.frames[idx].copy()).permute(3, 0, 1, 2).float() / 255.0
        return (clip - _MEAN) / _STD

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        clip = self._clip(int(row["frame_index"]))
        accel = ACCEL_TO_IDX[row["accel_label"]]
        steer = STEER_TO_IDX[row["steer_label"]]
        return clip, accel, steer


def _decode_video_frames(video_path: Path, size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        y, x = (nh - size) // 2, (nw - size) // 2
        out.append(rgb[y : y + size, x : x + size])
    cap.release()
    if not out:
        raise RuntimeError(f"디코딩된 프레임이 없습니다: {video_path}")
    return np.stack(out)


class MultiSegmentStage3Dataset(Dataset):
    """여러 세그먼트(각자 다른 video.hevc)를 하나의 학습셋으로 합친다.

    `label_table`은 `segment_id`, `frame_index`, `accel_label`, `steer_label` 컬럼을
    가져야 한다(`build_multi_segment_labels.py` 출력과 동일한 스키마). `video_paths`는
    {segment_id: video.hevc 경로}. 각 세그먼트의 프레임은 처음 접근할 때 한 번만
    디코딩해서 캐시한다(전부 미리 올리지 않음 — 세그먼트 수가 많아져도 안전).
    """

    def __init__(self, label_table: pd.DataFrame, video_paths: dict[str, Path], n_frames: int = 16, size: int = 224):
        self.table = label_table.reset_index(drop=True)
        self.video_paths = {k: Path(v) for k, v in video_paths.items()}
        self.n_frames = n_frames
        self.size = size
        self._frame_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.table)

    def _frames_for(self, segment_id: str) -> np.ndarray:
        if segment_id not in self._frame_cache:
            self._frame_cache[segment_id] = _decode_video_frames(self.video_paths[segment_id], self.size)
        return self._frame_cache[segment_id]

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        frames = self._frames_for(row["segment_id"])
        total = len(frames)
        idx = np.clip(int(row["frame_index"]) - self.n_frames // 2 + np.arange(self.n_frames), 0, total - 1)
        clip = torch.from_numpy(frames[idx].copy()).permute(3, 0, 1, 2).float() / 255.0
        clip = (clip - _MEAN) / _STD
        accel = ACCEL_TO_IDX[row["accel_label"]]
        steer = STEER_TO_IDX[row["steer_label"]]
        return clip, accel, steer
