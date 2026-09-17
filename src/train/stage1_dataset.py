"""Stage1(재녹화 판별) 학습 데이터셋.

manifest(src.data.stage1.extract_frames 출력)의 ORIGINAL 프레임 하나가 두 개의 학습 샘플이
된다 — 그대로(ORIGINAL=0)와 synth_rerecord를 즉석 적용한 것(RERECORDED=1). 같은 소스가 양쪽
클래스에 모두 등장하므로 모델이 "장면 내용"이 아니라 "재촬영 흔적"으로만 구분하도록 강제된다.

크롭은 원본 해상도에서 224x224를 랜덤으로 잘라낸다(재압축 블록/모아레 같은 고주파 단서를
축소로 뭉개지 않기 위해). 평가 영상 해상도가 다를 수 있어 양쪽 클래스에 동일하게 랜덤
스케일(0.6~1.2배)을 먼저 적용한다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.stage1.synth_rerecord import apply_rerecord, sample_params

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP = 224


def to_tensor(rgb_crop: np.ndarray) -> torch.Tensor:
    x = rgb_crop.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def random_crop(rgb: np.ndarray, rng: np.random.Generator, size: int = CROP) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w) + 1e-3
        rgb = cv2.resize(rgb, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return rgb[y : y + size, x : x + size]


def grid_crops(rgb: np.ndarray, size: int = CROP, n: int = 5) -> list[np.ndarray]:
    """추론용 고정 크롭: 중앙 + 4개 사분면 중심 (원본 해상도 유지)."""
    h, w = rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w) + 1e-3
        rgb = cv2.resize(rgb, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
    centers = [(h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4)]
    out = []
    for cy, cx in centers[:n]:
        y = int(np.clip(cy - size // 2, 0, h - size))
        x = int(np.clip(cx - size // 2, 0, w - size))
        out.append(rgb[y : y + size, x : x + size])
    return out


class Stage1TrainDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, seed: int = 20260825, scale_range=(0.6, 1.2)):
        self.rows = manifest.reset_index(drop=True)
        self.seed = seed
        self.scale_range = scale_range
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows) * 2

    def __getitem__(self, index: int):
        row = self.rows.iloc[index // 2]
        label = index % 2  # 0=ORIGINAL, 1=RERECORDED (같은 프레임이 두 클래스로 모두 등장)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        bgr = cv2.imread(row["frame_path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        s = float(rng.uniform(*self.scale_range))
        if abs(s - 1.0) > 0.02:
            rgb = cv2.resize(rgb, (max(CROP, int(rgb.shape[1] * s)), max(CROP, int(rgb.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

        if label == 1:
            rgb = apply_rerecord(rgb, sample_params(rng), rng)

        crop = random_crop(rgb, rng)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        return to_tensor(np.ascontiguousarray(crop)), label


class Stage1EvalFrames:
    """검증용: 프레임 경로 목록 -> (source_id, crops tensor). RERECORDED 합성 여부는 호출자가 결정."""

    @staticmethod
    def crops_for_frame(rgb: np.ndarray) -> torch.Tensor:
        return torch.stack([to_tensor(np.ascontiguousarray(c)) for c in grid_crops(rgb)])
