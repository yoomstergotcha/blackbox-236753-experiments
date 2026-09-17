"""Stage1(재녹화 판별) 학습 데이터셋.

manifest(src.data.stage1.extract_frames 출력)의 각 행이 학습 샘플이 된다:
  - label==ORIGINAL 행 -> (그대로, 0) 과 (synth_rerecord 즉석 합성, 1) 두 샘플.
    같은 소스가 양쪽 클래스에 모두 등장하므로 모델은 장면 내용이 아니라 재촬영 흔적만으로
    구분하도록 강제된다.
  - label==RERECORDED 행(공식 DACON식 실제 재녹화) -> (그대로, 1) 한 샘플.
    EXP-S1-SYNTH-001~004에서 합성만으로는 DACON 실제 흔적을 재현하지 못해 실제 쌍을
    학습에 포함한다(`official_repeat`로 오버샘플링).

크롭은 원본 해상도에서 224x224 랜덤(고주파 단서 보존). 평가 해상도가 다를 수 있어 양쪽
클래스에 동일하게 랜덤 스케일(0.6~1.2)을 먼저 적용한다. '블랙박스 품질' 기저 열화는
합성 소스(comma/OPEN)에는 항상, 이미 그 품질인 공식 프레임에는 30%만 적용한다.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.stage1.synth_rerecord import (
    apply_base,
    apply_capture,
    apply_rerecord,
    sample_base_params,
    sample_capture_params,
    sample_params,
)

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


def grid_crops(rgb: np.ndarray, size: int = CROP, n: int = 9) -> list[np.ndarray]:
    """추론용 고정 크롭: 중앙 + 4개 사분면 중심 + 4개 모서리 (원본 해상도 유지).
    모서리 크롭은 화면 재촬영의 베젤/배경 영역을 보기 위해 추가(v4)."""
    h, w = rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w) + 1e-3
        rgb = cv2.resize(rgb, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
    centers = [
        (h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4),
        (size // 2, size // 2), (size // 2, w - size // 2), (h - size // 2, size // 2), (h - size // 2, w - size // 2),
    ]
    out = []
    for cy, cx in centers[:n]:
        y = int(np.clip(cy - size // 2, 0, h - size))
        x = int(np.clip(cx - size // 2, 0, w - size))
        out.append(rgb[y : y + size, x : x + size])
    return out


class Stage1TrainDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, seed: int = 20260825, scale_range=(0.6, 1.2), official_repeat: int = 3, capture_ratio: float = 0.5):
        self.seed = seed
        self.scale_range = scale_range
        self.capture_ratio = capture_ratio
        self.epoch = 0
        items = []  # (frame_path, is_official, label, synth)
        for row in manifest.itertuples(index=False):
            official = row.domain == "official_stage1"
            rep = official_repeat if official else 1
            for _ in range(rep):
                if row.label == "ORIGINAL":
                    items.append((row.frame_path, official, 0, False))
                    items.append((row.frame_path, official, 1, True))
                else:
                    items.append((row.frame_path, official, 1, False))
        self.items = items

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        frame_path, official, label, synth = self.items[index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        rgb = cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        s = float(rng.uniform(*self.scale_range))
        if abs(s - 1.0) > 0.02:
            rgb = cv2.resize(rgb, (max(CROP, int(rgb.shape[1] * s)), max(CROP, int(rgb.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

        if (not official) or rng.random() < 0.3:
            rgb = apply_base(rgb, sample_base_params(rng))
        if synth:
            # 절반은 공개 예제식(미세: 블러+그레인), 절반은 실제 화면 재촬영식(v4, 강함)
            if rng.random() < self.capture_ratio:
                rgb = apply_capture(rgb, sample_capture_params(rng), rng)
            else:
                rgb = apply_rerecord(rgb, sample_params(rng), rng)

        crop = random_crop(rgb, rng)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        return to_tensor(np.ascontiguousarray(crop)), label


class Stage1EvalFrames:
    @staticmethod
    def crops_for_frame(rgb: np.ndarray) -> torch.Tensor:
        return torch.stack([to_tensor(np.ascontiguousarray(c)) for c in grid_crops(rgb)])
