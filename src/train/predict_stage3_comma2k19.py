"""공식 inference.py에 붙여넣을 Stage3 predict 함수 (comma2k19로 학습한 모델용).

self-contained 코드 블록이다 - submit.zip에는 src/ 패키지가 들어가지 않고
inference.py 파일 하나만 들어가므로, 여기서 모델 클래스를 다시 정의한다
(src.train.stage3_model_light.Stage3ResNetHead와 동일한 정의).

`build_submission_v2.py`가 이 파일의 내용을 그대로 읽어 공식 Stage1/2 코드 뒤에
이어붙여 inference.py를 만든다.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

_ACCEL_LABELS = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
_STEER_LABELS = ["LEFT", "STRAIGHT", "RIGHT"]
_CLIP_SIZE = 224
_CLIP_FRAMES = 16
# 모든 정규화 상수는 broadcasting 축을 착각하지 않도록 view()로 shape을 명시한다.
# (B, 3, T, H, W) 클립 텐서에 바로 곱/뺄셈되는 것은 4-D(1,3,1,1,1이 아니라 3,1,1,1 -
# 맨 앞 배치축은 broadcasting이 알아서 1로 채워준다)로 충분하다.
_MVIT_MEAN5 = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
_MVIT_STD5 = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
# (N, 3, H, W) 프레임 텐서(시간축을 배치로 합친 뒤)에 곱/뺄셈되는 것은 3-D.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _Stage3ResNetHead(nn.Module):
    """comma2k19로 학습한 가벼운 Stage3 모델 - ImageNet-pretrained ResNet18(frozen 상태로
    학습됨) 시간축 평균 풀링 + 작은 MLP head. EXPERIMENT_DESIGN.md EXP-S3-BASE-004/005 참고."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)  # 가중치는 아래 state_dict 로딩으로 채움(인터넷 불필요)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.accel = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))
        self.steer = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, t, h, w = clip.shape
        frames01 = clip * _MVIT_STD5.to(clip.device) + _MVIT_MEAN5.to(clip.device)
        frames = frames01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frames = (frames - _IMAGENET_MEAN.to(clip.device)) / _IMAGENET_STD.to(clip.device)
        feats = self.backbone(frames)
        feats = feats.reshape(b, t, -1).mean(1)
        return self.accel(feats), self.steer(feats)


def _decode_comma_style_frames(path: Path, size: int = _CLIP_SIZE):
    """학습 때(src/train/stage3_dataset.py `_decode_video_frames`)와 동일한 전처리:
    짧은 변을 size로 리사이즈 후 중앙 crop. 베이스라인의 256->224 크롭과는 다르다 -
    학습 전처리와 반드시 일치시켜야 하므로 재사용하지 않고 그대로 복제했다."""
    cap = cv2.VideoCapture(str(path))
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
        raise ValueError(f"cannot decode video: {path.name}")
    return np.stack(out)  # (N, size, size, 3) uint8


def predict_stage3(data_dir, model_dir):
    device = _device()
    model = _Stage3ResNetHead()
    state = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()

    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    mean5 = _MVIT_MEAN5.to(device)
    std5 = _MVIT_STD5.to(device)
    with torch.inference_mode():
        for path in videos:
            frames = _decode_comma_style_frames(path)  # (N, 224, 224, 3) uint8
            total = len(frames)
            batch_size = 16
            for start in range(0, total, batch_size):
                sample_indices = list(range(start, min(start + batch_size, total)))
                clips = []
                for i in sample_indices:
                    idx = np.clip(i - _CLIP_FRAMES // 2 + np.arange(_CLIP_FRAMES), 0, total - 1)
                    clip = torch.from_numpy(frames[idx].copy()).permute(3, 0, 1, 2).float() / 255.0
                    clips.append(clip)
                clip_batch = torch.stack(clips).to(device, non_blocking=True)
                clip_batch = (clip_batch - mean5) / std5
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clip_batch)
                accel_idx = accel_logits.float().argmax(1).cpu().tolist()
                steer_idx = steer_logits.float().argmax(1).cpu().tolist()
                for sample_index, a, s in zip(sample_indices, accel_idx, steer_idx):
                    rows.append(
                        {
                            "ID": path.stem,
                            "sample_index": sample_index,
                            "accel_label": _ACCEL_LABELS[a],
                            "steer_label": _STEER_LABELS[s],
                        }
                    )
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
