"""Stage3 프레임별 ResNet18(ImageNet, frozen) 특징을 한 번만 추출해 저장한다.

EXP-S3-BASE-005의 Stage3ResNetHead는 backbone이 frozen이고 클립 특징 = 16프레임 특징의
평균(TSN)이므로, 프레임별 512-d 특징을 캐시해 두면 head 학습은 수학적으로 동일한 채로
초 단위가 된다. 291세그먼트 x 1200프레임 x 512 x float16 ≈ 715MB.

전처리는 학습/추론과 동일: 짧은 변 224 리사이즈 + 중앙 crop, ImageNet 정규화.
(Stage3ResNetHead는 MViT 정규화 입력을 받아 내부에서 ImageNet 정규화로 되돌리므로 결과 동일)

실행:
    python -m src.train.stage3_features --video-paths output/comma2k19_multi/video_paths.csv \
        --out output/stage3_features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .stage3_dataset import _decode_video_frames

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-paths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    mean, std = _IMAGENET_MEAN.to(device), _IMAGENET_STD.to(device)

    vp = pd.read_csv(args.video_paths)
    done = 0
    with torch.inference_mode():
        for row in vp.itertuples(index=False):
            out_path = args.out / (row.segment_id.replace("/", "__") + ".npy")
            if out_path.is_file():
                done += 1
                continue
            frames = _decode_video_frames(Path(row.video_path), 224)  # (N, 224, 224, 3) uint8
            feats = []
            for s in range(0, len(frames), args.batch):
                x = torch.from_numpy(frames[s : s + args.batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = (x - mean) / std
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    f = backbone(x)
                feats.append(f.float().cpu().numpy().astype(np.float16))
            np.save(out_path, np.concatenate(feats))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(vp)} segments")
    print(f"완료: {done}/{len(vp)} -> {args.out}")


if __name__ == "__main__":
    main()
