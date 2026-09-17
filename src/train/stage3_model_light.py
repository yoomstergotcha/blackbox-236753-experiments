"""가벼운 Stage3 모델 — ImageNet-pretrained ResNet18(frozen) 평균 풀링 + 작은 학습 head.

`EXP-S3-BASE-002`/`EXP-S3-BASE-003`(MViT scratch, lr 1e-4/2e-5) 둘 다 majority-class
수준을 넘지 못한 뒤의 진단: train 4000 sample이 실제로는 47 segment/13 route에서 나온
것이라 유효 다양성이 작다. 사전학습 없는 대형 video transformer(mvit_v2_s, ~34M
파라미터)가 이 정도 데이터로 학습되길 기대하기 어렵다고 판단해, 공식 Stage2
baseline이 이미 쓰는 것과 같은 패턴(고정 ImageNet 특징 + 작은 학습 head)으로 모델
용량을 데이터 규모에 맞춘다 — 학습 대상 파라미터가 약 12만 개로 MViT 대비 1/300 이하.

주의: 여기서 목표는 대회 제출용 최종 모델이 아니라 "이 정도 데이터로도 majority-class
collapse를 벗어날 수 있는가"를 빠르게 확인하는 진단 실험이다.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

# Stage3ClipDataset/MultiSegmentStage3Dataset은 MViT 스타일(mean=0.45, std=0.225)로
# 이미 정규화된 텐서를 내놓는다 - 여기서 [0,1]로 되돌린 뒤 ImageNet 정규화를 다시 적용한다.
_MVIT_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
_MVIT_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Stage3ResNetHead(nn.Module):
    def __init__(self, pretrained: bool = True):
        """pretrained=False로 만들면 torchvision에서 내려받지 않는다 - 추론 시 저장된
        state_dict(backbone 포함)를 그대로 불러올 것이므로 인터넷 접근이 필요 없다
        (가이드 §15: 평가 서버는 인터넷이 안 됨)."""
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.accel = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))
        self.steer = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, t, h, w = clip.shape
        frames01 = clip * _MVIT_STD.to(clip.device) + _MVIT_MEAN.to(clip.device)
        frames = frames01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frames = (frames - _IMAGENET_MEAN.to(clip.device)) / _IMAGENET_STD.to(clip.device)

        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(frames)  # (b*t, 512)
        feats = feats.reshape(b, t, -1).mean(1)  # 시간축 평균 풀링(TSN 스타일) -> (b, 512)
        return self.accel(feats), self.steer(feats)


class Stage3ResNetMotionHead(nn.Module):
    """EXP-S3-MOTION-001: Stage3ResNetHead + optical-flow ego-motion 요약 특징(3K=39-d, 학습셋 평균/표준편차로
    표준화 - 버퍼로 저장해 추론 snippet이 동일하게 적용). head 입력 = 512 (appearance) + 39 (motion).
    use_appearance=False면 motion 39-d만 쓰는 ablation."""

    def __init__(self, motion_dim: int = 39, pretrained: bool = True, use_appearance: bool = True, horizons: tuple[int, ...] = ()):
        super().__init__()
        self.register_buffer("motion_horizons", torch.tensor(list(horizons), dtype=torch.long))  # 추론 시 같은 지평으로 특징 재구성
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.use_appearance = use_appearance
        in_dim = (512 if use_appearance else 0) + motion_dim
        self.register_buffer("motion_mean", torch.zeros(motion_dim))
        self.register_buffer("motion_std", torch.ones(motion_dim))
        self.accel = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))
        self.steer = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def head_input(self, app, motion):
        z = (motion - self.motion_mean) / self.motion_std
        return torch.cat([app, z], 1) if self.use_appearance else z

    def forward(self, clip, motion):
        app = None
        if self.use_appearance:
            b, c, t, h, w = clip.shape
            frames01 = clip * _MVIT_STD.to(clip.device) + _MVIT_MEAN.to(clip.device)
            frames = frames01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            frames = (frames - _IMAGENET_MEAN.to(clip.device)) / _IMAGENET_STD.to(clip.device)
            self.backbone.eval()
            with torch.no_grad():
                app = self.backbone(frames).reshape(b, t, -1).mean(1)
        x = self.head_input(app, motion)
        return self.accel(x), self.steer(x)
