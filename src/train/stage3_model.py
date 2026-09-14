"""EXP-S3-BASE-001 첫 baseline 모델.

베이스라인 노트북(`[Baseline_Train]_3Stage_학습.ipynb`)의 `Stage3MViT` 구조를 그대로
재사용한다 — 첫 실험에서는 backbone을 바꾸지 않고 데이터/파이프라인 정합성만 확인한다
(`AUTONOMOUS_RESEARCH_AGENT.md` §7 STEP 4: 한 번에 한 축만 바꾼다).
"""
from __future__ import annotations

from torch import nn
from torchvision.models.video import mvit_v2_s


class Stage3MViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dim = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dim, 4)
        self.steer = nn.Linear(dim, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.accel(z), self.steer(z)
