"""EXP-S3-DINO-001 — DINOv2 ViT-S/14 (timm `vit_small_patch14_dinov2.lvd142m`, img_size 224) 프레임 특징 캐시.

ResNet18 ImageNet 특징(512-d) 대신 자기지도 DINOv2 CLS 특징(384-d)을 appearance 입력으로 쓴다.
전처리는 ResNet 캐시와 동일(짧은 변 224 리사이즈 + 중앙 crop, ImageNet 정규화). 가중치는 output/dinov2_vits14_224_timm.pt
(img_size=224로 생성한 timm 모델의 state_dict, 86MB)로 오프라인 로드 — 추론 zip에 그대로 포함하고 timm을 requirements에 추가한다.

실행:
    python -m src.train.stage3_features_dino --video-paths output/comma2k19_multi_v5/video_paths.csv --out output/stage3_features_dino --only-prefix b0c9d2329ad1606b
    python -m src.train.stage3_features_dino --video-paths output/open_video_paths.csv --out output/stage3_features_dino_open
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

from .stage3_dataset import _decode_video_frames

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(weights: Path, device):
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-paths", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("output/dinov2_vits14_224_timm.pt"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--only-prefix", default="", help="segment_id 접두어 필터(예: RAV4 dongle)")
    ap.add_argument("--shard", default="0/1", help="i/n: 병렬 실행용 분할")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(args.weights, device)
    mean, std = _MEAN.to(device), _STD.to(device)
    vp = pd.read_csv(args.video_paths)
    if args.only_prefix:
        vp = vp[vp.segment_id.str.startswith(args.only_prefix)]
    si, sn = (int(x) for x in args.shard.split("/"))
    vp = vp.iloc[si::sn]
    done = 0
    with torch.inference_mode():
        for row in vp.itertuples(index=False):
            out_path = args.out / (row.segment_id.replace("/", "__") + ".npy")
            if out_path.is_file():
                done += 1
                continue
            frames = _decode_video_frames(Path(row.video_path), 224)
            feats = []
            for s in range(0, len(frames), args.batch):
                x = torch.from_numpy(frames[s : s + args.batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = (x - mean) / std
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    f = model(x)
                feats.append(f.float().cpu().numpy().astype(np.float16))
            np.save(out_path, np.concatenate(feats))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(vp)}", flush=True)
    print(f"완료: {done}/{len(vp)} -> {args.out}")


if __name__ == "__main__":
    main()
