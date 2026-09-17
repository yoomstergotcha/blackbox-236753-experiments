"""공식 Stage3 라벨(data/stage3/labels.csv, 공개 OPEN 5영상 50행)로 head 체크포인트를 채점한다.

캐시된 프레임 특징(output/stage3_features_open, 필요 시 output/stage3_motion_open)에서 head만 돌리므로 수 초.
공개 OPEN 영상은 20fps라 라벨 frame_index = 2*sample_index — 예측은 프레임 단위이므로 **frame_index로 merge**
(EVAL-FIX-STAGE3-OFFICIAL). --sim10은 짝수 프레임 + stride-2 motion으로 비공개 10fps를 시뮬레이션(frame_index//2).

실행:
    python -m src.eval.eval_stage3_official --ckpt output/exp_s3_class_001/best.pt
    python -m src.eval.eval_stage3_official --ckpt output/exp_s3_motion_001/best.pt --motion output/stage3_motion_open --sim10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.eval.metrics import score_stage3
from src.train.stage3_motion import motion_clip_features
from src.train.train_stage3_head import clip_features

ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]


def _head(state: dict, prefix: str, out: int) -> nn.Sequential:
    in_dim = state[f"{prefix}.0.weight"].shape[1]
    h = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, out))
    h.load_state_dict({k[len(prefix) + 1 :]: v for k, v in state.items() if k.startswith(prefix + ".")})
    return h.eval()


def predict_from_cache(ckpt: Path, feat_dir: Path, motion_dir: Path | None, ids: list[str], sim10: bool) -> pd.DataFrame:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    head_a, head_s = _head(state, "accel", 4), _head(state, "steer", 3)
    has_motion = "motion_mean" in state
    if has_motion and motion_dir is None:
        raise SystemExit("체크포인트가 motion 특징을 쓰는데 --motion 이 없습니다")
    in_dim = state["accel.0.weight"].shape[1]
    use_app = in_dim > (len(state["motion_mean"]) if has_motion else 0)
    rows = []
    for vid in ids:
        f = np.load(feat_dir / f"open__{vid}.npy")
        m = np.load(motion_dir / f"open__{vid}.npy") if has_motion else None
        if sim10:
            f = f[::2]
            m = m[::2, 1, :] if m is not None else None
        elif m is not None:
            m = m[:, 0, :]
        fi = np.arange(len(f))
        parts = []
        if use_app:
            parts.append(clip_features(f, fi))
        if has_motion:
            horizons = tuple(state["motion_horizons"].tolist()) if "motion_horizons" in state else ()
            z = (motion_clip_features(m, fi, horizons=horizons) - state["motion_mean"].numpy()) / state["motion_std"].numpy()
            parts.append(z.astype(np.float32))
        x = torch.from_numpy(np.concatenate(parts, 1))
        with torch.inference_mode():
            pa, ps = head_a(x).argmax(1).numpy(), head_s(x).argmax(1).numpy()
        rows.append(pd.DataFrame({"ID": vid, "frame": fi, "accel_label": [ACCEL[i] for i in pa], "steer_label": [STEER[i] for i in ps]}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=Path("output/stage3_features_open"))
    ap.add_argument("--motion", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=Path("data/stage3/labels.csv"))
    ap.add_argument("--sim10", action="store_true")
    args = ap.parse_args()
    gt = pd.read_csv(args.labels)
    ids = sorted(gt["ID"].unique())
    pred = predict_from_cache(args.ckpt, args.features, args.motion, ids, args.sim10)
    key = gt["frame_index"] // 2 if args.sim10 else gt["frame_index"]
    merged = gt[["ID", "sample_index"]].assign(frame=key).merge(pred, on=["ID", "frame"], how="left")
    assert merged["accel_label"].notna().all(), "라벨 프레임에 대응하는 예측이 없음"
    sub = merged[["ID", "sample_index", "accel_label", "steer_label"]]
    res = score_stage3(sub, gt[["ID", "sample_index", "accel_label", "steer_label"]])
    mode = "10fps-sim" if args.sim10 else "20fps-native"
    print(f"{args.ckpt} [{mode}] -> {res}")
    for col in ("accel_label", "steer_label"):
        print(f"  {col} pred dist:", sub[col].value_counts().to_dict(), "| gt:", gt[col].value_counts().to_dict())


if __name__ == "__main__":
    main()
