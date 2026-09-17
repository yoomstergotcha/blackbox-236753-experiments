"""EXP-S3-BASE-006 — 캐시된 프레임 특징(stage3_features.py)으로 Stage3 head만 학습.

Stage3ResNetHead(frozen ResNet18 + MLP head, 클립 = 16프레임 특징 평균)와 수학적으로 동일하다.
프레임별 512-d 특징을 미리 뽑아 두었으므로 클립 특징은 이동 평균(cumsum)으로 즉시 계산되고,
epoch당 수 초면 끝나 데이터 5배(291세그먼트/42route)에도 즉시 실험할 수 있다.

평가는 공식 채점식(src.eval.metrics.macro_f1: accel 0.7 + steer 0.3, STOPPED 행 steer 제외)으로
매 epoch 계산, best는 val_stage3_score 기준. 저장 포맷은 기존 predict_stage3 snippet이 그대로
읽을 수 있도록 `_Stage3ResNetHead` state_dict(backbone 포함)로 내보낸다.

실행:
    python -m src.train.train_stage3_head --labels output/comma2k19_multi/labels_10hz.csv \
        --features output/stage3_features --out output/exp_s3_base_006 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.comma2k19.split import assert_no_route_leakage, group_train_val_split, route_id
from src.eval.metrics import macro_f1

from .stage3_dataset import ACCEL_TO_IDX, STEER_TO_IDX
from .stage3_model_light import Stage3ResNetHead

CLIP = 16


def clip_features_temporal(feats: np.ndarray, frame_idx: np.ndarray, n_frames: int = CLIP) -> np.ndarray:
    """EXP-S3-TEMP-001: 16프레임을 앞 8/뒤 8로 나눠 각각 평균한 뒤 concat(1024-d).
    가감속은 시간 순서 정보인데 전체 평균은 순서를 버린다 — 앞/뒤 반윈도 차이가 속도 변화의 단서."""
    n = len(feats)
    out = np.empty((len(frame_idx), feats.shape[1] * 2), dtype=np.float32)
    f32 = feats.astype(np.float32)
    h = n_frames // 2
    for k, i in enumerate(frame_idx):
        idx = np.clip(int(i) - h + np.arange(n_frames), 0, n - 1)
        out[k, : feats.shape[1]] = f32[idx[:h]].mean(0)
        out[k, feats.shape[1] :] = f32[idx[h:]].mean(0)
    return out


def clip_features(feats: np.ndarray, frame_idx: np.ndarray, n_frames: int = CLIP) -> np.ndarray:
    """feats: (N, 512). 각 frame_idx에 대해 [i-8, i+8) 구간(경계는 clip) 평균 — MultiSegmentStage3Dataset의
    클립 구성(np.clip(i-8+arange(16), 0, N-1))과 정확히 같은 프레임 집합의 평균."""
    n = len(feats)
    out = np.empty((len(frame_idx), feats.shape[1]), dtype=np.float32)
    f32 = feats.astype(np.float32)
    for k, i in enumerate(frame_idx):
        idx = np.clip(int(i) - n_frames // 2 + np.arange(n_frames), 0, n - 1)
        out[k] = f32[idx].mean(0)
    return out


def build_xy(labels: pd.DataFrame, feat_dir: Path, temporal: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ya, ys = [], [], []
    fn = clip_features_temporal if temporal else clip_features
    for sid, g in labels.groupby("segment_id", sort=False):
        f = np.load(feat_dir / (sid.replace("/", "__") + ".npy"))
        xs.append(fn(f, g["frame_index"].to_numpy()))
        ya.append(g["accel_label"].map(ACCEL_TO_IDX).to_numpy())
        ys.append(g["steer_label"].map(STEER_TO_IDX).to_numpy())
    return np.concatenate(xs), np.concatenate(ya), np.concatenate(ys)


def stage3_score(accel_true, accel_pred, steer_true, steer_pred) -> dict:
    accel_f1 = macro_f1(list(accel_true), list(accel_pred), list(range(4)))
    mask = accel_true != ACCEL_TO_IDX["STOPPED"]
    steer_f1 = macro_f1(list(steer_true[mask]), list(steer_pred[mask]), list(range(3))) if mask.any() else 0.0
    return {"accel_macro_f1": accel_f1, "steer_macro_f1": steer_f1, "stage3_score": 0.7 * accel_f1 + 0.3 * steer_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--class-weight", action="store_true", help="클래스 빈도 역수로 CE 가중 (EXP-S3-CLASS-001)")
    parser.add_argument("--temporal", action="store_true", help="앞/뒤 반윈도 평균 concat(1024-d) (EXP-S3-TEMP-001)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(args.labels)
    segs = sorted(labels["segment_id"].unique())
    train_segs, val_segs = group_train_val_split(segs, val_ratio=args.val_ratio, seed=args.seed)
    assert_no_route_leakage(train_segs, val_segs)
    tr, va = labels[labels["segment_id"].isin(train_segs)], labels[labels["segment_id"].isin(val_segs)]
    print(f"train {len(train_segs)} seg/{len({route_id(s) for s in train_segs})} route ({len(tr)} samples) | val {len(val_segs)} seg/{len({route_id(s) for s in val_segs})} route ({len(va)} samples)")

    t0 = time.time()
    xtr, atr, str_ = build_xy(tr, args.features, args.temporal)
    xva, ava, sva = build_xy(va, args.features, args.temporal)
    print(f"features built in {time.time() - t0:.0f}s: xtr {xtr.shape} xva {xva.shape}")
    print("val accel dist:", Counter(ava.tolist()), "val steer dist:", Counter(sva.tolist()))

    full = Stage3ResNetHead(pretrained=True)  # backbone 가중치는 export용, 학습은 head만
    if args.temporal:  # 입력 1024-d head로 교체 (추론 snippet도 temporal 버전을 써야 함)
        full.accel = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))
        full.steer = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))
    head_a, head_s = full.accel.to(device), full.steer.to(device)
    params = list(head_a.parameters()) + list(head_s.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    wa = ws = None
    if args.class_weight:
        ca = np.bincount(atr, minlength=4).astype(np.float32); cs = np.bincount(str_, minlength=3).astype(np.float32)
        wa = torch.tensor(ca.sum() / (4 * np.maximum(ca, 1)), device=device)
        ws = torch.tensor(cs.sum() / (3 * np.maximum(cs, 1)), device=device)
        print("class weights accel", wa.tolist(), "steer", ws.tolist())

    Xtr = torch.from_numpy(xtr).to(device); Atr = torch.from_numpy(atr).long().to(device); Str = torch.from_numpy(str_).long().to(device)
    Xva = torch.from_numpy(xva).to(device)
    history, best, best_epoch = [], -1.0, -1
    n = len(Xtr)
    for epoch in range(args.epochs):
        head_a.train(); head_s.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s : s + args.batch_size]
            la, ls = head_a(Xtr[idx]), head_s(Xtr[idx])
            loss = nn.functional.cross_entropy(la, Atr[idx], weight=wa) + nn.functional.cross_entropy(ls, Str[idx], weight=ws)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        head_a.eval(); head_s.eval()
        with torch.inference_mode():
            pa = head_a(Xva).argmax(1).cpu().numpy(); ps = head_s(Xva).argmax(1).cpu().numpy()
        ev = stage3_score(ava, pa, sva, ps)
        rec = {"epoch": epoch, "train_loss": tot / n, **ev, "val_accel_pred_dist": dict(Counter(pa.tolist())), "val_steer_pred_dist": dict(Counter(ps.tolist()))}
        history.append(rec)
        print(rec)
        if ev["stage3_score"] > best:
            best, best_epoch = ev["stage3_score"], epoch
            full.accel.load_state_dict(head_a.state_dict()); full.steer.load_state_dict(head_s.state_dict())
            torch.save(full.state_dict(), args.out / "best.pt")

    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_stage3_score": best, "train_segments": len(train_segs), "val_segments": len(val_segs), "temporal": args.temporal, "class_weight": args.class_weight}, f, ensure_ascii=False, indent=2)
    print(f"best val_stage3_score={best:.4f} (epoch {best_epoch}) -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
