"""EXP-S3-BASE-002 — 여러 route의 실제 영상으로 Stage 3 baseline을 학습한다.

`EXP-S3-BASE-001`(smoke-test, 단일 route)과의 차이는 **데이터 소스 하나만**이다 — 모델
구조(`Stage3MViT`)와 학습 루프는 그대로 재사용한다. route-level group split
(`src.data.comma2k19.split`)을 실제로 적용해 같은 route의 segment가 train/val에
동시에 들어가지 않도록 한다.

실행:
    python -m src.train.train_stage3_multiroute \
        --labels output/comma2k19_chunk1_subset/labels_10hz.csv \
        --video-paths output/comma2k19_chunk1_subset/video_paths.csv \
        --out output/exp_s3_base_002 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.comma2k19.split import assert_no_route_leakage, group_train_val_split, route_id
from src.eval.metrics import macro_f1

from .stage3_dataset import ACCEL_TO_IDX, STEER_TO_IDX, MultiSegmentStage3Dataset
from .stage3_model import Stage3MViT
from .stage3_model_light import Stage3ResNetHead

MODELS = {"mvit": Stage3MViT, "resnet": Stage3ResNetHead}


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-S3-BASE-002: 다중 route baseline 학습")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video-paths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="mvit")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-train-samples", type=int, default=None, help="route 비율은 유지하고 train 표본 수만 제한(빠른 확인용)")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50, help="N step마다 중간 진행 출력")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    labels = pd.read_csv(args.labels)
    video_paths = dict(pd.read_csv(args.video_paths)[["segment_id", "video_path"]].itertuples(index=False, name=None))

    segment_ids = sorted(labels["segment_id"].unique().tolist())
    train_segments, val_segments = group_train_val_split(segment_ids, val_ratio=args.val_ratio, seed=args.seed)
    assert_no_route_leakage(train_segments, val_segments)
    print(
        f"route split: train {len(train_segments)}segment/{len({route_id(s) for s in train_segments})}route, "
        f"val {len(val_segments)}segment/{len({route_id(s) for s in val_segments})}route"
    )

    train_table = labels[labels["segment_id"].isin(train_segments)]
    val_table = labels[labels["segment_id"].isin(val_segments)]
    if args.max_train_samples and len(train_table) > args.max_train_samples:
        train_table = train_table.sample(n=args.max_train_samples, random_state=args.seed)
    if args.max_val_samples and len(val_table) > args.max_val_samples:
        val_table = val_table.sample(n=args.max_val_samples, random_state=args.seed)
    print(f"train sample={len(train_table)}  val sample={len(val_table)}")
    print("val accel dist:", Counter(val_table["accel_label"]))
    print("val steer dist:", Counter(val_table["steer_label"]))

    train_set = MultiSegmentStage3Dataset(train_table, video_paths)
    val_set = MultiSegmentStage3Dataset(val_table, video_paths)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MODELS[args.model]().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} trainable_params={n_trainable:,} / total_params={n_total:,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    history = []
    best_score = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = len(train_loader)
        for step, (clips, accel, steer) in enumerate(train_loader):
            clips, accel, steer = clips.to(device), accel.to(device), steer.to(device)
            accel_logits, steer_logits = model(clips)
            loss = nn.functional.cross_entropy(accel_logits, accel) + nn.functional.cross_entropy(
                steer_logits, steer
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(accel)
            if args.log_every and (step + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (step + 1) / elapsed
                eta = (n_batches - step - 1) / rate if rate > 0 else float("nan")
                print(
                    f"  epoch {epoch} step {step + 1}/{n_batches} loss={loss.item():.3f} "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        accel_true, accel_pred, steer_true, steer_pred = [], [], [], []
        with torch.inference_mode():
            for clips, accel, steer in val_loader:
                clips, accel, steer = clips.to(device), accel.to(device), steer.to(device)
                accel_logits, steer_logits = model(clips)
                loss = nn.functional.cross_entropy(accel_logits, accel) + nn.functional.cross_entropy(
                    steer_logits, steer
                )
                val_loss += loss.item() * len(accel)
                accel_true += accel.cpu().tolist()
                accel_pred += accel_logits.argmax(1).cpu().tolist()
                steer_true += steer.cpu().tolist()
                steer_pred += steer_logits.argmax(1).cpu().tolist()
        val_loss /= len(val_set)

        accel_acc = sum(t == p for t, p in zip(accel_true, accel_pred)) / len(accel_true)
        steer_acc = sum(t == p for t, p in zip(steer_true, steer_pred)) / len(steer_true)

        # 실제 대회 채점 지표(macro-F1)로도 계산한다 - accuracy는 majority-class collapse를
        # 못 걸러내는 오도된 지표였다(STRAIGHT 100% 예측이 accuracy=0.658을 냄).
        # ACCEL_TO_IDX/STEER_TO_IDX 순서 = src.eval.metrics의 STAGE3_*_LABELS 순서와 동일하므로
        # 정수 인덱스를 label로 그대로 써도 된다.
        accel_f1 = macro_f1(accel_true, accel_pred, list(range(4)))
        stopped_idx = ACCEL_TO_IDX["STOPPED"]
        steer_mask = [t != stopped_idx for t in accel_true]
        steer_true_scored = [s for s, m in zip(steer_true, steer_mask) if m]
        steer_pred_scored = [s for s, m in zip(steer_pred, steer_mask) if m]
        steer_f1 = macro_f1(steer_true_scored, steer_pred_scored, list(range(3))) if steer_true_scored else 0.0
        stage3_score = 0.7 * accel_f1 + 0.3 * steer_f1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accel_acc": accel_acc,
            "val_steer_acc": steer_acc,
            "val_accel_macro_f1": accel_f1,
            "val_steer_macro_f1": steer_f1,
            "val_stage3_score": stage3_score,
            "val_accel_pred_dist": dict(Counter(accel_pred)),
            "val_steer_pred_dist": dict(Counter(steer_pred)),
            "elapsed_sec": time.time() - t0,
        }
        history.append(record)
        print(record)

        if stage3_score > best_score:
            best_score = stage3_score
            best_epoch = epoch
            torch.save(model.state_dict(), args.out / "best.pt")

    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM: {peak_vram_gb:.2f} GB")
        history.append({"peak_vram_gb": peak_vram_gb})

    torch.save(model.state_dict(), args.out / "final.pt")
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n최종 epoch 체크포인트: {args.out / 'final.pt'}")
    print(f"최고 val_stage3_score={best_score:.4f} (epoch {best_epoch}) 체크포인트: {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
