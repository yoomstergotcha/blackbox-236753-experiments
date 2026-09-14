"""EXP-S3-BASE-001 — 첫 Stage 3 RGB temporal baseline (smoke-test).

⚠️ 이것은 성능을 노리는 실험이 아니라 **파이프라인 정합성 확인**이다. 현재 영상+CAN
라벨이 모두 있는 세그먼트는 `data/external/comma2k19_example/segment` 1개
(Example_1, 60초, STOPPED/LEFT 샘플 없음)뿐이다 — route가 1개뿐이라 route-level
split(EXPERIMENT_DESIGN.md §9-6)도 적용할 수 없어, 여기서는 순번 기반으로만 나눈다.
이 실행의 목적은 "학습 루프가 실제로 도는가, loss가 내려가는가, 체크포인트가
저장되는가"를 확인하는 것이다. 의미 있는 성능/일반화 평가는 여러 route의 실제 영상을
확보(별도 Human Review 필요)한 뒤 진행한다.

실행:
    python -m src.train.train_stage3_baseline \
        --labels output/comma2k19_subset/labels_10hz.csv \
        --video data/external/comma2k19_example/segment/video.hevc \
        --out output/exp_s3_base_001 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .stage3_dataset import Stage3ClipDataset
from .stage3_model import Stage3MViT


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-S3-BASE-001 smoke-test 학습")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dataset = Stage3ClipDataset(args.labels, args.video)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Stage3MViT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for clips, accel, steer in train_loader:
            clips, accel, steer = clips.to(device), accel.to(device), steer.to(device)
            accel_logits, steer_logits = model(clips)
            loss = nn.functional.cross_entropy(accel_logits, accel) + nn.functional.cross_entropy(
                steer_logits, steer
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(accel)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct_accel = correct_steer = 0
        with torch.inference_mode():
            for clips, accel, steer in val_loader:
                clips, accel, steer = clips.to(device), accel.to(device), steer.to(device)
                accel_logits, steer_logits = model(clips)
                loss = nn.functional.cross_entropy(accel_logits, accel) + nn.functional.cross_entropy(
                    steer_logits, steer
                )
                val_loss += loss.item() * len(accel)
                correct_accel += (accel_logits.argmax(1) == accel).sum().item()
                correct_steer += (steer_logits.argmax(1) == steer).sum().item()
        val_loss /= n_val

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accel_acc": correct_accel / n_val,
            "val_steer_acc": correct_steer / n_val,
            "elapsed_sec": time.time() - t0,
        }
        history.append(record)
        print(record)

    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM: {peak_vram_gb:.2f} GB")
        history.append({"peak_vram_gb": peak_vram_gb})

    torch.save(model.state_dict(), args.out / "smoke_test.pt")
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n체크포인트: {args.out / 'smoke_test.pt'}")
    print(f"학습 로그: {args.out / 'history.json'}")


if __name__ == "__main__":
    main()
