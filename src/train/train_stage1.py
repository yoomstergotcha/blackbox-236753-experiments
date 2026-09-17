"""EXP-S1-SYNTH-001 — 합성 재녹화 데이터로 Stage1(ORIGINAL/RERECORDED) 분류기 학습.

모델: ImageNet-pretrained ResNet18 전체 fine-tune, fc -> 1 logit(BCE). 재압축/모아레 같은
텍스처 단서는 저수준 필터에 있어 backbone까지 학습시킨다(Stage3처럼 frozen이 아님).

Split(그룹 단위, 같은 소스는 한쪽에만):
  - 검증(official): data/stage1 공식 5쌍(ORIGINAL 실제 + RERECORDED 실제/DACON식) — "DACON식" 검증
  - 검증(synthetic): comma2k19 route 3개 hold-out — ORIGINAL 실제 + RERECORDED 합성
  - 학습: 나머지 comma2k19 route + OPEN 5개
평가는 영상 단위(프레임별 5크롭 확률 평균 -> 영상 확률 평균 -> 0.5 임계) Macro-F1 —
대회 채점 단위와 동일. best checkpoint는 두 검증 Macro-F1의 평균으로 고른다.

실행:
    python -m src.train.train_stage1 --manifest output/stage1_frames/manifest.csv --out output/exp_s1_synth_001
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from src.data.stage1.synth_rerecord import apply_rerecord, sample_params
from src.eval.metrics import macro_f1

from .stage1_dataset import Stage1EvalFrames, Stage1TrainDataset

LABELS = ["ORIGINAL", "RERECORDED"]


def build_model(pretrained: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(512, 1)
    return m


@torch.inference_mode()
def video_probs(model, device, frames_by_source: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """소스별 프레임 목록 -> 영상 RERECORDED 확률(프레임 5크롭 평균 -> 프레임 평균)."""
    model.eval()
    out = {}
    for sid, frames in frames_by_source.items():
        probs = []
        for rgb in frames:
            crops = Stage1EvalFrames.crops_for_frame(rgb).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(crops).float().squeeze(1)
            probs.append(torch.sigmoid(logit).mean().item())
        out[sid] = float(np.mean(probs))
    return out


def load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def evaluate(model, device, val_official: pd.DataFrame, val_synth: pd.DataFrame, seed: int) -> dict:
    # official: 실제 ORIGINAL + 실제(DACON식) RERECORDED
    off_frames, off_true = {}, {}
    for sid, g in val_official.groupby("source_id"):
        off_frames[sid] = [load_rgb(p) for p in g["frame_path"]]
        off_true[sid] = g["label"].iloc[0]
    off_prob = video_probs(model, device, off_frames)
    off_pred = {k: ("RERECORDED" if v >= 0.5 else "ORIGINAL") for k, v in off_prob.items()}
    ids = sorted(off_true)
    off_f1 = macro_f1([off_true[i] for i in ids], [off_pred[i] for i in ids], LABELS)

    # synthetic hold-out: 같은 소스의 ORIGINAL(실제) + RERECORDED(합성, 영상 단위 파라미터 고정)
    syn_frames, syn_true = {}, {}
    rng = np.random.default_rng([seed, 999])
    for sid, g in val_synth.groupby("source_id"):
        frames = [load_rgb(p) for p in g["frame_path"]]
        syn_frames[f"{sid}#orig"] = frames
        syn_true[f"{sid}#orig"] = "ORIGINAL"
        p = sample_params(rng)
        syn_frames[f"{sid}#rerec"] = [apply_rerecord(f, p, rng) for f in frames]
        syn_true[f"{sid}#rerec"] = "RERECORDED"
    syn_prob = video_probs(model, device, syn_frames)
    syn_pred = {k: ("RERECORDED" if v >= 0.5 else "ORIGINAL") for k, v in syn_prob.items()}
    ids = sorted(syn_true)
    syn_f1 = macro_f1([syn_true[i] for i in ids], [syn_pred[i] for i in ids], LABELS)

    return {
        "official_macro_f1": off_f1,
        "official_probs": {k: round(v, 3) for k, v in sorted(off_prob.items())},
        "synth_macro_f1": syn_f1,
        "synth_n_videos": len(syn_true),
        "score": 0.5 * (off_f1 + syn_f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--val-routes", type=int, default=3, help="comma2k19 route 몇 개를 synthetic 검증으로 뺄지")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = pd.read_csv(args.manifest)
    val_official = m[m["domain"] == "official_stage1"]
    comma = m[m["domain"] == "comma2k19"]
    routes = sorted(comma["group"].unique())
    rng = np.random.default_rng(args.seed)
    val_routes = set(rng.choice(routes, size=args.val_routes, replace=False).tolist())
    val_synth = comma[comma["group"].isin(val_routes)]
    train = m[(m["domain"] != "official_stage1") & (~m["group"].isin(val_routes))]
    assert set(train["group"]).isdisjoint(set(val_synth["group"])), "group leakage"
    print(f"train frames={len(train)} (x2 classes) | val_synth frames={len(val_synth)} routes={sorted(val_routes)} | val_official frames={len(val_official)}")

    ds = Stage1TrainDataset(train, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best = [], -1.0
    for epoch in range(args.epochs):
        ds.set_epoch(epoch)
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(x).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
            if (step + 1) % 20 == 0:
                print(f"  epoch {epoch} step {step + 1}/{len(loader)} loss={loss.item():.3f} {time.time() - t0:.0f}s")
        ev = evaluate(model, device, val_official, val_synth, args.seed)
        rec = {"epoch": epoch, "train_loss": total / max(n, 1), **ev, "elapsed_sec": time.time() - t0}
        history.append(rec)
        print(rec)
        if ev["score"] > best:
            best = ev["score"]
            torch.save(model.state_dict(), args.out / "best.pt")
            print(f"  -> best 갱신 (score={best:.4f}) 저장")

    if device.type == "cuda":
        history.append({"peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3})
    torch.save(model.state_dict(), args.out / "final.pt")
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"best score={best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
