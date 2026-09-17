"""EXP-S1-SYNTH — Stage1(ORIGINAL/RERECORDED) 분류기 학습.

모델: ImageNet-pretrained ResNet18 전체 fine-tune, fc -> 1 logit(BCE). 재압축/그레인 같은
텍스처 단서는 저수준 필터에 있어 backbone까지 학습시킨다.

--official-mode
  val      : 공식 5쌍은 검증 전용, 합성으로만 학습 (EXP-S1-SYNTH-001~004)
  lopo     : 공식 5쌍을 학습에 포함하되 leave-one-pair-out 5-fold로 out-of-fold 확률을 얻어
             실제 DACON 처리에 대한 일반화(F1, AUC, 최적 임계값)를 추정 (EXP-S1-SYNTH-005)
  trainall : 합성 + 공식 5쌍 전부로 최종 모델 학습 (lopo 뒤에 자동 실행)
평가는 항상 영상 단위(프레임별 5크롭 확률 평균 -> 영상 평균) macro-F1 — 대회 채점 단위.

실행:
    python -m src.train.train_stage1 --manifest output/stage1_frames/manifest.csv \
        --out output/exp_s1_synth_005 --official-mode lopo --epochs 4
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

from src.data.stage1.synth_rerecord import apply_base, apply_rerecord, sample_base_params, sample_params
from src.eval.metrics import macro_f1

from .stage1_dataset import Stage1EvalFrames, Stage1TrainDataset

LABELS = ["ORIGINAL", "RERECORDED"]


def build_model(pretrained: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(512, 1)
    return m


def load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def video_probs(model, device, frames_by_source: dict[str, list[np.ndarray]]) -> dict[str, float]:
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


def official_video_probs(model, device, official_df: pd.DataFrame, max_frames: int = 10) -> tuple[dict, dict]:
    frames, true = {}, {}
    for sid, g in official_df.groupby("source_id"):
        paths = g["frame_path"].tolist()
        idx = np.linspace(0, len(paths) - 1, num=min(max_frames, len(paths))).round().astype(int)
        frames[sid] = [load_rgb(paths[i]) for i in idx]
        true[sid] = g["label"].iloc[0]
    return video_probs(model, device, frames), true


def summarize(prob: dict, true: dict) -> dict:
    ids = sorted(true)
    pred = ["RERECORDED" if prob[i] >= 0.5 else "ORIGINAL" for i in ids]
    f1 = macro_f1([true[i] for i in ids], pred, LABELS)
    pos = [prob[i] for i in ids if true[i] == "RERECORDED"]
    neg = [prob[i] for i in ids if true[i] == "ORIGINAL"]
    auc = float(np.mean([[1.0 if p > n else 0.5 if p == n else 0.0 for n in neg] for p in pos])) if pos and neg else float("nan")
    cands = sorted(set(prob.values()))
    mids = [0.5] + [(a + b) / 2 for a, b in zip(cands[:-1], cands[1:])]
    best_f1, best_thr = max((macro_f1([true[i] for i in ids], ["RERECORDED" if prob[i] >= t else "ORIGINAL" for i in ids], LABELS), t) for t in mids)
    return {"macro_f1": f1, "auc": auc, "best_thr_f1": best_f1, "best_thr": best_thr, "probs": {k: round(v, 3) for k, v in sorted(prob.items())}}


def synth_holdout_probs(model, device, val_synth: pd.DataFrame, seed: int) -> tuple[dict, dict]:
    frames, true = {}, {}
    rng = np.random.default_rng([seed, 999])
    for sid, g in val_synth.groupby("source_id"):
        base = sample_base_params(rng)
        fr = [apply_base(load_rgb(p), base) for p in g["frame_path"]]
        frames[f"{sid}#orig"], true[f"{sid}#orig"] = fr, "ORIGINAL"
        p = sample_params(rng)
        frames[f"{sid}#rerec"], true[f"{sid}#rerec"] = [apply_rerecord(f, p, rng) for f in fr], "RERECORDED"
    return video_probs(model, device, frames), true


def train_model(train_df: pd.DataFrame, args, device, tag: str, eval_fn=None) -> tuple[nn.Module, list]:
    ds = Stage1TrainDataset(train_df, seed=args.seed, official_repeat=args.official_repeat)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    print(f"[{tag}] train items={len(ds)} steps/epoch={len(loader)}")
    for epoch in range(args.epochs):
        ds.set_epoch(epoch)
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in loader:
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
        rec = {"epoch": epoch, "train_loss": total / max(n, 1), "elapsed_sec": time.time() - t0}
        if eval_fn is not None:
            rec.update(eval_fn(model))
        history.append(rec)
        print(f"[{tag}]", rec)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--official-mode", choices=["val", "lopo", "trainall"], default="val")
    parser.add_argument("--official-repeat", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--val-routes", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    m = pd.read_csv(args.manifest)
    official = m[m["domain"] == "official_stage1"]
    comma = m[m["domain"] == "comma2k19"]
    rng = np.random.default_rng(args.seed)
    val_routes = set(rng.choice(sorted(comma["group"].unique()), size=args.val_routes, replace=False).tolist())
    val_synth = comma[comma["group"].isin(val_routes)]
    synth_train = m[(m["domain"] != "official_stage1") & (~m["group"].isin(val_routes))]
    assert set(synth_train["group"]).isdisjoint(set(val_synth["group"]))
    print(f"synth train frames={len(synth_train)} | val_synth frames={len(val_synth)} routes={sorted(val_routes)} | official frames={len(official)} pairs={official['group'].nunique()}")

    result = {"mode": args.official_mode}

    if args.official_mode == "val":
        def ev(model):
            prob, true = official_video_probs(model, device, official)
            s = summarize(prob, true)
            sp, st = synth_holdout_probs(model, device, val_synth, args.seed)
            s2 = summarize(sp, st)
            return {"official": s, "synth_macro_f1": s2["macro_f1"], "score": 0.4 * s["auc"] + 0.3 * s["macro_f1"] + 0.3 * s2["macro_f1"]}
        model, history = train_model(synth_train, args, device, "val", ev)
        best_i = int(np.argmax([h["score"] for h in history]))
        torch.save(model.state_dict(), args.out / "final.pt")
        result.update({"history": history, "best_epoch": best_i})

    else:
        pairs = sorted(official["group"].unique())
        oof_prob, oof_true = {}, {}
        fold_hist = []
        if args.official_mode == "lopo":
            for k, held in enumerate(pairs):
                tr = pd.concat([synth_train, official[official["group"] != held]])
                va = official[official["group"] == held]
                model, hist = train_model(tr, args, device, f"fold{k}:{held}")
                prob, true = official_video_probs(model, device, va)
                oof_prob.update(prob)
                oof_true.update(true)
                print(f"[fold{k}] held-out {held}: {prob}")
                fold_hist.append(hist)
                del model
                torch.cuda.empty_cache()
            oof = summarize(oof_prob, oof_true)
            print("=== LOPO out-of-fold (10 videos) ===", json.dumps(oof, ensure_ascii=False))
            result.update({"lopo": oof, "fold_history": fold_hist})

        # 최종: 합성 + 공식 5쌍 전부
        def ev(model):
            sp, st = synth_holdout_probs(model, device, val_synth, args.seed)
            return {"synth_macro_f1": summarize(sp, st)["macro_f1"]}
        model, hist = train_model(pd.concat([synth_train, official]), args, device, "trainall", ev)
        torch.save(model.state_dict(), args.out / "best.pt")
        threshold = 0.5
        if "lopo" in result:
            oof = result["lopo"]
            if oof["auc"] >= 0.75 and oof["best_thr_f1"] > oof["macro_f1"]:
                threshold = float(oof["best_thr"])
        with open(args.out / "threshold.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold, "source": "lopo out-of-fold best threshold" if threshold != 0.5 else "default"}, f, indent=2)
        result.update({"trainall_history": hist, "threshold": threshold})
        print(f"final best.pt 저장, threshold={threshold:.4f}")

    if device.type == "cuda":
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
