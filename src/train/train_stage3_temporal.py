"""EXP-S3-TEMPORAL-001 — 프레임 특징 시퀀스 위 dilated 1D conv (긴 문맥) Stage3 모델.

지금까지의 head는 프레임마다 16프레임 평균 + 수작업 지평 log-ratio를 입력으로 받는 MLP였다(MOTION-004, LB 0.515).
가감속은 수 초 문맥이 필요한데(§17 사전 분석: 3.2s 지평에서 상관 최대) 수작업 집계 대신 dilated conv가 문맥을 학습하게 한다.
입력 = 프레임별 [ResNet18 512-d (표준화) + Farneback 13-d (표준화)] = 525-d 시퀀스, 수용 범위 ≈ 253프레임(20fps 12.6s / 10fps 25s).
--fps-aug: 짝수 프레임 + stride-2 motion 시퀀스(10fps 시뮬레이션)를 학습에 추가. class weight 없음(LB 교훈).

실행:
    python -m src.train.train_stage3_temporal --labels output/comma2k19_multi_v2/labels_10hz.csv \
        --features output/stage3_features --motion output/stage3_motion --out output/exp_s3_temporal_001 \
        --holdout-route b0c9d2329ad1606b_2018-07-27--06-03-57 --fps-aug
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
from src.eval.metrics import score_stage3

from .stage3_dataset import ACCEL_TO_IDX, STEER_TO_IDX
from .train_stage3_head import stage3_score

ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]


class TemporalNet(nn.Module):
    def __init__(self, in_dim: int = 525, hidden: int = 128, kernel: int = 5, dilations=(1, 2, 4, 8, 16, 32), dropout: float = 0.2):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, padding=d * (kernel - 1) // 2, dilation=d) for d in dilations])
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in dilations])
        self.drop = nn.Dropout(dropout)
        self.head_a = nn.Conv1d(hidden, 4, 1)
        self.head_s = nn.Conv1d(hidden, 3, 1)
        self.cfg = {"in_dim": in_dim, "hidden": hidden, "kernel": kernel, "dilations": list(dilations), "dropout": dropout}

    def forward(self, x: torch.Tensor):  # x: (B, N, in_dim) raw
        x = (x - self.in_mean) / self.in_std
        h = self.proj(x).transpose(1, 2)  # (B, H, N)
        for conv, norm in zip(self.blocks, self.norms):
            h = h + self.drop(torch.nn.functional.gelu(norm(conv(h))))
        return self.head_a(h).transpose(1, 2), self.head_s(h).transpose(1, 2)


def load_sequence(sid: str, g: pd.DataFrame, feat_dir: Path, motion_dir: Path, sim10: bool):
    """(x (N,525) float32, ya (N,) int64 with -1 unlabeled, ys (N,))"""
    name = sid.replace("/", "__") + ".npy"
    f = np.load(feat_dir / name).astype(np.float32)
    m = np.load(motion_dir / name)
    assert len(f) == len(m), sid
    fi = g["frame_index"].to_numpy()
    if sim10:
        f, m, fi = f[::2], m[::2, 1, :], fi // 2
    else:
        m = m[:, 0, :]
    x = np.concatenate([f, m.astype(np.float32)], 1)
    ya = np.full(len(x), -1, np.int64)
    ys = np.full(len(x), -1, np.int64)
    idx = np.clip(fi.astype(int), 0, len(x) - 1)
    ya[idx] = g["accel_label"].map(ACCEL_TO_IDX).to_numpy()
    ys[idx] = g["steer_label"].map(STEER_TO_IDX).to_numpy()
    return x, ya, ys


def collate(items, device):
    n = max(len(x) for x, _, _ in items)
    X = torch.zeros(len(items), n, items[0][0].shape[1])
    A = torch.full((len(items), n), -1, dtype=torch.long)
    S = torch.full((len(items), n), -1, dtype=torch.long)
    for i, (x, ya, ys) in enumerate(items):
        X[i, : len(x)] = torch.from_numpy(x)
        A[i, : len(x)] = torch.from_numpy(ya)
        S[i, : len(x)] = torch.from_numpy(ys)
    return X.to(device), A.to(device), S.to(device)


def smooth_logits(l: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return l
    pad = w // 2
    lp = np.pad(l, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.stack([np.convolve(lp[:, j], k, mode="valid") for j in range(l.shape[1])], 1)


@torch.no_grad()
def predict_sequences(model, seqs, device, smooth: int = 1):
    model.eval()
    PA, PS, A, S = [], [], [], []
    for x, ya, ys in seqs:
        la, ls = model(torch.from_numpy(x)[None].to(device))
        la, ls = smooth_logits(la[0].cpu().numpy(), smooth), smooth_logits(ls[0].cpu().numpy(), smooth)
        mask = ya >= 0
        PA.append(la.argmax(1)[mask]); PS.append(ls.argmax(1)[mask]); A.append(ya[mask]); S.append(ys[mask])
    return stage3_score(np.concatenate(A), np.concatenate(PA), np.concatenate(S), np.concatenate(PS))


@torch.no_grad()
def evaluate_open(model, device, feat_dir: Path, motion_dir: Path, labels_csv: Path, sim10: bool, smooth: int):
    gt = pd.read_csv(labels_csv)
    rows = []
    model.eval()
    for vid in sorted(gt.ID.unique()):
        f = np.load(feat_dir / f"open__{vid}.npy").astype(np.float32)
        m = np.load(motion_dir / f"open__{vid}.npy")
        f, m = (f[::2], m[::2, 1, :]) if sim10 else (f, m[:, 0, :])
        x = np.concatenate([f, m.astype(np.float32)], 1)
        la, ls = model(torch.from_numpy(x)[None].to(device))
        pa = smooth_logits(la[0].cpu().numpy(), smooth).argmax(1)
        ps = smooth_logits(ls[0].cpu().numpy(), smooth).argmax(1)
        key = gt[gt.ID == vid]
        fr = (key.frame_index // 2 if sim10 else key.frame_index).to_numpy().clip(0, len(x) - 1)
        rows.append(pd.DataFrame({"ID": vid, "sample_index": key.sample_index.to_numpy(), "accel_label": [ACCEL[i] for i in pa[fr]], "steer_label": [STEER[i] for i in ps[fr]]}))
    return score_stage3(pd.concat(rows), gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--fps-aug", action="store_true")
    ap.add_argument("--holdout-route", default="")
    ap.add_argument("--smooth", type=int, default=1, help="평가 시 logit 이동평균 폭(프레임)")
    ap.add_argument("--open-features", type=Path, default=Path("output/stage3_features_open"))
    ap.add_argument("--open-motion", type=Path, default=Path("output/stage3_motion_open"))
    ap.add_argument("--open-labels", type=Path, default=Path("data/stage3/labels.csv"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(args.labels)
    segs = sorted(labels.segment_id.unique())
    holdout = {r for r in args.holdout_route.split(",") if r}
    if holdout:
        segs = [s for s in segs if route_id(s) not in holdout]
    train_segs, val_segs = group_train_val_split(segs, val_ratio=args.val_ratio, seed=args.seed)
    assert_no_route_leakage(train_segs, val_segs)
    groups = {sid: g.sort_values("frame_index") for sid, g in labels.groupby("segment_id")}
    t0 = time.time()
    train_seqs = [load_sequence(s, groups[s], args.features, args.motion, False) for s in train_segs]
    if args.fps_aug:
        train_seqs += [load_sequence(s, groups[s], args.features, args.motion, True) for s in train_segs]
    val_native = [load_sequence(s, groups[s], args.features, args.motion, False) for s in val_segs]
    val_sim10 = [load_sequence(s, groups[s], args.features, args.motion, True) for s in val_segs]
    allx = np.concatenate([x for x, _, _ in train_seqs])
    mean, std = allx.mean(0), allx.std(0) + 1e-6
    print(f"train {len(train_segs)} seg ({len(train_seqs)} seqs) | val {len(val_segs)} seg | loaded in {time.time() - t0:.0f}s | in_dim {allx.shape[1]}")

    model = TemporalNet(in_dim=allx.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    model.in_mean.copy_(torch.from_numpy(mean)); model.in_std.copy_(torch.from_numpy(std))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    history, best, best_epoch = [], -1.0, -1
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train_seqs))
        tot, nb = 0.0, 0
        for s in range(0, len(order), args.batch):
            X, A, S = collate([train_seqs[i] for i in order[s : s + args.batch]], device)
            la, ls = model(X)
            loss = nn.functional.cross_entropy(la.reshape(-1, 4), A.reshape(-1), ignore_index=-1) + nn.functional.cross_entropy(ls.reshape(-1, 3), S.reshape(-1), ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        ev = predict_sequences(model, val_native, device, args.smooth)
        ev10 = predict_sequences(model, val_sim10, device, args.smooth)
        select = (ev["stage3_score"] + ev10["stage3_score"]) / 2
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1), **ev, **{"sim10_" + k: v for k, v in ev10.items()}, "select_score": select}
        history.append(rec)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        if select > best:
            best, best_epoch = select, epoch
            torch.save({"model": model.state_dict(), "config": model.cfg}, args.out / "best.pt")
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device)["model"])
    open_scores = {}
    for sim in (False, True):
        for sm in (1, 31):
            r = evaluate_open(model, device, args.open_features, args.open_motion, args.open_labels, sim, sm)
            open_scores[f"{'sim10' if sim else 'native'}_smooth{sm}"] = {k: v for k, v in r.items() if isinstance(v, float)}
    print("best epoch", best_epoch, "select", round(best, 4))
    print("official OPEN (holdout):", {k: round(v["stage3_score"], 4) for k, v in open_scores.items()})
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_select": best, "open": open_scores, "args": {k: str(v) for k, v in vars(args).items()}}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
