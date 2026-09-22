"""EXP-S2-LEARN-001 — CCD 사고 프레임 라벨(onset)로 충돌 시점 localizer를 학습한다.

입력: output/stage2_ccd_feats/<vid>.npz (signals (N,6), motion (N,13), app (N,512)) — stage2_ccd_features.py
라벨: Crash-1500.txt onset(첫 사고 프레임). ego 관여 클립만 사용(비공개 Stage2는 직접 충돌만). 공개 5클립(000001~5)은 항상 holdout.
모델: 프레임 시퀀스 위 양방향 GRU → 프레임별 '충돌 시작' 점수. 목표 = onset 중심 가우시안(σ=1.5프레임) soft label, BCE.
추론 = argmax(점수) (+ 선택적으로 이른 burst 규칙과 비교). 평가 = ±3프레임 적중률, 유튜브 소스 단위 GroupKFold(같은 소스 클립이
train/val에 섞이지 않게).

비공개 클립이 더 길거나 fps가 다를 가능성 대비: 학습 시 시간축 증강(클립 앞뒤에 다른 클립 구간 이어붙이기, 2배 시간 스트레치)을 옵션으로 둔다.

실행:
    python -m src.train.train_stage2_collision --feats output/stage2_ccd_feats --ann data/external/ccd/Crash-1500.txt --out output/exp_s2_learn_001
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def load_ann(path: Path) -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"(\d+),\[(.*?)\],(\d+),([^,]+),([^,]+),([^,]+),(\w+)", line.strip())
        if not m:
            continue
        lab = np.array([int(x) for x in m.group(2).split(",")])
        rows.append({"vid": m.group(1), "onset": int(np.argmax(lab)) if lab.max() > 0 else -1, "src": int(m.group(4)), "ego": m.group(7) == "Yes"})
    return pd.DataFrame(rows)


def load_feats(feat_dir: Path, vid: str, use_app: bool) -> np.ndarray:
    z = np.load(feat_dir / f"{vid}.npz")
    parts = [z["signals"], z["motion"]]
    if use_app:
        parts.append(z["app"].astype(np.float32))
    return np.concatenate(parts, 1).astype(np.float32)


class Localizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 1)
        self.cfg = {"in_dim": in_dim, "hidden": hidden, "dropout": dropout}

    def forward(self, x):  # (B, N, D) -> (B, N)
        h, _ = self.gru(self.proj((x - self.in_mean) / self.in_std))
        return self.head(h).squeeze(-1)


def soft_target(n: int, onset: int, sigma: float = 1.5) -> np.ndarray:
    t = np.arange(n)
    return np.exp(-0.5 * ((t - onset) / sigma) ** 2).astype(np.float32)


def augment(x: np.ndarray, onset: int, pool: list[np.ndarray], rng: np.random.Generator):
    """시간축 증강: 앞/뒤에 다른 클립 구간(사고 없는 앞부분 25프레임)을 붙이거나 2배 스트레치(프레임 반복)."""
    r = rng.random()
    if r < 0.35:
        other = pool[int(rng.integers(len(pool)))][:25]
        x, onset = np.concatenate([other, x]), onset + len(other)
    elif r < 0.6:
        other = pool[int(rng.integers(len(pool)))][:25]
        x = np.concatenate([x, other])
    elif r < 0.75:
        x, onset = np.repeat(x, 2, axis=0), onset * 2
    return x, onset


def collate(items, device):
    n = max(len(x) for x, _ in items)
    X = torch.zeros(len(items), n, items[0][0].shape[1])
    Y = torch.zeros(len(items), n)
    M = torch.zeros(len(items), n)
    for i, (x, y) in enumerate(items):
        X[i, : len(x)] = torch.from_numpy(x)
        Y[i, : len(x)] = torch.from_numpy(y)
        M[i, : len(x)] = 1
    return X.to(device), Y.to(device), M.to(device)


@torch.no_grad()
def hit_rate(model, seqs, device, tol: int = 3):
    model.eval()
    hits, preds = [], []
    for x, onset in seqs:
        s = model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
        p = int(np.argmax(s))
        preds.append(p)
        hits.append(abs(p - onset) <= tol)
    return float(np.mean(hits)), preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, default=Path("output/stage2_ccd_feats"))
    ap.add_argument("--ann", type=Path, default=Path("data/external/ccd/Crash-1500.txt"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--no-app", action="store_true", help="ResNet 특징 제외(모션·신호만)")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann = load_ann(args.ann)
    ann = ann[ann.ego & (ann.onset >= 0)]
    ann = ann[[(args.feats / f"{v}.npz").is_file() for v in ann.vid]]
    public = {"000001", "000002", "000003", "000004", "000005"}
    data = {v: (load_feats(args.feats, v, not args.no_app), int(o)) for v, o in zip(ann.vid, ann.onset)}
    train_pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
    srcs = train_pool.src.to_numpy()
    fold_of = {s: i % args.folds for i, s in enumerate(rng.permutation(sorted(set(srcs))))}
    train_pool["fold"] = [fold_of[s] for s in srcs]
    print(f"ego clips {len(train_pool)} | sources {len(set(srcs))} | in_dim {next(iter(data.values()))[0].shape[1]}")

    oof = {}
    fold_hits = []
    for k in range(args.folds):
        tr = train_pool[train_pool.fold != k]
        va = train_pool[train_pool.fold == k]
        tr_seqs = [data[v] for v in tr.vid]
        va_seqs = [data[v] for v in va.vid]
        allx = np.concatenate([x for x, _ in tr_seqs])
        model = Localizer(allx.shape[1], hidden=args.hidden).to(device)
        model.in_mean.copy_(torch.from_numpy(allx.mean(0)))
        model.in_std.copy_(torch.from_numpy(allx.std(0) + 1e-6))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        pool = [x for x, _ in tr_seqs]
        best, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(len(tr_seqs))
            for s in range(0, len(order), args.batch):
                items = []
                for i in order[s : s + args.batch]:
                    x, o = tr_seqs[i]
                    if not args.no_aug:
                        x, o = augment(x, o, pool, rng)
                    items.append((x, soft_target(len(x), o)))
                X, Y, M = collate(items, device)
                logits = model(X)
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, Y, reduction="none") * M).sum() / M.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            h, _ = hit_rate(model, va_seqs, device)
            if h > best:
                best, best_state = h, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best_state)
        h, preds = hit_rate(model, va_seqs, device)
        fold_hits.append(h)
        for v, p in zip(va.vid, preds):
            oof[v] = p
        torch.save({"model": best_state, "config": model.cfg}, args.out / f"fold{k}.pt")
        print(f"fold {k}: val hit(±3) {h:.3f} (n={len(va)})", flush=True)
    err = np.array([oof[v] - data[v][1] for v in train_pool.vid])
    print(f"OOF hit(±3) = {np.mean(np.abs(err) <= 3):.3f} | mean fold {np.mean(fold_hits):.3f} | err percentiles 5/25/50/75/95: {np.percentile(err, [5, 25, 50, 75, 95]).astype(int).tolist()}")
    pub = [(v, data[v]) for v in sorted(public) if v in data]
    if pub:
        h_pub, p_pub = hit_rate(model, [d for _, d in pub], device)
        print("public 5 (last fold model):", [(v, p, d[1]) for (v, d), p in zip(pub, p_pub)], f"hit {h_pub:.2f}")
    json.dump({"fold_hits": fold_hits, "oof_hit": float(np.mean(np.abs(err) <= 3)), "args": {k: str(v) for k, v in vars(args).items()}}, open(args.out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
