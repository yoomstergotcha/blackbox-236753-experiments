"""EXP-S2-INV-002 — 위치 사전정보가 생길 수 없는 국소 시간 CNN 충돌 localizer (길이 불변).

BiGRU(EXP-S2-LEARN-001)는 클립 끝까지 보므로 CCD의 'onset은 후반부' 위치 사전정보를 학습한다 → 길이가 다른 비공개에서 무의미.
여기서는 수용영역이 ±12프레임인 dilated 1D CNN으로 프레임별 점수를 내고(패딩 zero), 학습 시 앞뒤에 다른 클립 구간을
무작위로 이어붙여(길이 20~200) 절대 위치가 라벨과 무관하게 만든다. 입력은 19-d 물리 신호(전역 이동·수직 충격·흐름 통계)만.
추론은 predict_stage2_invariant.py: stride 1~6 점수를 z-정규화해 합산(다중 스케일 융합) → argmax.

실행: python -m src.train.train_stage2_local --out output/exp_s2_local_001 [--seed N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.train.train_stage2_collision import collate, load_ann, load_feats, soft_target


class LocalCNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2, dilations=(1, 2, 3)):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        layers = [nn.Conv1d(in_dim, hidden, 1)]
        for d in dilations:
            layers += [nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, 1)
        self.cfg = {"in_dim": in_dim, "hidden": hidden, "dropout": dropout, "dilations": list(dilations)}

    def forward(self, x):  # (B, N, D) -> (B, N)
        h = self.body(((x - self.in_mean) / self.in_std).transpose(1, 2))
        return self.head(nn.functional.gelu(h)).squeeze(1)


def _segment(pre_pool: list[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    """다른 클립들의 사고 전 정상 주행 구간을 이어 붙여 length 프레임을 만든다."""
    if length <= 0:
        return pre_pool[0][:0]
    parts = []
    total = 0
    while total < length:
        seg = pre_pool[int(rng.integers(len(pre_pool)))]
        parts.append(seg)
        total += len(seg)
    return np.concatenate(parts)[:length]


def augment_pos(x: np.ndarray, onset: int, pre_pool: list[np.ndarray], rng: np.random.Generator):
    """길이·위치 무작위화: 앞/뒤에 정상 주행 구간(0~100프레임)을 붙이고, 확률적으로 2배 스트레치. 충돌은 하나뿐."""
    if rng.random() < 0.8:
        seg = _segment(pre_pool, int(rng.integers(0, 101)), rng)
        x, onset = np.concatenate([seg, x]), onset + len(seg)
    if rng.random() < 0.8:
        x = np.concatenate([x, _segment(pre_pool, int(rng.integers(0, 101)), rng)])
    if rng.random() < 0.15:
        x, onset = np.repeat(x, 2, axis=0), onset * 2
    return x, onset


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
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--arch", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--dilations", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--split-seed", type=int, default=20260825)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    split_rng = np.random.default_rng(args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann = load_ann(args.ann)
    ann = ann[ann.ego & (ann.onset >= 0)]
    ann = ann[[(args.feats / f"{v}.npz").is_file() for v in ann.vid]]
    public = {"000001", "000002", "000003", "000004", "000005"}
    data = {v: (load_feats(args.feats, v, False), int(o)) for v, o in zip(ann.vid, ann.onset)}
    train_pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
    srcs = train_pool.src.to_numpy()
    fold_of = {s: i % args.folds for i, s in enumerate(split_rng.permutation(sorted(set(srcs))))}
    train_pool["fold"] = [fold_of[s] for s in srcs]
    print(f"ego clips {len(train_pool)} | in_dim {next(iter(data.values()))[0].shape[1]}")

    oof, fold_hits, oof_long = {}, [], []
    for k in range(args.folds):
        tr = train_pool[train_pool.fold != k]
        va = train_pool[train_pool.fold == k]
        tr_seqs = [data[v] for v in tr.vid]
        va_seqs = [data[v] for v in va.vid]
        allx = np.concatenate([x for x, _ in tr_seqs])
        if args.arch == "gru":
            from src.train.train_stage2_collision import Localizer

            model = Localizer(allx.shape[1], hidden=args.hidden).to(device)
        else:
            model = LocalCNN(allx.shape[1], hidden=args.hidden, dilations=tuple(args.dilations)).to(device)
        model.in_mean.copy_(torch.from_numpy(allx.mean(0)))
        model.in_std.copy_(torch.from_numpy(allx.std(0) + 1e-6))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        pool = [x[: max(o - 5, 5)] for x, o in tr_seqs]  # 사고 전 정상 주행 구간만
        # 검증도 길이 무작위화 버전으로(고정 시드): 앞 0~100 + 뒤 0~100
        vrng = np.random.default_rng(k)
        va_long = []
        for x, o in va_seqs:
            pre = _segment(pool, int(vrng.integers(0, 101)), vrng)
            post = _segment(pool, int(vrng.integers(0, 101)), vrng)
            va_long.append((np.concatenate([pre, x, post]), o + len(pre)))
        best, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(len(tr_seqs))
            for s in range(0, len(order), args.batch):
                items = []
                for i in order[s : s + args.batch]:
                    x, o = augment_pos(*tr_seqs[i], pool, rng)
                    items.append((x, soft_target(len(x), o, args.sigma)))
                X, Y, M = collate(items, device)
                logits = model(X)
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, Y, reduction="none") * M).sum() / M.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            h, _ = hit_rate(model, va_long, device)
            if h > best:
                best, best_state = h, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best_state)
        h_orig, preds = hit_rate(model, va_seqs, device)
        h_long, _ = hit_rate(model, va_long, device)
        fold_hits.append(h_orig)
        oof_long.append(h_long)
        for v, p in zip(va.vid, preds):
            oof[v] = p
        torch.save({"model": best_state, "config": {**model.cfg, "arch": args.arch}}, args.out / f"fold{k}.pt")
        print(f"fold {k}: val hit(±3) orig {h_orig:.3f} | long(앞뒤 0~100) {h_long:.3f} (n={len(va)})", flush=True)
    err = np.array([oof[v] - data[v][1] for v in train_pool.vid])
    print(f"OOF hit(±3) orig = {np.mean(np.abs(err) <= 3):.3f} | long = {np.mean(oof_long):.3f} | err pct 5/25/50/75/95: {np.percentile(err, [5, 25, 50, 75, 95]).astype(int).tolist()}")
    json.dump({"fold_hits": fold_hits, "fold_hits_long": oof_long, "oof_hit": float(np.mean(np.abs(err) <= 3)), "args": {k: str(v) for k, v in vars(args).items()}}, open(args.out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
