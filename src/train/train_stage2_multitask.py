"""EXP-S2-MT-001 — collision head + entry head 독립 multi-task temporal localizer.

동기: 지금까지 entry = collision − 상수(0.7s). Stage2 점수의 70%가 collision 피크 하나에 종속. 공유 트렁크(BiGRU) 위에
  - collision head: 전 ego 클립(796) onset soft target
  - entry head: 수동 라벨 클립(~74) entry(onset+offset) soft target, 나머지는 손실 마스크
학습 증강은 train_stage2_local.augment_pos(앞뒤 정상 주행 이어붙임; 두 라벨을 같이 이동).
평가(fold별 OOF): collision hit(±3), 라벨 클립에서 entry hit(±3) — entry head argmax(창 [c−15, c−1]) vs 상수(c−7), c=모델 collision 예측.

실행: python -m src.train.train_stage2_multitask --out output/exp_s2_mt_001 [--seed N] [--lam 1.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.train.train_stage2_collision import load_ann, load_feats, soft_target
from src.train.train_stage2_local import _segment


class MultiTaskLocalizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head_c = nn.Linear(2 * hidden, 1)
        self.head_e = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.cfg = {"in_dim": in_dim, "hidden": hidden, "dropout": dropout, "arch": "gru_mt"}

    def forward(self, x):  # (B,N,D) -> (B,N), (B,N)
        h, _ = self.gru(self.proj((x - self.in_mean) / self.in_std))
        return self.head_c(h).squeeze(-1), self.head_e(h).squeeze(-1)


def collate(items, device):
    """items: [(x (N,D), y (N,2))] → X (B,N,D), Y (B,N,2), M (B,N)"""
    n = max(len(x) for x, _ in items)
    X = torch.zeros(len(items), n, items[0][0].shape[1])
    Y = torch.zeros(len(items), n, 2)
    M = torch.zeros(len(items), n)
    for i, (x, y) in enumerate(items):
        X[i, : len(x)] = torch.from_numpy(x)
        Y[i, : len(x)] = torch.from_numpy(y)
        M[i, : len(x)] = 1
    return X.to(device), Y.to(device), M.to(device)


def augment(x, onset, entry, pre_pool, rng):
    if rng.random() < 0.8:
        seg = _segment(pre_pool, int(rng.integers(0, 101)), rng)
        x, onset, entry = np.concatenate([seg, x]), onset + len(seg), (entry + len(seg) if entry is not None else None)
    if rng.random() < 0.8:
        x = np.concatenate([x, _segment(pre_pool, int(rng.integers(0, 101)), rng)])
    if rng.random() < 0.15:
        x, onset, entry = np.repeat(x, 2, axis=0), onset * 2, (entry * 2 if entry is not None else None)
    return x, onset, entry


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    sc, se = model(torch.from_numpy(x)[None].to(device))
    return sc[0].cpu().numpy(), se[0].cpu().numpy()


def entry_from_head(se, c, lo=15, hi=1):
    a, b = max(c - lo, 0), max(c - hi, 1)
    return int(a + np.argmax(se[a:b])) if b > a else max(c - 7, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, default=Path("output/stage2_ccd_feats"))
    ap.add_argument("--ann", type=Path, default=Path("data/external/ccd/Crash-1500.txt"))
    ap.add_argument("--entry-labels", type=Path, default=Path("output/ccd_entry_labels_v2.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lam", type=float, default=1.0, help="entry 손실 가중치")
    ap.add_argument("--sigma", type=float, default=1.5)
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
    ent = pd.read_csv(args.entry_labels, dtype={"vid": str})
    entry_off = {v: int(o) for v, o in zip(ent.vid, ent.entry_offset)}
    data = {v: (load_feats(args.feats, v, False), int(o), (int(o) + entry_off[v]) if v in entry_off else None) for v, o in zip(ann.vid, ann.onset)}
    pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
    srcs = pool.src.to_numpy()
    fold_of = {s: i % 5 for i, s in enumerate(split_rng.permutation(sorted(set(srcs))))}
    pool["fold"] = [fold_of[s] for s in srcs]
    n_lab = sum(1 for v in pool.vid if data[v][2] is not None)
    print(f"ego clips {len(pool)} | entry-labeled {n_lab} | in_dim {next(iter(data.values()))[0].shape[1]}")

    res = {"c_hit": [], "e_hit_head": [], "e_hit_const7": [], "e_hit_const6": [], "e_err_head": [], "e_err_const": []}
    for k in range(5):
        tr = pool[pool.fold != k]
        va = pool[pool.fold == k]
        tr_seqs = [data[v] for v in tr.vid]
        allx = np.concatenate([x for x, _, _ in tr_seqs])
        model = MultiTaskLocalizer(allx.shape[1], hidden=args.hidden).to(device)
        model.in_mean.copy_(torch.from_numpy(allx.mean(0)))
        model.in_std.copy_(torch.from_numpy(allx.std(0) + 1e-6))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        pre_pool = [x[: max(o - 5, 5)] for x, o, _ in tr_seqs]
        # 라벨 클립 오버샘플링(entry head 학습 신호 확보)
        lab_idx = [i for i, (_, _, e) in enumerate(tr_seqs) if e is not None]
        best, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            order = np.concatenate([rng.permutation(len(tr_seqs)), rng.choice(lab_idx, size=len(lab_idx) * 3) if lab_idx else np.array([], int)])
            order = rng.permutation(order)
            for s in range(0, len(order), args.batch):
                items, masks = [], []
                for i in order[s : s + args.batch]:
                    x, o, e = augment(*tr_seqs[i], pre_pool, rng)
                    yc = soft_target(len(x), o, args.sigma)
                    ye = soft_target(len(x), e, args.sigma) if e is not None else np.zeros(len(x), np.float32)
                    items.append((x, np.stack([yc, ye], 1)))
                    masks.append(1.0 if e is not None else 0.0)
                X, Y, M = collate(items, device)  # Y: (B,N,2)
                lc, le = model(X)
                loss_c = (nn.functional.binary_cross_entropy_with_logits(lc, Y[..., 0], reduction="none") * M).sum() / M.sum()
                me = M * torch.tensor(masks, device=device)[:, None]
                loss_e = (nn.functional.binary_cross_entropy_with_logits(le, Y[..., 1], reduction="none") * me).sum() / me.sum().clamp(min=1.0)
                loss = loss_c + args.lam * loss_e
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            # 선택 기준: collision hit + entry hit(라벨 val)
            ch, eh = [], []
            for v in va.vid:
                x, o, e = data[v]
                sc, se = predict(model, x, device)
                c = int(np.argmax(sc[:-3]))
                ch.append(abs(c - o) <= 3)
                if e is not None:
                    eh.append(abs(entry_from_head(se, c) - e) <= 3)
            score = float(np.mean(ch)) + (float(np.mean(eh)) if eh else 0.0)
            if score > best:
                best, best_state = score, {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best_state)
        for v in va.vid:
            x, o, e = data[v]
            sc, se = predict(model, x, device)
            c = int(np.argmax(sc[:-3]))
            res["c_hit"].append(abs(c - o) <= 3)
            if e is not None:
                eh = entry_from_head(se, c)
                res["e_hit_head"].append(abs(eh - e) <= 3)
                res["e_hit_const7"].append(abs(max(c - 7, 0) - e) <= 3)
                res["e_hit_const6"].append(abs(max(c - 6, 0) - e) <= 3)
                res["e_err_head"].append(eh - e)
                res["e_err_const"].append(max(c - 7, 0) - e)
        torch.save({"model": best_state, "config": model.cfg}, args.out / f"fold{k}.pt")
        print(f"fold {k}: collision {np.mean(res['c_hit']):.3f} (cum) | entry head {np.mean(res['e_hit_head']) if res['e_hit_head'] else float('nan'):.3f} const7 {np.mean(res['e_hit_const7']) if res['e_hit_const7'] else float('nan'):.3f} (n={len(res['e_hit_head'])})", flush=True)
    summary = {k: float(np.mean(v)) for k, v in res.items() if k.endswith("hit") or k.startswith("e_hit")}
    summary["n_entry"] = len(res["e_hit_head"])
    summary["e_err_head_p10_50_90"] = np.percentile(res["e_err_head"], [10, 50, 90]).tolist() if res["e_err_head"] else None
    summary["e_err_const_p10_50_90"] = np.percentile(res["e_err_const"], [10, 50, 90]).tolist() if res["e_err_const"] else None
    print("OOF:", json.dumps(summary))
    json.dump({**summary, "args": {k: str(v) for k, v in vars(args).items()}}, open(args.out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
