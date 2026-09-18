"""EXP-S3-PRIOR-001 — 공개 OPEN 5영상(전체 프레임) 예측 클래스 분포를 공식 라벨 비율에 맞추는 logit bias 보정.

LB 실측(제출 2/5b/6): comma2k19 로컬 검증과 LB가 역상관이었고, 공통 원인 후보는 소수 클래스 과예측
(official 50행에서 LEFT+RIGHT 24 vs 정답 11). 공개 라벨 비율(accel CONSTANT .60/ACC .18/DEC .16/STOP .06,
steer STRAIGHT .78/LEFT .12/RIGHT .10)은 공개 데이터이므로 보정에 써도 된다(가이드 §12: 비공개 통계만 금지).
bias는 빌드 시 고정되는 상수 벡터라 추론 시 파일 간 통계를 쓰지 않는다.

방법: 캐시 특징으로 OPEN 전체 프레임 logit을 구하고(비공개가 10fps라 --sim10 경로 기본), 반복 비례 맞춤으로
argmax 분포의 L1 거리를 줄이는 bias를 찾는다. 결과는 체크포인트 사본(best_calibrated.pt)에 accel_bias/steer_bias
텐서로 저장 — 추론 snippet은 이 키가 있으면 logit에 더한다(state_dict 로드 전에 pop).

실행:
    python -m src.eval.calibrate_stage3_prior --ckpt output/exp_s3_base_005/best.pt [--motion output/stage3_motion_open] [--smooth 31]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.eval_stage3_official import _head, smooth_logits
from src.eval.metrics import score_stage3
from src.train.stage3_motion import motion_clip_features
from src.train.train_stage3_head import clip_features

ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]


def open_logits(state: dict, feat_dir: Path, motion_dir: Path | None, ids: list[str], sim10: bool):
    """{ID: (frame_idx, accel_logits (N,4), steer_logits (N,3))}"""
    head_a, head_s = _head(state, "accel", 4), _head(state, "steer", 3)
    has_motion = "motion_mean" in state
    in_dim = state["accel.0.weight"].shape[1]
    use_app = in_dim > (len(state["motion_mean"]) if has_motion else 0)
    horizons = tuple(state["motion_horizons"].tolist()) if "motion_horizons" in state else ()
    out = {}
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
            parts.append(((motion_clip_features(m, fi, horizons=horizons) - state["motion_mean"].numpy()) / state["motion_std"].numpy()).astype(np.float32))
        x = torch.from_numpy(np.concatenate(parts, 1))
        with torch.inference_mode():
            out[vid] = (fi, head_a(x).numpy(), head_s(x).numpy())
    return out


def fit_bias(logit_list: list[np.ndarray], target: np.ndarray, iters: int = 300, lr: float = 0.3, smooth: int = 1, max_bias: float = 1.0, tol: float = 0.05) -> np.ndarray:
    """max_bias로 크기를 제한하고(과도한 클래스 억제 방지), 분포 L1 거리가 tol 아래면 멈춘다."""
    b = np.zeros(target.shape[0])
    for _ in range(iters):
        counts = np.zeros_like(target)
        for l in logit_list:
            p = smooth_logits(l + b, smooth).argmax(1)
            counts += np.bincount(p, minlength=len(target))
        dist = counts / counts.sum()
        if np.abs(dist - target).sum() < tol:
            break
        b += lr * (np.log(target + 1e-3) - np.log(dist + 1e-3))
        b -= b.mean()
        b = np.clip(b, -max_bias, max_bias)
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=Path("output/stage3_features_open"))
    ap.add_argument("--motion", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=Path("data/stage3/labels.csv"))
    ap.add_argument("--smooth", type=int, default=31)
    ap.add_argument("--native", action="store_true", help="20fps 경로로 맞춤(기본은 비공개와 같은 10fps 시뮬레이션)")
    ap.add_argument("--out", type=Path, default=None, help="기본: ckpt 옆 best_calibrated.pt")
    ap.add_argument("--max-bias", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()
    gt = pd.read_csv(args.labels)
    ids = sorted(gt.ID.unique())
    tA = np.array([(gt.accel_label == k).mean() for k in ACCEL])
    tS = np.array([(gt.steer_label == k).mean() for k in STEER])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state.pop("accel_bias", None)
    state.pop("steer_bias", None)
    if args.motion is None and "motion_mean" in state:
        raise SystemExit("motion 체크포인트에는 --motion 이 필요합니다")
    sim10 = not args.native
    lg = open_logits(state, args.features, args.motion, ids, sim10)
    bA = fit_bias([v[1] for v in lg.values()], tA, smooth=args.smooth, max_bias=args.max_bias, tol=args.tol)
    bS = fit_bias([v[2] for v in lg.values()], tS, smooth=args.smooth, max_bias=args.max_bias, tol=args.tol)
    print("target accel", dict(zip(ACCEL, tA.round(3))), "| steer", dict(zip(STEER, tS.round(3))))
    print("bias accel", dict(zip(ACCEL, bA.round(3))), "| steer", dict(zip(STEER, bS.round(3))))

    def evaluate(ba, bs, label):
        rows, cnt_a, cnt_s = [], np.zeros(4), np.zeros(3)
        for vid, (fi, la, ls) in lg.items():
            pa = smooth_logits(la + ba, args.smooth).argmax(1)
            ps = smooth_logits(ls + bs, args.smooth).argmax(1)
            cnt_a += np.bincount(pa, minlength=4)
            cnt_s += np.bincount(ps, minlength=3)
            rows.append(pd.DataFrame({"ID": vid, "frame": fi, "accel_label": [ACCEL[i] for i in pa], "steer_label": [STEER[i] for i in ps]}))
        pred = pd.concat(rows)
        key = gt.frame_index // 2 if sim10 else gt.frame_index
        merged = gt[["ID", "sample_index"]].assign(frame=key).merge(pred, on=["ID", "frame"], how="left")
        res = score_stage3(merged[["ID", "sample_index", "accel_label", "steer_label"]], gt)
        dA, dS = cnt_a / cnt_a.sum(), cnt_s / cnt_s.sum()
        print(f"[{label}] official S3={res['stage3_score']:.4f} accel={res['accel_macro_f1']:.4f} steer={res['steer_macro_f1']:.4f} | pred dist accel {dict(zip(ACCEL, dA.round(2)))} L1={np.abs(dA - tA).sum():.2f} | steer {dict(zip(STEER, dS.round(2)))} L1={np.abs(dS - tS).sum():.2f}")

    evaluate(np.zeros(4), np.zeros(3), "before")
    evaluate(bA, bS, "after ")
    out = args.out or (args.ckpt.parent / "best_calibrated.pt")
    state["accel_bias"] = torch.tensor(bA, dtype=torch.float32)
    state["steer_bias"] = torch.tensor(bS, dtype=torch.float32)
    torch.save(state, out)
    print("saved", out)


if __name__ == "__main__":
    main()
