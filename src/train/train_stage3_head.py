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
from .stage3_model_light import Stage3ResNetHead, Stage3ResNetMotionHead
from .stage3_motion import motion_clip_features

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


def build_xy(labels: pd.DataFrame, feat_dir: Path, temporal: bool = False, motion_dir: Path | None = None, sim10: bool = False, horizons: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sim10=True: 20fps 캐시에서 짝수 프레임만 취하고(frame_index//2) motion은 stride-2 채널을 써서
    비공개 10fps 영상을 시뮬레이션(EXP-S3-MOTION-001 --fps-aug). motion_dir가 있으면 3K-d motion 특징을 뒤에 concat."""
    xs, ya, ys = [], [], []
    fn = clip_features_temporal if temporal else clip_features
    for sid, g in labels.groupby("segment_id", sort=False):
        name = sid.replace("/", "__") + ".npy"
        f = np.load(feat_dir / name)
        fi = g["frame_index"].to_numpy()
        m = None
        if motion_dir is not None:
            m = np.load(motion_dir / name)  # (N, 2, K)
            assert len(m) == len(f), f"{sid}: motion {len(m)} vs feats {len(f)} 프레임 수 불일치"
            m = m[::2, 1, :] if sim10 else m[:, 0, :]
        if sim10:
            f, fi = f[::2], fi // 2
        x = fn(f, fi)
        if m is not None:
            x = np.concatenate([x, motion_clip_features(m, fi, horizons=horizons)], 1)
        xs.append(x)
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
    parser.add_argument("--motion", type=Path, default=None, help="stage3_motion.py 캐시 디렉터리 - optical-flow ego-motion 특징 concat (EXP-S3-MOTION-001)")
    parser.add_argument("--fps-aug", action="store_true", help="10fps 시뮬레이션(짝수 프레임+stride-2 motion) 샘플을 학습에 추가, val은 20fps/10fps 둘 다 보고")
    parser.add_argument("--no-appearance", action="store_true", help="motion 특징만 사용(ablation)")
    parser.add_argument("--holdout-route", default="", help="쉼표 구분 route id — train/val 모두에서 제외(공개 OPEN 원본 route 등)")
    parser.add_argument("--motion-horizons", default="16,32,64,128", help="다중 지평 log-ratio 특징의 H(프레임) 목록, '0'이면 없음")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(args.labels)
    horizons = tuple(int(h) for h in args.motion_horizons.split(",") if int(h) > 0) if args.motion is not None else ()
    segs = sorted(labels["segment_id"].unique())
    holdout = {r for r in args.holdout_route.split(",") if r}
    if holdout:
        before = len(segs)
        segs = [s_ for s_ in segs if route_id(s_) not in holdout]
        print(f"holdout routes {sorted(holdout)}: {before - len(segs)} segments excluded")
    train_segs, val_segs = group_train_val_split(segs, val_ratio=args.val_ratio, seed=args.seed)
    assert_no_route_leakage(train_segs, val_segs)
    tr, va = labels[labels["segment_id"].isin(train_segs)], labels[labels["segment_id"].isin(val_segs)]
    print(f"train {len(train_segs)} seg/{len({route_id(s) for s in train_segs})} route ({len(tr)} samples) | val {len(val_segs)} seg/{len({route_id(s) for s in val_segs})} route ({len(va)} samples)")

    t0 = time.time()
    xtr, atr, str_ = build_xy(tr, args.features, args.temporal, args.motion, horizons=horizons)
    xva, ava, sva = build_xy(va, args.features, args.temporal, args.motion, horizons=horizons)
    xva10 = ava10 = sva10 = None
    if args.fps_aug:
        xtr10, atr10, str10 = build_xy(tr, args.features, args.temporal, args.motion, sim10=True, horizons=horizons)
        xva10, ava10, sva10 = build_xy(va, args.features, args.temporal, args.motion, sim10=True, horizons=horizons)
        xtr, atr, str_ = np.concatenate([xtr, xtr10]), np.concatenate([atr, atr10]), np.concatenate([str_, str10])
    motion_mean = motion_std = None
    if args.motion is not None:
        motion_total = xtr.shape[1] - int(np.load(args.features / (sorted(labels.segment_id.unique())[0].replace('/', '__') + '.npy'), mmap_mode='r').shape[1]) * (2 if args.temporal else 1)
        app_dim = xtr.shape[1] - motion_total  # 특징 파일 차원에서 유도(ResNet 512 / DINOv2 384)
        print(f"app_dim={app_dim} motion_dim={motion_total}")
        motion_mean = xtr[:, app_dim:].mean(0)
        motion_std = xtr[:, app_dim:].std(0) + 1e-6

        def _prep(x):  # motion 표준화 (+ appearance 제거 옵션)
            a_, m_ = x[:, :app_dim], (x[:, app_dim:] - motion_mean) / motion_std
            return m_ if args.no_appearance else np.concatenate([a_, m_], 1)

        xtr, xva = _prep(xtr), _prep(xva)
        if xva10 is not None:
            xva10 = _prep(xva10)
    print(f"features built in {time.time() - t0:.0f}s: xtr {xtr.shape} xva {xva.shape}" + (f" xva10 {xva10.shape}" if xva10 is not None else ""))
    print("val accel dist:", Counter(ava.tolist()), "val steer dist:", Counter(sva.tolist()))

    if args.motion is not None:
        full = Stage3ResNetMotionHead(motion_dim=len(motion_mean), pretrained=True, use_appearance=not args.no_appearance, horizons=horizons, app_dim=app_dim)
        full.motion_mean.copy_(torch.from_numpy(motion_mean.astype(np.float32))); full.motion_std.copy_(torch.from_numpy(motion_std.astype(np.float32)))
    else:
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
    Xva10 = torch.from_numpy(xva10).to(device) if xva10 is not None else None
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
        select = ev["stage3_score"]
        if Xva10 is not None:
            with torch.inference_mode():
                pa10 = head_a(Xva10).argmax(1).cpu().numpy(); ps10 = head_s(Xva10).argmax(1).cpu().numpy()
            ev10 = stage3_score(ava10, pa10, sva10, ps10)
            rec.update({"sim10_" + k: v for k, v in ev10.items()})
            select = (ev["stage3_score"] + ev10["stage3_score"]) / 2  # 비공개는 10fps라 두 fps 평균으로 선택
        rec["select_score"] = select
        history.append(rec)
        print(rec)
        if select > best:
            best, best_epoch = select, epoch
            full.accel.load_state_dict(head_a.state_dict()); full.steer.load_state_dict(head_s.state_dict())
            torch.save(full.state_dict(), args.out / "best.pt")

    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_select_score": best, "best_epoch_record": history[best_epoch] if best_epoch >= 0 else None, "train_segments": len(train_segs), "val_segments": len(val_segs), "temporal": args.temporal, "class_weight": args.class_weight, "motion": str(args.motion) if args.motion else None, "fps_aug": args.fps_aug, "no_appearance": args.no_appearance, "motion_horizons": list(horizons)}, f, ensure_ascii=False, indent=2)
    print(f"best select_score={best:.4f} (epoch {best_epoch}) -> {args.out / 'best.pt'}")
    if best_epoch >= 0:
        print("best epoch record:", history[best_epoch])


if __name__ == "__main__":
    main()
