"""공식 inference.py에 붙여넣을 Stage3 predict 함수 — EXP-S3-MOTION-001 (appearance + optical-flow ego-motion).

self-contained: submit.zip에는 src/가 없으므로 모델 클래스·전처리·motion 특징 계산을 여기서 다시 정의한다.
학습 쪽과 반드시 동일해야 하는 것: (1) 프레임 전처리(짧은 변 224 리사이즈+중앙 crop, 160x120 gray INTER_AREA),
(2) Farneback 파라미터와 13개 요약 특징 순서(src/train/stage3_motion.py flow_features), (3) 16프레임 윈도우 집계와
다중 지평 log-ratio(motion_clip_features), (4) motion 표준화(체크포인트 버퍼 motion_mean/std), (5) ResNet 특징 float16 캐스팅.
비공개 영상이 10fps여도 그대로 stride-1 흐름을 쓴다 — 학습이 20fps/10fps(시뮬레이션) 둘 다에 노출됐다(--fps-aug).
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

_ACCEL_LABELS = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
_STEER_LABELS = ["LEFT", "STRAIGHT", "RIGHT"]
_CLIP_SIZE = 224
_CLIP_FRAMES = 16
_MOTION_W, _MOTION_H = 160, 120
_MOTION_DIM = 13
_LOGRATIO_IDX = [0, 3, 9, 10]  # mag_mean, div, mag_bottom, mag_top
_DIFF_IDX = [11]  # static_frac
_SMOOTH_FRAMES = 31  # 영상 내 logit 이동평균 폭(프레임). 라벨 구간이 수 초 단위라 프레임별 flicker 제거 (EXP-S3-SMOOTH-001)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _Stage3ResNetMotionHead(nn.Module):
    def __init__(self, motion_dim: int, use_appearance: bool, horizons):
        super().__init__()
        backbone = resnet18(weights=None)  # 가중치는 state_dict로 채움(인터넷 불필요)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.use_appearance = use_appearance
        in_dim = (512 if use_appearance else 0) + motion_dim
        self.register_buffer("motion_horizons", torch.tensor(list(horizons), dtype=torch.long))
        self.register_buffer("motion_mean", torch.zeros(motion_dim))
        self.register_buffer("motion_std", torch.ones(motion_dim))
        self.accel = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))
        self.steer = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def heads(self, x: torch.Tensor):
        return self.accel(x), self.steer(x)


def _decode_rgb_and_gray(path: Path):
    """한 번 디코딩해 (N,224,224,3) RGB 중앙크롭과 (N,120,160) gray를 같이 만든다."""
    cap = cv2.VideoCapture(str(path))
    rgbs, grays = [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        grays.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (_MOTION_W, _MOTION_H), interpolation=cv2.INTER_AREA))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = _CLIP_SIZE / min(h, w)
        nh, nw = max(_CLIP_SIZE, round(h * scale)), max(_CLIP_SIZE, round(w * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        y, x = (nh - _CLIP_SIZE) // 2, (nw - _CLIP_SIZE) // 2
        rgbs.append(rgb[y : y + _CLIP_SIZE, x : x + _CLIP_SIZE])
    cap.release()
    if not rgbs:
        raise ValueError(f"cannot decode video: {path.name}")
    return np.stack(rgbs), np.stack(grays)


def _radial_grid(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xs - (w - 1) / 2) / (w / 2)
    yn = (ys - (h - 1) / 2) / (h / 2)
    r = np.sqrt(xn * xn + yn * yn) + 1e-6
    return xn / r, yn / r


def _flow_features(prev, nxt, rx, ry) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    h, w = mag.shape
    return np.array(
        [
            mag.mean(), u.mean(), v.mean(), float((u * rx + v * ry).mean()), mag.std(),
            u[:, : w // 2].mean(), u[:, w // 2 :].mean(), v[: h // 2].mean(), v[h // 2 :].mean(),
            mag[h // 2 :].mean(), mag[: h // 2].mean(), float((mag < 0.15).mean()),
            float(np.abs(prev.astype(np.float32) - nxt.astype(np.float32)).mean() / 255.0),
        ],
        dtype=np.float32,
    )


def _motion_per_frame(gray: np.ndarray) -> np.ndarray:
    """stride 1 (i -> i+1). 마지막 프레임은 직전 값 복제. (N, 13)"""
    n = len(gray)
    rx, ry = _radial_grid(gray.shape[1], gray.shape[2])
    out = np.zeros((n, _MOTION_DIM), dtype=np.float32)
    for i in range(n - 1):
        out[i] = _flow_features(gray[i], gray[i + 1], rx, ry)
    if n > 1:
        out[n - 1] = out[n - 2]
    return out


def _motion_clip_features(m: np.ndarray, frame_idx: np.ndarray, horizons) -> np.ndarray:
    n, k = m.shape
    h = _CLIP_FRAMES // 2
    out = np.empty((len(frame_idx), 3 * k), dtype=np.float32)
    m32 = m.astype(np.float32)
    for r, i in enumerate(frame_idx):
        idx = np.clip(int(i) - h + np.arange(_CLIP_FRAMES), 0, n - 1)
        w = m32[idx]
        out[r, :k] = w.mean(0)
        out[r, k : 2 * k] = w.std(0)
        out[r, 2 * k :] = w[h:].mean(0) - w[:h].mean(0)
    if not horizons:
        return out
    c = np.concatenate([np.zeros((1, k), np.float64), np.cumsum(m32.astype(np.float64), 0)])
    i = np.clip(frame_idx.astype(int), 0, n - 1)
    extras = [out]
    for H in horizons:
        lo, hi = np.clip(i - H, 0, n), np.clip(i + H, 0, n)
        nb, na = np.maximum(i - lo, 1)[:, None], np.maximum(hi - i, 1)[:, None]
        before, after = (c[i] - c[lo]) / nb, (c[hi] - c[i]) / na
        valid = ((i - lo) > 0) & ((hi - i) > 0)
        lr = np.log(np.maximum(after[:, _LOGRATIO_IDX], 0) + 1e-3) - np.log(np.maximum(before[:, _LOGRATIO_IDX], 0) + 1e-3)
        df = after[:, _DIFF_IDX] - before[:, _DIFF_IDX]
        ex = np.concatenate([lr, df], 1).astype(np.float32)
        ex[~valid] = 0.0
        extras.append(ex)
    return np.concatenate(extras, 1)


def _smooth_logits(logits: np.ndarray, w: int = _SMOOTH_FRAMES) -> np.ndarray:
    """(N, C) logit을 프레임 축으로 폭 w 중심 이동평균(가장자리는 edge pad). 한 영상 안에서만 — 파일 간 통계 없음."""
    if w <= 1 or len(logits) < 2:
        return logits
    pad = w // 2
    lp = np.pad(logits, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(w, dtype=np.float64) / w
    return np.stack([np.convolve(lp[:, j], k, mode="valid") for j in range(logits.shape[1])], 1)


def _clip_features(feats: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
    n = len(feats)
    out = np.empty((len(frame_idx), feats.shape[1]), dtype=np.float32)
    f32 = feats.astype(np.float32)
    for r, i in enumerate(frame_idx):
        idx = np.clip(int(i) - _CLIP_FRAMES // 2 + np.arange(_CLIP_FRAMES), 0, n - 1)
        out[r] = f32[idx].mean(0)
    return out


def _frame_features(model, rgb: np.ndarray, device, batch: int = 128) -> np.ndarray:
    """학습 캐시(src/train/stage3_features.py)와 동일: /255 -> ImageNet 정규화 -> ResNet18 -> float16 저장."""
    mean, std = _IMAGENET_MEAN.to(device), _IMAGENET_STD.to(device)
    feats = []
    for s in range(0, len(rgb), batch):
        x = torch.from_numpy(rgb[s : s + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.backbone(x)
        feats.append(f.float().cpu().numpy().astype(np.float16))
    return np.concatenate(feats)


def _load_stage3_models(model_dir, device):
    """model_dir/best*.pt 앙상블. 멤버마다 특징 구성(외형 사용 여부)과 역할 가중치(role_mask=[w_accel, w_steer])가 달라도 된다:
    accel logit은 w_accel, steer logit은 w_steer로 가중 평균한다(EXP-S3-HYBRID-001: accel은 외형+모션, steer는 모션 전용이 차종 이동에 강함).
    반환 members=[(model, accel_bias, steer_bias, m_mean, m_std, use_app, w_accel, w_steer)], has_motion, horizons, any_app"""
    paths = sorted(Path(model_dir).glob("best*.pt"))
    members, meta = [], None
    for pth in paths:
        state = torch.load(pth, map_location="cpu", weights_only=False)
        accel_bias = state.pop("accel_bias", torch.zeros(4)).numpy()
        steer_bias = state.pop("steer_bias", torch.zeros(3)).numpy()
        role = state.pop("role_mask", torch.ones(2)).float().tolist()
        has_motion = "motion_mean" in state
        motion_dim = int(state["motion_mean"].shape[0]) if has_motion else 0
        horizons = [int(h) for h in state["motion_horizons"].tolist()] if "motion_horizons" in state else []
        use_app = int(state["accel.0.weight"].shape[1]) > motion_dim
        model = _Stage3ResNetMotionHead(motion_dim, use_app, horizons)
        model.load_state_dict(state, strict=has_motion)
        model.to(device).eval()
        m_mean, m_std = (state["motion_mean"].numpy(), state["motion_std"].numpy()) if has_motion else (None, None)
        this = (has_motion, tuple(horizons))
        if meta is None:
            meta = this
        elif this != meta:
            raise ValueError(f"앙상블 멤버 motion 구성이 다릅니다: {pth.name}")
        members.append((model, accel_bias, steer_bias, m_mean, m_std, use_app, float(role[0]), float(role[1])))
    if not members:
        raise FileNotFoundError(f"{model_dir}/best*.pt 없음")
    return members, meta[0], list(meta[1]), any(mb[5] for mb in members)


def predict_stage3(data_dir, model_dir):
    device = _device()
    members, has_motion, horizons, any_app = _load_stage3_models(model_dir, device)
    backbone_owner = next((mb[0] for mb in members if mb[5]), members[0][0])  # backbone은 공통(ImageNet frozen)
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    with torch.inference_mode():
        for path in videos:
            rgb, gray = _decode_rgb_and_gray(path)
            frame_idx = np.arange(len(rgb))
            app_clip = _clip_features(_frame_features(backbone_owner, rgb, device), frame_idx) if any_app else None
            motion_clip = _motion_clip_features(_motion_per_frame(gray), frame_idx, horizons) if has_motion else None
            la_sum, ls_sum, wa_sum, ws_sum = None, None, 0.0, 0.0
            for mdl, accel_bias, steer_bias, m_mean, m_std, use_app, w_a, w_s in members:
                feats = [app_clip] if use_app else []
                if has_motion:
                    feats.append(((motion_clip - m_mean) / m_std).astype(np.float32))
                x = torch.from_numpy(np.concatenate(feats, 1)).to(device)
                la, ls = mdl.heads(x)
                la = (la.float().cpu().numpy() + accel_bias) * w_a
                ls = (ls.float().cpu().numpy() + steer_bias) * w_s
                la_sum = la if la_sum is None else la_sum + la
                ls_sum = ls if ls_sum is None else ls_sum + ls
                wa_sum += w_a
                ws_sum += w_s
            la_sum, ls_sum = la_sum / max(wa_sum, 1e-6), ls_sum / max(ws_sum, 1e-6)
            accel_idx = _smooth_logits(la_sum).argmax(1).tolist()
            steer_idx = _smooth_logits(ls_sum).argmax(1).tolist()
            for sample_index, a, s in zip(frame_idx.tolist(), accel_idx, steer_idx):
                rows.append({"ID": path.stem, "sample_index": sample_index, "accel_label": _ACCEL_LABELS[a], "steer_label": _STEER_LABELS[s]})
    del members, backbone_owner
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
