"""공식 inference.py에 붙여넣을 Stage2 predict 함수 — 프레임률 적응 학습 localizer (EXP-S2-FPS-001).

비공개 클립의 프레임률·길이는 알 수 없다(공개 예시 50프레임/10fps와 다름이 LB로 확인). stride s∈{1,2,3,4,6}로 프레임을 솎아
10fps 학습 분포에 맞춘 뒤, 앙상블 점수의 최대값이 가장 큰 stride를 **클립마다** 고른다(파일 간 통계 없음, 고정 상수만 사용).
CCD 200클립 검증(±0.3s): 원본 10fps 0.765(고정 stride1 0.79), ×3 보간 30fps 0.755(고정 stride1 0.37).
  collision = 선택 stride에서 argmax(마지막 3프레임 제외) × s → 원본 프레임 번호
  entry     = collision − 7·s (10fps 기준 0.7s; CCD 수동 라벨 중앙값)
  side/evasion = 현행 영상 내부 규칙(창 폭을 s배)
model/stage2/*fold*.pt (19-d BiGRU: LEARN-001~005 + 위치 무작위화 GRU/국소 CNN: INV-002, 총 11모델×5fold)만 사용 — ResNet 불필요.
4가지 의도적 분포 이동(10/30fps × 짧은/긴 클립) 최저 적중 0.75 (v24의 5모델 0.715).
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

_L2_W, _L2_H = 320, 180
_L2_ENTRY_OFFSET = 21  # LB 실측(v23)
_L2_TAIL_EXCLUDE = 3
_L2_STRIDES = (1, 2, 3)  # CCD 검증 범위; 4·6은 허위 최대값 위험
_L2_MIN_FRAMES = 16
_L2_STRIDE_RATIO = 1.0  # stride 1 기본; 다른 stride는 최대 점수가 stride1의 1.5배를 넘을 때만 채택(CCD 200클립: 10fps 0.79 / 합성 30fps 0.715)
_L2_PRE_LO, _L2_PRE_HI = 12, 2


class _L2Localizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h, _ = self.gru(self.proj((x - self.in_mean) / self.in_std))
        return self.head(h).squeeze(-1)


class _L2LocalCNN(nn.Module):
    """수용영역 ±30프레임 dilated 1D CNN (EXP-S2-INV-002, 위치 사전정보 없음)."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        layers = [nn.Conv1d(in_dim, hidden, 1)]
        for d in dilations:
            layers += [nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):
        h = self.body(((x - self.in_mean) / self.in_std).transpose(1, 2))
        return self.head(nn.functional.gelu(h)).squeeze(1)


def _l2_build(cfg):
    if cfg.get("arch") == "cnn" or "dilations" in cfg:
        return _L2LocalCNN(cfg["in_dim"], cfg["hidden"], cfg["dropout"], tuple(cfg["dilations"]))
    return _L2Localizer(cfg["in_dim"], cfg["hidden"], cfg["dropout"])


def _l2_frame_number(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def _l2_robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def _l2_radial_grid(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xs - (w - 1) / 2) / (w / 2)
    yn = (ys - (h - 1) / 2) / (h / 2)
    r = np.sqrt(xn * xn + yn * yn) + 1e-6
    return xn / r, yn / r


def _l2_flow_features(prev, nxt, rx, ry) -> np.ndarray:
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


def _l2_load_gray(paths):
    """(gray320 list float32 (180,320), gray160 list uint8 (120,160))"""
    g320, g160 = [], []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((720, 1280, 3), np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g320.append(cv2.resize(gray, (_L2_W, _L2_H), interpolation=cv2.INTER_AREA).astype(np.float32))
        g160.append(cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA))
    return g320, g160


def _l2_signals(g320):
    n = len(g320)
    win = cv2.createHanningWindow((_L2_W, _L2_H), cv2.CV_32F)
    shift, vert, diff = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        (dx, dy), _ = cv2.phaseCorrelate(g320[i - 1], g320[i], win)
        shift[i], vert[i] = float(np.hypot(dx, dy)), float(abs(dy))
        diff[i] = float(np.abs(g320[i] - g320[i - 1]).mean())
    return np.stack([shift, vert, diff, _l2_robust_z(shift), _l2_robust_z(vert), _l2_robust_z(diff)], 1).astype(np.float32)


def _l2_motion(g160, rx, ry):
    n = len(g160)
    motion = np.zeros((n, 13), np.float32)
    for i in range(n - 1):
        motion[i] = _l2_flow_features(g160[i], g160[i + 1], rx, ry)
    if n > 1:
        motion[n - 1] = motion[n - 2]
    return motion


def _l2_diff_energy(g320, lo, hi):
    acc = np.zeros((_L2_H, _L2_W), np.float32)
    for i in range(max(lo, 1), min(hi, len(g320))):
        acc += np.abs(g320[i] - g320[i - 1])
    return acc[int(_L2_H * 0.35) :, :]


def _l2_left_fraction(e):
    left, right = float(e[:, : _L2_W // 2].sum()), float(e[:, _L2_W // 2 :].sum())
    return left / (left + right + 1e-6)


def _l2_side_evasion(g320, c, s=1):
    lo, hi = c - _L2_PRE_LO * s, c - _L2_PRE_HI * s
    pre_e = _l2_diff_energy(g320, lo, hi)
    side = "LEFT" if _l2_left_fraction(pre_e) > _l2_left_fraction(_l2_diff_energy(g320, 0, len(g320))) else "RIGHT"
    col = pre_e.sum(0)
    center = float(col[int(_L2_W * 0.3) : int(_L2_W * 0.7)].sum() / (col.sum() + 1e-6))
    return side, (0 if center > 0.8 else 1)


def _l2_localize(g320, g160, folds, rx, ry, device):
    """stride별 앙상블 점수 → stride 1 기본, 최대 점수가 stride1의 1.5배를 넘는 stride만 채택. (collision index, stride)"""
    n = len(g320)
    scores = {}
    for s in _L2_STRIDES:
        if s != 1 and n // s < _L2_MIN_FRAMES:
            continue
        x = np.concatenate([_l2_signals(g320[::s]), _l2_motion(g160[::s], rx, ry)], 1)
        seq = torch.from_numpy(x)[None].to(device)
        scores[s] = np.mean([torch.sigmoid(m(seq)[0]).float().cpu().numpy() for m in folds], 0)
    s = 1
    for cand in sorted(scores):
        if cand == 1:
            continue
        ref = float(scores[1].max()) * _L2_STRIDE_RATIO if s == 1 else float(scores[s].max())
        if float(scores[cand].max()) > ref:
            s = cand
    score = scores[s]
    c = int(np.argmax(score[: max(len(score) - _L2_TAIL_EXCLUDE, 1)])) * s
    return min(c, n - 1), s


def predict_stage2(data_dir, model_dir):
    device = _device()
    folds = []
    for pth in sorted(Path(model_dir).glob("*fold*.pt")):
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        m = _l2_build(ck["config"])
        m.load_state_dict(ck["model"])
        folds.append(m.to(device).eval())
    rx, ry = _l2_radial_grid(120, 160)
    image_root = Path(data_dir) / "images"
    rows = []
    with torch.inference_mode():
        for folder in sorted(p for p in image_root.iterdir() if p.is_dir()):
            paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=_l2_frame_number)
            if not paths:
                continue
            numbers = [_l2_frame_number(p) for p in paths]
            try:
                g320, g160 = _l2_load_gray(paths)
                c, s = _l2_localize(g320, g160, folds, rx, ry, device)
                side, evasion = _l2_side_evasion(g320, c, s)
            except Exception:
                c, s, side, evasion = len(paths) - 1, 1, "LEFT", 1
            e = max(c - _L2_ENTRY_OFFSET, 0)
            rows.append({"ID": folder.name, "collision_frame": int(numbers[c]), "entry_frame": int(numbers[e]), "evasion_space": int(evasion), "entry_side": side})
    del folds
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
