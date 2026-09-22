"""공식 inference.py에 붙여넣을 Stage2 predict 함수 — EXP-S2-LEARN-001 (CCD 사고 프레임 라벨로 학습한 충돌 localizer).

구성:
  - collision_frame: 프레임별 [전역 이동 3신호 + robust z 3 + Farneback 13 + ResNet18 512] 시퀀스 → BiGRU(5-fold 앙상블 평균) 점수 argmax.
    model/stage2/ 에 fold*.pt (train_stage2_collision.py 출력) 와 resnet18-f37072fd.pth(베이스라인 zip 동봉, ImageNet)가 있어야 한다.
  - entry_frame: collision - 7프레임 (CCD ego 40클립 수동 라벨 중앙값 -7, 10fps 기준)
  - entry_side / evasion_space: predict_stage2_heuristic.py와 동일한 영상 내부 규칙(방향은 신뢰할 단서가 없어 상대 좌측비율 유지).
파일 간 통계 없음. 프레임 번호는 파일명 숫자를 그대로 제출.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

_L2_W, _L2_H = 320, 180
_L2_ENTRY_OFFSET = 7
_L2_PRE_LO, _L2_PRE_HI = 12, 2
_L2_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_L2_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def _l2_load(paths):
    """(rgb224 (N,224,224,3) uint8, gray320 list float32, gray160 (N,120,160) uint8)"""
    rgb, g320, g160 = [], [], []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((720, 1280, 3), np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g320.append(cv2.resize(gray, (_L2_W, _L2_H), interpolation=cv2.INTER_AREA).astype(np.float32))
        g160.append(cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA))
        r = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = r.shape[:2]
        s = 224 / min(h, w)
        nh, nw = max(224, round(h * s)), max(224, round(w * s))
        r = cv2.resize(r, (nw, nh), interpolation=cv2.INTER_AREA)
        y, x = (nh - 224) // 2, (nw - 224) // 2
        rgb.append(r[y : y + 224, x : x + 224])
    return np.stack(rgb), g320, np.stack(g160)


def _l2_signals(g320):
    n = len(g320)
    win = cv2.createHanningWindow((_L2_W, _L2_H), cv2.CV_32F)
    shift, vert, diff = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        (dx, dy), _ = cv2.phaseCorrelate(g320[i - 1], g320[i], win)
        shift[i], vert[i] = float(np.hypot(dx, dy)), float(abs(dy))
        diff[i] = float(np.abs(g320[i] - g320[i - 1]).mean())
    return np.stack([shift, vert, diff, _l2_robust_z(shift), _l2_robust_z(vert), _l2_robust_z(diff)], 1).astype(np.float32)


def _l2_diff_energy(g320, lo, hi):
    acc = np.zeros((_L2_H, _L2_W), np.float32)
    for i in range(max(lo, 1), min(hi, len(g320))):
        acc += np.abs(g320[i] - g320[i - 1])
    return acc[int(_L2_H * 0.35) :, :]


def _l2_left_fraction(e):
    left, right = float(e[:, : _L2_W // 2].sum()), float(e[:, _L2_W // 2 :].sum())
    return left / (left + right + 1e-6)


def _l2_side_evasion(g320, c):
    lo, hi = c - _L2_PRE_LO, c - _L2_PRE_HI
    pre_e = _l2_diff_energy(g320, lo, hi)
    side = "LEFT" if _l2_left_fraction(pre_e) > _l2_left_fraction(_l2_diff_energy(g320, 0, len(g320))) else "RIGHT"
    col = pre_e.sum(0)
    center = float(col[int(_L2_W * 0.3) : int(_L2_W * 0.7)].sum() / (col.sum() + 1e-6))
    return side, (0 if center > 0.8 else 1)


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(model_dir / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True))
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    mean, std = _L2_IMAGENET_MEAN.to(device), _L2_IMAGENET_STD.to(device)
    folds = []
    use_app = True
    for pth in sorted(model_dir.glob("fold*.pt")):
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        m = _L2Localizer(ck["config"]["in_dim"], ck["config"]["hidden"], ck["config"]["dropout"])
        m.load_state_dict(ck["model"])
        folds.append(m.to(device).eval())
        use_app = ck["config"]["in_dim"] > 19  # 19 = 전역이동 6 + 광류 13 (EXP-S2-LEARN-001 no-app 변형이 최적: OOF 0.771)
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
                rgb, g320, g160 = _l2_load(paths)
                n = len(rgb)
                motion = np.zeros((n, 13), np.float32)
                for i in range(n - 1):
                    motion[i] = _l2_flow_features(g160[i], g160[i + 1], rx, ry)
                if n > 1:
                    motion[n - 1] = motion[n - 2]
                parts = [_l2_signals(g320), motion]
                if use_app:
                    feats = []
                    for s in range(0, n, 64):
                        x = torch.from_numpy(rgb[s : s + 64]).to(device).permute(0, 3, 1, 2).float() / 255.0
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                            f = backbone((x - mean) / std)
                        feats.append(f.float().cpu().numpy().astype(np.float16).astype(np.float32))
                    parts.append(np.concatenate(feats))
                seq = torch.from_numpy(np.concatenate(parts, 1))[None].to(device)
                score = np.mean([m(seq)[0].float().cpu().numpy() for m in folds], 0) if folds else -_l2_robust_z(motion[:, 12])
                c = int(np.argmax(score))
                side, evasion = _l2_side_evasion(g320, c)
            except Exception:
                c, side, evasion = len(paths) - 1, "LEFT", 1
            e = max(c - _L2_ENTRY_OFFSET, 0)
            rows.append({"ID": folder.name, "collision_frame": int(numbers[c]), "entry_frame": int(numbers[e]), "evasion_space": int(evasion), "entry_side": side})
    del backbone, folds
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
