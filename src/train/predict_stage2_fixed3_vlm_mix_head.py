"""공식 inference.py에 붙여넣을 Stage2 predict 함수 — 프레임률 적응 학습 localizer (EXP-S2-FPS-001/INV-001) + VLM 진입방향/회피공간 (EXP-S2-VLM-001).

model/stage2/qwen2vl/ (Qwen2-VL-2B-Instruct, Apache-2.0)가 있으면 진입방향·회피공간을 zero-shot log-prob 비교(고정 임계)로 판정, 없거나 실패하면 규칙 폴백.
entry = collision − 21 고정(LB 실측 v23: −21이 −7보다 +0.038).

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
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

_L2_W, _L2_H = 320, 180
_L2_ENTRY_OFFSET = 21
_L2_TAIL_EXCLUDE = 3
_L2_NEW_WEIGHT = 0.5  # 'n*' 체크포인트(ego+non-ego 학습) 그룹 가중치; CCD ego 0.793 / non-ego 0.421 (배포 11: 0.788 / 0.282)
_L2_STRIDES = (3,)  # CCD 검증 범위; 4·6은 허위 최대값 위험
_L2_MIN_FRAMES = 30  # stride 3는 원본 90프레임 이상일 때만(LB v30 S2 0.340 확정); 짧은 클립은 stride 1
_L2_STRIDE_RATIO = 1.5  # stride 1 기본; 다른 stride는 최대 점수가 stride1의 1.5배를 넘을 때만 채택(CCD 200클립: 10fps 0.79 / 합성 30fps 0.715)
_L2_PRE_LO, _L2_PRE_HI = 12, 2
_L2_VLM_BUDGET_S = 1500  # Stage2 시작 후 이 시간(초)을 넘기면 남은 클립은 VLM 생략(규칙 폴백) — 60분 한도 보호
_L2_VLM_DIR = "qwen2vl"
_L2_VLM_NF, _L2_VLM_W, _L2_VLM_SPAN = 8, 448, 14
_L2_VLM_SIDE_THR = 1.375  # 수동 라벨 53클립 점수 중앙값(LOO acc 0.66); 모델이 LEFT로 치우쳐 0이 아닌 고정 상수 사용
_L2_EVA_M1, _L2_EVA_S1, _L2_EVA_M2, _L2_EVA_S2, _L2_EVA_THR = -1.1012, 0.1585, 0.0337, 0.1171, -0.5  # z-정규화 상수(63클립) / 임계 −0.5는 라벨 563클립 재보정(macroF1 0.577→0.619, pred1 0.57 vs 라벨 0.59)
_L2_Q_SIDE = "This is a dashcam video from the ego car, ending at the moment it collides with another vehicle. From which side of the screen did that other vehicle come into the ego car's path? Answer with exactly one word: LEFT or RIGHT."
_L2_Q_EVA_IMG = "This dashcam image shows the moment the camera car collides with another vehicle. Is there an open lane or free road space right next to the camera car where it could have steered to avoid the crash? Answer with exactly one word: YES or NO."
_L2_Q_EVA_VID = "This is a dashcam video ending at a collision. Just before the collision, was there empty road space to the left or right of the camera car (no other vehicle, wall, barrier or curb blocking it)? Answer with exactly one word: YES or NO."


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


def _l2_vlm_load(model_dir, device):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    d = Path(model_dir) / _L2_VLM_DIR
    if not d.is_dir():
        return None
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    last = None
    for _ in range(3):  # 일시적 I/O·메모리 오류 대비 재시도
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(d, dtype=dtype).to(device).eval()
            return model, AutoProcessor.from_pretrained(d)
        except Exception as exc:  # noqa: BLE001
            last = exc
            torch.cuda.empty_cache()
            time.sleep(5)
    raise last


def _l2_vlm_frame(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = np.zeros((720, 1280, 3), np.uint8)
    h, w = bgr.shape[:2]
    s = _L2_VLM_W / w
    return cv2.cvtColor(cv2.resize(bgr, (_L2_VLM_W, max(28, int(round(h * s / 28)) * 28))), cv2.COLOR_BGR2RGB)


def _l2_vlm_frames(paths, c, s=1):
    idx = np.linspace(max(c - _L2_VLM_SPAN * s, 0), min(c + s, len(paths) - 1), _L2_VLM_NF).round().astype(int)
    return np.stack([_l2_vlm_frame(paths[i]) for i in idx])


def _l2_vlm_logprobs(vlm, device, media, question, cands, kind="video"):
    model, proc = vlm
    msgs = [{"role": "user", "content": [{"type": kind}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {"videos": [media]} if kind == "video" else {"images": [media]}
    n_prompt = proc(text=[text], return_tensors="pt", **kw)["input_ids"].shape[1]
    out = []
    for cand in cands:
        full = proc(text=[text + cand], return_tensors="pt", **kw).to(device)
        logits = model(**full).logits[0].float()
        ids = full["input_ids"][0]
        out.append(torch.log_softmax(logits[n_prompt - 1 : -1], -1).gather(1, ids[n_prompt:, None]).sum().item())
    return out


def _l2_vlm_side_evasion(vlm, device, paths, c, s=1):
    """VLM 로짓 3종(side LEFT−RIGHT, evasion 이미지/비디오 YES−NO) 반환."""
    video = _l2_vlm_frames(paths, c, s)
    ls = _l2_vlm_logprobs(vlm, device, video, _L2_Q_SIDE, ["LEFT", "RIGHT"])
    l1 = _l2_vlm_logprobs(vlm, device, _l2_vlm_frame(paths[min(c, len(paths) - 1)]), _L2_Q_EVA_IMG, ["YES", "NO"], kind="image")
    l2 = _l2_vlm_logprobs(vlm, device, video, _L2_Q_EVA_VID, ["YES", "NO"])
    return ls[0] - ls[1], l1[0] - l1[1], l2[0] - l2[1]


def _l2_rule_from_logits(side_score, eva_img, eva_vid):
    eva = (eva_img - _L2_EVA_M1) / _L2_EVA_S1 + (eva_vid - _L2_EVA_M2) / _L2_EVA_S2
    return ("LEFT" if side_score > _L2_VLM_SIDE_THR else "RIGHT"), (1 if eva > _L2_EVA_THR else 0)


_L2_LR = None  # EXP-S2-SIDEEVA-001: 수동 라벨(CCD, 253클립) 로지스틱 회귀 상수; None이면 임계 규칙
_L2_YOLO_FILE = "yolov8s.pt"  # Ultralytics YOLOv8s (AGPL-3.0), 트랙 cue용
_L2_TRACK_COLS = {"lat", "has", "occ", "nb", "area"}
_L2_SIDE_TRACK_GATE = 0.1  # 횡위치 |x−0.5|가 이보다 작으면 VLM으로 폴백 (라벨 431: 게이트 0.06 0.629 / 0.1 0.639)


def _l2_yolo_load(model_dir):
    p = Path(model_dir) / _L2_YOLO_FILE
    if not p.is_file():
        return None
    from ultralytics import YOLO

    return YOLO(str(p))


def _l2_track_cues(yolo, paths, c, s=1):
    """[c−8s, c] 구간(stride s)에서 차량 박스 → greedy IoU 트랙 → 피해차량(최대 면적) 횡위치·자유공간·박스 수·면적. 실패 시 has=0."""
    cues = {"lat": 0.0, "has": 0.0, "occ": 0.0, "nb": 0.0, "area": 0.0}
    if yolo is None:
        return cues
    idx = list(range(max(c - 8 * s, 0), min(c + 1, len(paths)), max(s, 1)))
    if len(idx) < 2:
        return cues
    per = []
    for i in idx:
        bgr = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
        boxes = []
        if bgr is not None:
            h, w = bgr.shape[:2]
            res = yolo.predict(bgr, imgsz=640, conf=0.3, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
            for b, cf, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if int(cl) in (2, 3, 5, 7):
                    boxes.append(((b[0] / w, b[1] / h, b[2] / w, b[3] / h), float(cf)))
        per.append(boxes)

    def iou(a, b):
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    tracks, active = [], []
    for f, bs in enumerate(per):
        used, new_active = set(), []
        for ti, last in active:
            best, bj = 0.3, -1
            for j, (b, cf) in enumerate(bs):
                if j in used:
                    continue
                v = iou(tracks[ti][last][0], b)
                if v > best:
                    best, bj = v, j
            if bj >= 0:
                tracks[ti][f] = bs[bj]
                used.add(bj)
                new_active.append((ti, f))
            else:
                new_active.append((ti, last))
        for j, (b, cf) in enumerate(bs):
            if j not in used:
                tracks.append({f: (b, cf)})
                new_active.append((len(tracks) - 1, f))
        active = [(ti, last) for ti, last in new_active if f - last <= 2]
    tracks = [t for t in tracks if len(t) >= 2]
    cf_ = len(per) - 1  # 충돌 프레임 인덱스(창 내)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    alive = [(tr, max(g for g in tr if g <= cf_)) for tr in tracks if any(cf_ - 5 <= g <= cf_ for g in tr)]
    if alive:
        vic = max(alive, key=lambda x: area(x[0][x[1]][0]))[0]
        xs = [(vic[g][0][0] + vic[g][0][2]) / 2 for g in vic if cf_ - 8 <= g < cf_]
        if xs:
            cues["lat"], cues["has"] = float(np.mean(xs) - 0.5), 1.0
    else:
        vic = None
    f0 = max(cf_ - 2, 0)
    occ = [0.0, 0.0]
    for tr in tracks:
        if tr is vic or f0 not in tr:
            continue
        b = tr[f0][0]
        if b[3] < 0.6:
            continue
        occ[0 if (b[0] + b[2]) / 2 < 0.5 else 1] += min(b[2], 1) - max(b[0], 0)
    cues["occ"] = float(min(occ))
    cues["nb"] = float(sum(1 for tr in tracks if cf_ in tr))
    cues["area"] = float(max((area(tr[cf_][0]) for tr in tracks if cf_ in tr), default=0.0))
    return cues


def _l2_lr_apply(spec, feats):
    z = spec["b"]
    for col, mu, sd, w in zip(spec["cols"], spec["mu"], spec["sd"], spec["w"]):
        z += w * (feats[col] - mu) / sd
    return 1.0 / (1.0 + np.exp(-z))


def _l2_head_side_evasion(vlm, device, paths, c, s, yolo):
    """side: 피해차량 트랙 횡위치 부호(|lat| > gate, EXP-S2-SIDEEVA-001: 라벨 214클립 macroF1 0.545→0.660), 트랙 없으면 VLM 임계. evasion: VLM z-합 규칙."""
    side_score, eva_img, eva_vid = _l2_vlm_side_evasion(vlm, device, paths, c, s)
    side, evasion = _l2_rule_from_logits(side_score, eva_img, eva_vid)
    try:
        cues = _l2_track_cues(yolo, paths, c, s)
        if cues["has"] > 0 and abs(cues["lat"]) > _L2_SIDE_TRACK_GATE:
            side = "LEFT" if cues["lat"] < 0 else "RIGHT"
    except Exception:
        pass
    return side, evasion


def _l2_localize(g320, g160, folds, rx, ry, device):
    """stride 3 고정(원본 90프레임 미만이면 stride 1). LB v30: S2 0.340 (적응 규칙 0.283). (collision index, stride)"""
    n = len(g320)

    def run(s):
        x = np.concatenate([_l2_signals(g320[::s]), _l2_motion(g160[::s], rx, ry)], 1)
        seq = torch.from_numpy(x)[None].to(device)
        sc = [(new, torch.sigmoid(m(seq)[0]).float().cpu().numpy()) for new, m in folds]
        a = [v for new, v in sc if not new]
        b = [v for new, v in sc if new]
        if a and b:  # 그룹 가중 평균: ego 전용 11모델 vs ego+non-ego 3-seed (EXP-S2-ALL-001, w=0.5)
            return (1.0 - _L2_NEW_WEIGHT) * np.mean(a, 0) + _L2_NEW_WEIGHT * np.mean(b, 0)
        return np.mean([v for _, v in sc], 0)

    s = _L2_STRIDES[0] if n // _L2_STRIDES[0] >= _L2_MIN_FRAMES else 1
    score = run(s)
    c = int(np.argmax(score[: max(len(score) - _L2_TAIL_EXCLUDE, 1)])) * s
    return min(c, n - 1), s


def predict_stage2(data_dir, model_dir):
    device = _device()
    folds = []
    for pth in sorted(Path(model_dir).glob("*fold*.pt")):
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        m = _l2_build(ck["config"])
        m.load_state_dict(ck["model"])
        folds.append((pth.name.startswith("n"), m.to(device).eval()))
    try:
        vlm = _l2_vlm_load(model_dir, device)
    except Exception:
        vlm = None
    try:
        yolo = _l2_yolo_load(model_dir)
    except Exception:
        yolo = None
    rx, ry = _l2_radial_grid(120, 160)
    image_root = Path(data_dir) / "images"
    rows = []
    t_start = time.time()
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
            if vlm is not None and time.time() - t_start < _L2_VLM_BUDGET_S:
                try:
                    side, evasion = _l2_head_side_evasion(vlm, device, paths, c, s, yolo)
                except Exception:
                    pass
            e = max(c - _L2_ENTRY_OFFSET, 0)
            rows.append({"ID": folder.name, "collision_frame": int(numbers[c]), "entry_frame": int(numbers[e]), "evasion_space": int(evasion), "entry_side": side})
    del folds, vlm, yolo
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
