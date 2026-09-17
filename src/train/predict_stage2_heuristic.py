"""공식 inference.py에 붙여넣을 Stage2 predict 함수 — EXP-S2-HEUR-001 (학습 없는 물리 휴리스틱).

베이스라인 Stage2(공개 5클립으로 학습한 GRU)는 사실상 랜덤(LB 0.131). 비공개 Stage2에는 라벨 데이터가 없고
공개 라벨도 t_collision 5개뿐이므로, 학습 대신 충돌의 물리 신호를 쓴다:
  - collision_frame: 접촉 순간 카메라가 위아래로 튄다(피칭). 프레임 간 phase-correlation 수직 이동량의 robust z-score
    (영상 내부 중앙값/MAD — 파일 간 통계 없음)에서 최대 jolt 구간의 시작 프레임. 공개 5클립 5/5 (±0.3s).
  - entry_frame: 피해차량이 차선에 진입한 뒤 0.3~0.6s 안에 접촉하는 경우가 많아(공개 5클립 판독) collision - 5프레임.
  - entry_side: 충돌 직전 0.8s 동안 도로 영역(하단 65%) 프레임 변화 에너지의 가로 무게중심이 화면 중앙의 왼쪽이면 LEFT.
  - evasion_space: 판단 근거가 없어 1(베이스라인과 동일한 상수).
프레임 번호는 파일명의 숫자(가이드 §3.3)를 그대로 제출한다. torch 불필요(CPU만).
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_S2_W, _S2_H = 320, 180
_S2_TAU = 5.0  # jolt 판정 z-score
_S2_GAP = 5  # 최대 jolt에서 뒤로 탐색할 때 허용하는 비-jolt 프레임 수
_S2_ENTRY_OFFSET = 5  # 프레임 (10fps 가정 0.5s)
_S2_SIDE_WINDOW = 8  # 프레임


def _s2_frame_number(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def _s2_load_gray(paths):
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((_S2_H, _S2_W), np.uint8)
        out.append(cv2.resize(img, (_S2_W, _S2_H), interpolation=cv2.INTER_AREA).astype(np.float32))
    return out


def _s2_robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def _s2_signals(gray):
    n = len(gray)
    win = cv2.createHanningWindow((_S2_W, _S2_H), cv2.CV_32F)
    shift, vert, diff = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        (dx, dy), _ = cv2.phaseCorrelate(gray[i - 1], gray[i], win)
        shift[i], vert[i] = float(np.hypot(dx, dy)), float(abs(dy))
        diff[i] = float(np.abs(gray[i] - gray[i - 1]).mean())
    return shift, vert, diff


def _s2_collision_index(zs, zv, zd) -> int:
    """최대 수직 jolt에 앵커한 뒤, 앞쪽으로 GAP 프레임 안에 또 다른 jolt(z>TAU)가 있으면 그쪽으로 옮겨 가며 시작점을 찾는다."""
    n = len(zv)
    if n < 3:
        return n - 1
    if zv.max() < _S2_TAU:  # 뚜렷한 수직 jolt가 없으면 세 신호의 최대 스파이크
        return int(np.argmax(np.maximum.reduce([zs, zv, zd])))
    i = int(np.argmax(zv))
    while True:
        lo = max(i - _S2_GAP, 1)
        prev = [j for j in range(lo, i) if zv[j] > _S2_TAU]
        if not prev:
            return i
        i = prev[0]


def _s2_entry_side(gray, c: int) -> str:
    lo = max(c - _S2_SIDE_WINDOW, 1)
    acc = np.zeros((_S2_H, _S2_W), np.float32)
    for i in range(lo, max(c, lo) + 1):
        if i < len(gray):
            acc += np.abs(gray[i] - gray[i - 1])
    road = acc[int(_S2_H * 0.35) :, :]
    col = road.sum(0)
    total = float(col.sum())
    if total <= 0:
        return "LEFT"
    cx = float((col * np.arange(_S2_W)).sum() / total)
    return "LEFT" if cx < _S2_W / 2 else "RIGHT"


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir) / "images"
    folders = sorted(p for p in image_root.iterdir() if p.is_dir())
    rows = []
    for folder in folders:
        paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=_s2_frame_number)
        if not paths:
            continue
        numbers = [_s2_frame_number(p) for p in paths]
        try:
            gray = _s2_load_gray(paths)
            shift, vert, diff = _s2_signals(gray)
            zs, zv, zd = _s2_robust_z(shift), _s2_robust_z(vert), _s2_robust_z(diff)
            c = _s2_collision_index(zs, zv, zd)
            side = _s2_entry_side(gray, c)
        except Exception:
            c, side = len(paths) - 1, "LEFT"
        e = max(c - _S2_ENTRY_OFFSET, 0)
        rows.append({"ID": folder.name, "collision_frame": int(numbers[c]), "entry_frame": int(numbers[e]), "evasion_space": 1, "entry_side": side})
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
