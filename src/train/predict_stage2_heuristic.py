"""공식 inference.py에 붙여넣을 Stage2 predict 함수 — EXP-S2-HEUR-001 (학습 없는 물리 휴리스틱).

베이스라인 Stage2(공개 5클립으로 학습한 GRU)는 사실상 랜덤(LB 0.131). 비공개 Stage2에는 라벨 데이터가 없고
공개 라벨도 t_collision 5개뿐이므로, 학습 대신 충돌의 물리 신호를 쓴다:
  - collision_frame: 접촉 순간 카메라가 위아래로 튄다(피칭). 프레임 간 phase-correlation 수직 이동량의 robust z-score
    (영상 내부 중앙값/MAD — 파일 간 통계 없음)에서 최대 jolt 구간의 시작 프레임. 공개 5클립 5/5 (±0.3s).
  - entry_frame: 피해차량이 차선에 진입한 뒤 0.3~0.6s 안에 접촉하는 경우가 많아(공개 5클립 판독) collision - 5프레임.
  - entry_side: 접근 구간 도로 영역(하단 65%) 변화 에너지의 좌측 비율이 영상 전체 기준선보다 크면 LEFT(상대값).
    공개 5클립 수동 판독과 2/4 일치 — 사실상 코인플립이지만 두 클래스를 모두 내므로 macro-F1상 상수 예측보다 기대값이 높다.
  - evasion_space: 접근 구간 변화 에너지가 화면 중앙에 집중(>0.8)이면 정면 추돌로 0, 아니면 1.
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
_S2_PRE_LO, _S2_PRE_HI = 12, 2  # 충돌 직전 접근 구간 [c-12, c-2): 접촉 프레임 자체의 전역 jolt는 제외 (±1프레임 이동에 안정)
# EXP-S2-HEUR-002 (CCD 사전정보): 공개 5클립 = CCD(Car Crash Dataset, MIT) 000001~000005이고 충돌 라벨 = CCD 첫 사고 프레임.
# CCD 1,500클립의 사고 onset은 50프레임 중 30~49(중앙값 36, 10퍼센타일 30) -> 검색을 클립 후반(0.55N 이후)으로 제한,
# jolt가 없으면 0.72N(=36). 위 상수들은 50프레임/10fps 기준이며 클립이 5초라는 가정 아래 프레임 수 N에 비례해 스케일한다.
_S2_SEARCH_FROM = 0.0  # LB 실측(제출 12: 0.250->0.175)으로 후반부 제한 폐기 -> 0.0 = 전체 검색(HEUR-001 동작)
_S2_FALLBACK = 0.72
_S2_REF_N = 50
_S2_SCALE_BY_N = False  # N 비례 스케일도 폐기(HEUR-001 고정 프레임 값)


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
    """클립 후반(>= SEARCH_FROM*N)에서 최대 수직 jolt에 앵커한 뒤, 앞쪽 GAP 프레임 안의 다른 jolt(z>TAU)로 옮겨 가며 시작점을 찾는다.
    jolt(z>TAU)가 후반에 없으면 세 신호의 후반 최대 스파이크, 그것도 약하면(z<3) CCD 중앙값 위치(FALLBACK*N)."""
    n = len(zv)
    if n < 3:
        return n - 1
    scale = (n / _S2_REF_N) if _S2_SCALE_BY_N else 1.0
    gap = max(1, int(round(_S2_GAP * scale)))
    start = min(int(_S2_SEARCH_FROM * n), n - 1)
    zv_late = zv[start:]
    if zv_late.max() < _S2_TAU:
        comb = np.maximum.reduce([zs, zv, zd])[start:]
        if comb.max() < 3.0:
            return min(int(round(_S2_FALLBACK * n)), n - 1)
        return start + int(np.argmax(comb))
    i = start + int(np.argmax(zv_late))
    while True:
        lo = max(i - gap, start, 1)
        prev = [j for j in range(lo, i) if zv[j] > _S2_TAU]
        if not prev:
            return i
        i = prev[0]


def _s2_diff_energy(gray, lo: int, hi: int) -> np.ndarray:
    """[lo, hi) 구간의 |frame_i - frame_{i-1}| 누적 맵(도로 영역 = 하단 65%)."""
    acc = np.zeros((_S2_H, _S2_W), np.float32)
    for i in range(max(lo, 1), min(hi, len(gray))):
        acc += np.abs(gray[i] - gray[i - 1])
    return acc[int(_S2_H * 0.35) :, :]


def _s2_left_fraction(energy: np.ndarray) -> float:
    left, right = float(energy[:, : _S2_W // 2].sum()), float(energy[:, _S2_W // 2 :].sum())
    return left / (left + right + 1e-6)


def _s2_pre_window(gray, c: int):
    scale = (len(gray) / _S2_REF_N) if _S2_SCALE_BY_N else 1.0
    return c - max(2, int(round(_S2_PRE_LO * scale))), c - max(1, int(round(_S2_PRE_HI * scale)))


def _s2_entry_side(gray, c: int) -> str:
    """충돌 직전 창의 좌측 에너지 비율이 영상 전체(에고 모션·도로 기하의 기준선)보다 크면 LEFT.
    절대값은 카메라 장착/도로 기하로 한쪽에 치우쳐(샘플 5개 전부 RIGHT) 영상 자체 기준선 대비 상대값을 쓴다."""
    pre = _s2_left_fraction(_s2_diff_energy(gray, *_s2_pre_window(gray, c)))
    base = _s2_left_fraction(_s2_diff_energy(gray, 0, len(gray)))
    return "LEFT" if pre > base else "RIGHT"


def _s2_evasion_space(gray, c: int) -> int:
    """충돌 직전 변화 에너지가 화면 중앙(가로 30~70%)에 집중되면 정면 추돌(앞차 급정거)로 보고 0, 측면 진입이면 1.
    공개 5클립(창 [c-12,c-2)): 추돌 클립 0.92~0.93 vs 측면 진입 0.52~0.71 -> 임계 0.8 (검출 c ±1프레임에 안정)."""
    energy = _s2_diff_energy(gray, *_s2_pre_window(gray, c))
    col = energy.sum(0)
    center = float(col[int(_S2_W * 0.3) : int(_S2_W * 0.7)].sum() / (col.sum() + 1e-6))
    return 0 if center > 0.8 else 1


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
            side, evasion = _s2_entry_side(gray, c), _s2_evasion_space(gray, c)
        except Exception:
            c, side, evasion = len(paths) - 1, "LEFT", 1
        scale = (len(paths) / _S2_REF_N) if _S2_SCALE_BY_N else 1.0
        e = max(c - max(1, int(round(_S2_ENTRY_OFFSET * scale))), 0)
        rows.append({"ID": folder.name, "collision_frame": int(numbers[c]), "entry_frame": int(numbers[e]), "evasion_space": int(evasion), "entry_side": side})
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])
