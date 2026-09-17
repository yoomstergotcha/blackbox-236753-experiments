"""화면 재촬영(re-recording) 시뮬레이션 — Stage1 학습용 RERECORDED 클래스 합성.

COMPETITION_GUIDE.md §2.3에 열거된 재녹화 특징을 무작위 조합·강도로 적용한다:
화면 테두리(베젤), 디스플레이 주사 패턴(모아레/스캔라인), 반사광, 밝기·색상 변화,
추가 압축·화질 저하, 재촬영 기기의 움직임(프레임 간 흔들림), 원근 변화, 해상도 변화.

공식 공개 예제의 RERECORDED 5건(data/stage1/rerecorded)을 실측한 결과 프레이밍·해상도·fps는
원본과 동일하고 강한 재압축(블록 노이즈)만 들어가 있었다(EXPERIMENT_DESIGN.md §14).
그래서 재압축은 항상 적용하고 나머지는 확률적으로 적용해 "DACON식"과 "실제 재촬영식"을
모두 커버한다. 파라미터는 영상 단위로 한 번 뽑고(같은 재촬영 세팅), 프레임마다는
흔들림·노이즈·JPEG만 달라진다.
"""
from __future__ import annotations

import cv2
import numpy as np


def sample_base_params(rng: np.random.Generator) -> dict:
    """'블랙박스 품질' 기저 열화 — 양쪽 클래스에 공통으로 먼저 적용한다 (EXP-S1-SYNTH-002).

    EXP-S1-SYNTH-001에서 깨끗한 comma/OPEN 원본만 ORIGINAL로 학습했더니 모델이
    "깨끗함=원본, 열화=재녹화"를 배워 이미 저품질인 공식 원본 5건을 전부 RERECORDED로 판정했다.
    테스트 원본도 그런 저품질 블랙박스 영상이므로, 두 클래스 모두 이 기저(중간 품질 JPEG +
    확률적 저해상도/블러)에서 출발하게 해 모델이 그 위에 얹힌 *2차* 열화만 배우게 한다.
    base JPEG q(45~85)는 재녹화 q(8~40)보다 항상 높아 재압축 단서가 사라지지 않는다.
    """
    # v3 (EXP-S1-SYNTH-003): 공식 원본 5건은 이미 블록이 심한 유튜브 재인코딩 품질
    # (blockiness8 2.5~3.7). 원본 클래스가 그 수준이 되도록 base JPEG q를 낮게 잡는다.
    return {
        "base_jpeg_q": int(rng.integers(18, 51)),
        "base_downscale": float(rng.uniform(0.5, 1.0)) if rng.random() < 0.5 else 1.0,
        "base_blur_sigma": float(rng.uniform(0.2, 0.8)) if rng.random() < 0.3 else 0.0,
    }


def apply_base(rgb: np.ndarray, p: dict) -> np.ndarray:
    h, w = rgb.shape[:2]
    x = rgb
    if p["base_downscale"] < 0.98:
        nh, nw = max(16, int(h * p["base_downscale"])), max(16, int(w * p["base_downscale"]))
        x = cv2.resize(cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if p["base_blur_sigma"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["base_blur_sigma"])
    return _jpeg(x, p["base_jpeg_q"])


def sample_params(rng: np.random.Generator) -> dict:
    """v3 (EXP-S1-SYNTH-003) — 공식 5쌍 실측(EXPERIMENT_DESIGN.md §15)에 맞춘 '핵심' 흔적 +
    실제 재촬영용 '선택' 흔적.

    실측된 DACON식 변환: 정렬 이동/스케일 없음, 기존 블록은 부드러워짐(블러), 미세 그레인
    노이즈 추가(hf_ratio +15%, 라플라시안 분산 2배), 대비 소폭 증가(std +2~4), 밝기 소폭 감소
    (-1~-3), 가벼운 재인코딩. -> 핵심 흔적은 항상 적용. 이전 v1/v2의 강한 JPEG 블록·다운스케일은
    원본(이미 블록 심함)과 반대 방향이라 낮은 확률의 선택 항목으로 내렸다.
    """
    heavy = rng.random() < 0.2  # 재압축만 강하게 들어간 변형도 일부 커버
    return {
        "jpeg_q": int(rng.integers(12, 40)) if heavy else int(rng.integers(75, 96)),
        "downscale": float(rng.uniform(0.5, 0.95)) if rng.random() < 0.2 else 1.0,
        "blur_sigma": float(rng.uniform(0.1, 0.35)),
        # 재촬영은 화면 픽셀 격자를 카메라 센서 격자로 재샘플링한다 -> 8px 블록 경계가 뭉개진다
        # (실측: blockiness8 -25%, 완전히 사라지진 않음). 확률 0.35로 미세 스케일+서브픽셀 이동.
        "resample_scale": float(rng.uniform(0.99, 1.01)) if rng.random() < 0.35 else 1.0,
        "resample_shift": rng.uniform(-0.5, 0.5, size=2).astype(np.float32),
        # 그레인은 최종 인코딩 뒤에 얹는다(실측: 라플라시안 분산 약 2배, hf_ratio +15%)
        "noise_sigma": float(rng.uniform(3.0, 8.0)),
        "gamma": float(rng.uniform(0.95, 1.08)),
        "contrast": float(rng.uniform(1.0, 1.12)),
        "saturation": float(rng.uniform(0.9, 1.08)),
        "color_shift": rng.uniform(-5, 1, size=3).astype(np.float32),
        "moire": bool(rng.random() < 0.25),
        "moire_freq": float(rng.uniform(0.15, 0.6)),
        "moire_angle": float(rng.uniform(0, np.pi)),
        "moire_alpha": float(rng.uniform(0.03, 0.10)),
        "scanline": bool(rng.random() < 0.15),
        "scan_period": int(rng.integers(2, 5)),
        "scan_alpha": float(rng.uniform(0.05, 0.15)),
        "perspective": bool(rng.random() < 0.2),
        "persp_mag": float(rng.uniform(0.01, 0.04)),
        "bezel": bool(rng.random() < 0.15),
        "bezel_scale": float(rng.uniform(0.85, 0.97)),
        "bezel_color": int(rng.integers(0, 40)),
        "reflection": bool(rng.random() < 0.25),
        "refl_alpha": float(rng.uniform(0.05, 0.2)),
        "refl_center": rng.uniform(0.1, 0.9, size=2).astype(np.float32),
        "refl_radius": float(rng.uniform(0.15, 0.5)),
        "jitter_px": int(rng.integers(0, 3)) if rng.random() < 0.3 else 0,
    }


def _perspective(rgb: np.ndarray, mag: float, rng: np.random.Generator) -> np.ndarray:
    h, w = rgb.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx, dy = mag * w, mag * h
    dst = src + rng.uniform(-1, 1, size=(4, 2)).astype(np.float32) * np.float32([dx, dy])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgb, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _bezel(rgb: np.ndarray, scale: float, color: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    inner = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(rgb, color)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y : y + nh, x : x + nw] = inner
    return canvas


def _color(rgb: np.ndarray, p: dict) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), p["gamma"])
    x = (x - 0.5) * p["contrast"] + 0.5
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * p["saturation"]
    x = x * 255.0 + p["color_shift"][None, None, :]
    return np.clip(x, 0, 255).astype(np.uint8)


def _reflection(rgb: np.ndarray, p: dict) -> np.ndarray:
    h, w = rgb.shape[:2]
    cy, cx = p["refl_center"][0] * h, p["refl_center"][1] * w
    yy, xx = np.mgrid[0:h, 0:w]
    r = p["refl_radius"] * max(h, w)
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))).astype(np.float32)
    x = rgb.astype(np.float32) + (p["refl_alpha"] * 255.0) * blob[:, :, None]
    return np.clip(x, 0, 255).astype(np.uint8)


def _moire(rgb: np.ndarray, p: dict) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * np.cos(p["moire_angle"]) + yy * np.sin(p["moire_angle"])
    pattern = 0.5 * (1.0 + np.sin(2.0 * np.pi * p["moire_freq"] * u))
    mod = (1.0 - p["moire_alpha"]) + p["moire_alpha"] * pattern
    return np.clip(rgb.astype(np.float32) * mod[:, :, None], 0, 255).astype(np.uint8)


def _scanline(rgb: np.ndarray, p: dict) -> np.ndarray:
    x = rgb.astype(np.float32)
    x[:: p["scan_period"]] *= 1.0 - p["scan_alpha"]
    return np.clip(x, 0, 255).astype(np.uint8)


def _jpeg(rgb: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return rgb
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def apply_rerecord(rgb: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """rgb: (H, W, 3) uint8. 영상 단위 파라미터 p는 sample_params()로 뽑고, 프레임마다
    흔들림·노이즈·JPEG 디테일만 rng로 달라진다."""
    h, w = rgb.shape[:2]
    x = rgb
    if p["perspective"]:
        x = _perspective(x, p["persp_mag"], rng)
    if p["bezel"]:
        x = _bezel(x, p["bezel_scale"], p["bezel_color"])
    x = _color(x, p)
    if p["reflection"]:
        x = _reflection(x, p)
    if p["moire"]:
        x = _moire(x, p)
    if p["scanline"]:
        x = _scanline(x, p)
    if p["downscale"] < 0.98:
        nh, nw = max(16, int(h * p["downscale"])), max(16, int(w * p["downscale"]))
        x = cv2.resize(cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if p["blur_sigma"] > 0.2:
        x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"])
    # 센서 격자 재샘플링: 미세 스케일 + 서브픽셀 이동(+ 프레임별 흔들림)
    s = p.get("resample_scale", 1.0)
    sx, sy = p.get("resample_shift", (0.0, 0.0))
    jx = jy = 0.0
    if p["jitter_px"] > 0:
        jx, jy = rng.integers(-p["jitter_px"], p["jitter_px"] + 1, size=2)
    m = np.float32([[s, 0, (1 - s) * w / 2 + sx + jx], [0, s, (1 - s) * h / 2 + sy + jy]])
    x = cv2.warpAffine(x, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x = _jpeg(x, p["jpeg_q"])
    if p["noise_sigma"] > 0.5:
        # 재촬영 센서 그레인은 이후 코덱을 거치며 공간적으로 약간 뭉친다(백색 노이즈가 아님).
        # 백색 노이즈는 8px 블록 경계 대비(blockiness)를 과하게 지워 실측(-25%)과 어긋났다.
        noise = rng.normal(0, p["noise_sigma"], size=(h, w, 1)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.7)[:, :, None] * 1.8  # 블러로 줄어든 진폭 보정
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return x


def rerecord_frame(rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return apply_rerecord(rgb, sample_params(rng), rng)
