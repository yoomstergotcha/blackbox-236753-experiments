"""CAN 신호(speed, steering_angle)를 COMPETITION_GUIDE.md Stage 3 스키마의
accel_label/steer_label로 변환한다.

파라미터는 코드에 박아두지 않고 configs/stage3/comma2k19_label.yaml에서 읽는다
(EXPERIMENT_DESIGN.md EXP-S3-LABEL-001 — "라벨 파라미터는 config로 분리").

steer_label 부호(steer_sign)는 2026-09-14 motion-signal 실측(64세그먼트/21route) +
사용자 승인으로 확정됨 — 아래 steer_sign 코멘트와 EXPERIMENT_DESIGN.md §9~11 참고.
근거를 갱신하지 않고 이 값만 바꾸지 말 것.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ACCEL_LABELS = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_LABELS = ("LEFT", "STRAIGHT", "RIGHT")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "stage3" / "comma2k19_label.yaml"

# configs/stage3/comma2k19_label.yaml이 없을 때만 쓰이는 하드코딩 fallback.
# 값은 config 파일의 초기값과 반드시 동일하게 유지한다.
_FALLBACK = {
    "stopped_speed_threshold_mps": 0.5,
    "accel_positive_threshold_mps2": 0.3,
    "accel_negative_threshold_mps2": -0.3,
    "steering_deadzone_deg": 3.0,
    # VERIFIED 2026-09-14 (motion-signal, human-approved — EXPERIMENT_DESIGN.md §9/§10/§11):
    # comma2k19 21개 route/64개 세그먼트 전체에서 steering_angle과 IMU gyro_z(요레이트)의
    # Pearson corr 평균 -0.907(범위 -0.997~-0.358), 예외 없이 64/64 전부 음의 상관.
    # 뚜렷한 구간(|steer|>1deg,|gyro_z|>1deg/s) 96,036개 중 반대 부호 100%.
    # FRD(forward-right-down) 좌표계에서 각속도 공식 d(forward)/dt = omega x forward로
    # 유도하면 +gyro_z는 forward축이 +right로 회전하는 것 = 우회전이다.
    # steering_angle이 gyro_z와 항상 반대 부호이므로 steering_angle > 0 은 좌회전(LEFT).
    # => steer_sign=+1이 맞는 방향이다.
    # 이후 comma2k19 raw_data/Chunk_1.zip에서 b0c9d2329ad1606b|2018-08-17--14-55-39/2
    # segment(9~45도 sustained steering, 주행속도 3~9m/s인 실제 회전 구간, t=1.2~7.9s)의
    # 실제 영상으로 육안 확인도 완료함(2026-09-14) — steering_angle이 전부 음수(RIGHT)인
    # 구간에서 "FWY" 램프(Ford 대리점 간판)를 실제로 우회전해 완전히 다른 도로(다른 건물,
    # 다른 신호등 배치)로 진입하는 장면을 프레임으로 직접 확인함
    # (output/turn_check/overlay/sample_00012~00075.jpg). motion-signal(64세그먼트)과
    # 영상 육안 확인이 둘 다 같은 결론을 지지 — AUTONOMOUS_RESEARCH_AGENT.md §8.2 기준
    # 완전히 충족됨.
    "steer_sign": 1,
    "speed_smoothing_window": 1,
    "steering_smoothing_window": 1,
}


@dataclass(frozen=True)
class LabelConfig:
    stopped_speed_threshold_mps: float
    accel_positive_threshold_mps2: float
    accel_negative_threshold_mps2: float
    steering_deadzone_deg: float
    steer_sign: int
    speed_smoothing_window: int
    steering_smoothing_window: int

    @classmethod
    def load(cls, path: Path | None = None) -> "LabelConfig":
        path = path or DEFAULT_CONFIG_PATH
        data = dict(_FALLBACK)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            data.update(loaded)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


DEFAULT_CONFIG = LabelConfig.load()
# 하위호환용 모듈 레벨 상수(기존 코드/노트북이 직접 참조할 수 있어 유지).
STOPPED_SPEED_MPS = DEFAULT_CONFIG.stopped_speed_threshold_mps
ACCEL_THRESHOLD_MPS2 = DEFAULT_CONFIG.accel_positive_threshold_mps2
STEER_STRAIGHT_DEG = DEFAULT_CONFIG.steering_deadzone_deg
STEER_SIGN = DEFAULT_CONFIG.steer_sign


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """window<=1이면 원본을 그대로 반환한다(smoothing 없음, 기본 동작)."""
    window = int(window)
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def accel_from_speed(grid: np.ndarray, speed: np.ndarray, config: LabelConfig = DEFAULT_CONFIG) -> np.ndarray:
    """(옵션) smoothing 후 중심차분으로 가속도(m/s^2)를 추정한다. 양 끝점은 전/후진 차분."""
    if len(speed) < 2:
        raise ValueError("가속도를 계산하려면 최소 2개 샘플이 필요합니다")
    smoothed = moving_average(speed, config.speed_smoothing_window)
    accel = np.empty_like(smoothed)
    accel[1:-1] = (smoothed[2:] - smoothed[:-2]) / (grid[2:] - grid[:-2])
    accel[0] = (smoothed[1] - smoothed[0]) / (grid[1] - grid[0])
    accel[-1] = (smoothed[-1] - smoothed[-2]) / (grid[-1] - grid[-2])
    return accel


def accel_label(speed: np.ndarray, accel: np.ndarray, config: LabelConfig = DEFAULT_CONFIG) -> list[str]:
    labels = []
    for v, a in zip(speed, accel):
        if v < config.stopped_speed_threshold_mps:
            labels.append("STOPPED")
        elif a > config.accel_positive_threshold_mps2:
            labels.append("ACCELERATING")
        elif a < config.accel_negative_threshold_mps2:
            labels.append("DECELERATING")
        else:
            labels.append("CONSTANT")
    return labels


def steer_label(steering_angle_deg: np.ndarray, config: LabelConfig = DEFAULT_CONFIG) -> list[str]:
    smoothed = moving_average(steering_angle_deg, config.steering_smoothing_window)
    labels = []
    for angle in smoothed:
        effective = angle * config.steer_sign
        if effective > config.steering_deadzone_deg:
            labels.append("LEFT")
        elif effective < -config.steering_deadzone_deg:
            labels.append("RIGHT")
        else:
            labels.append("STRAIGHT")
    return labels
