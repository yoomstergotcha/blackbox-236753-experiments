"""comma2k19 세그먼트의 영상 프레임 timestamp와 CAN 신호(speed, steering_angle)를
실제 timestamp 기준으로 동기화한다.

파일 배치 (commaai/comma2k19 공식 저장소 + devkit notebook 기준, 2026-09-13 실측 확인
— data/external/comma2k19_example/README.md 참고):

    <segment>/video.hevc
    <segment>/processed_log/CAN/speed/t              (device-boot 초, 1-D, ~4974 샘플/60초, 불균일 ~83Hz)
    <segment>/processed_log/CAN/speed/value           (m/s) — 실측 결과 (N, 1) 2-D로 저장되어 있어 반드시 squeeze
    <segment>/processed_log/CAN/steering_angle/t      (device-boot 초, 1-D, speed/t와는 다른 배열)
    <segment>/processed_log/CAN/steering_angle/value  (degree, 1-D)
    <segment>/global_pose/frame_times                 (device-boot 초, 1-D, 영상 프레임 개수와 동일, ~20fps)

실측으로 확인된 점:
- speed/t와 steering_angle/t는 값이 서로 정확히 일치하지 않는 별개의 CAN 메시지 스트림이다.
  인덱스로 zip하면 두 신호가 어긋난다 — 반드시 timestamp 기준 선형보간으로 동기화해야 한다.
- CAN 샘플링은 고정 주기가 아니다(4974개/~60초 → 평균 83Hz이지만 간격이 불균일).
- CAN t와 frame_times는 모두 "device boot time(초)" 기준 같은 클럭을 공유한다(devkit notebook이
  둘을 같은 x축에 그리는 것으로 확인, 실측 값 범위도 서로 겹침).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _prepare_signal(t: np.ndarray, value: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    """(t, value)를 정렬된 (t, value) 1-D 배열로 강제한다.

    value가 (N, 1) 형태로 저장된 경우(실측: CAN/speed/value)가 있어 항상 reshape(-1)로
    강제 1-D화한다.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(t) != len(value):
        raise ValueError(f"{label}: t와 value 길이가 다름 ({len(t)} vs {len(value)})")
    order = np.argsort(t, kind="stable")
    return t[order], value[order]


def _load_signal(signal_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """{signal_dir}/t, {signal_dir}/value 파일을 읽어 _prepare_signal로 정리."""
    t = np.load(signal_dir / "t")
    value = np.load(signal_dir / "value")
    return _prepare_signal(t, value, str(signal_dir))


@dataclass
class Comma2k19Segment:
    segment_dir: Path | str
    speed_t: np.ndarray
    speed_v: np.ndarray
    steer_t: np.ndarray
    steer_v: np.ndarray
    frame_times: np.ndarray

    @classmethod
    def load(cls, segment_dir: Path) -> "Comma2k19Segment":
        segment_dir = Path(segment_dir)
        speed_t, speed_v = _load_signal(segment_dir / "processed_log" / "CAN" / "speed")
        steer_t, steer_v = _load_signal(segment_dir / "processed_log" / "CAN" / "steering_angle")
        frame_times = np.load(segment_dir / "global_pose" / "frame_times").astype(np.float64).reshape(-1)
        return cls(segment_dir, speed_t, speed_v, steer_t, steer_v, np.sort(frame_times))

    @classmethod
    def from_arrays(
        cls,
        name: str,
        speed_t: np.ndarray,
        speed_v: np.ndarray,
        steer_t: np.ndarray,
        steer_v: np.ndarray,
        frame_times: np.ndarray,
    ) -> "Comma2k19Segment":
        """디스크 파일이 아니라 이미 메모리에 있는 배열(예: HuggingFace parquet mirror의
        `log` struct)로부터 세그먼트를 구성한다. `.load()`와 동일한 정리 로직(정렬,
        (N,1) squeeze)을 거친다 — 로직 중복을 피하기 위해 `_prepare_signal`을 공유한다.
        """
        speed_t, speed_v = _prepare_signal(speed_t, speed_v, f"{name}/speed")
        steer_t, steer_v = _prepare_signal(steer_t, steer_v, f"{name}/steering_angle")
        frame_times = np.sort(np.asarray(frame_times, dtype=np.float64).reshape(-1))
        return cls(name, speed_t, speed_v, steer_t, steer_v, frame_times)

    def overlap_window(self) -> tuple[float, float]:
        """CAN 두 신호와 영상 프레임이 모두 존재하는 공통 시간 구간(device-boot 초)."""
        t0 = max(self.speed_t[0], self.steer_t[0], self.frame_times[0])
        t1 = min(self.speed_t[-1], self.steer_t[-1], self.frame_times[-1])
        if t1 <= t0:
            raise ValueError(f"{self.segment_dir}: CAN/frame_times 공통 구간이 없음")
        return t0, t1


def build_10hz_grid(segment: Comma2k19Segment, hz: float = 10.0) -> np.ndarray:
    """공통 구간 안에서 0.1초(10Hz) 간격의 timestamp 그리드를 만든다.

    grid[i]는 COMPETITION_GUIDE.md §7의 sample_index=i, time_seconds=i/hz에 대응하도록
    설계했다 (비공개 평가에서는 decoded frame 수 == sample_index 수라는 규칙과 동일한 해석).
    """
    t0, t1 = segment.overlap_window()
    n = int(np.floor((t1 - t0) * hz)) + 1
    return t0 + np.arange(n) / hz


def nearest_frame_indices(segment: Comma2k19Segment, grid: np.ndarray) -> np.ndarray:
    """그리드의 각 timestamp에 가장 가까운 video.hevc 프레임 인덱스 (오버레이 검증용)."""
    idx = np.searchsorted(segment.frame_times, grid)
    idx = np.clip(idx, 1, len(segment.frame_times) - 1)
    left = segment.frame_times[idx - 1]
    right = segment.frame_times[idx]
    use_left = (grid - left) <= (right - grid)
    return np.where(use_left, idx - 1, idx).astype(np.int64)


def interpolate(segment: Comma2k19Segment, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """grid 위에서 timestamp 기준 선형보간된 (speed_m_s, steering_angle_deg)."""
    speed = np.interp(grid, segment.speed_t, segment.speed_v)
    steer = np.interp(grid, segment.steer_t, segment.steer_v)
    return speed, steer
