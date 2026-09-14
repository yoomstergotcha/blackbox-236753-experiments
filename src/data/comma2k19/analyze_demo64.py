"""HuggingFace `commaai/comma2k19` "demo" parquet split(64개 실제 세그먼트, MIT, 약 228MB —
raw_data/Chunk_*.zip 94.6GB 전체와는 별개의 경량 미러) 전체에 대해 EXP-S3-LABEL-002를 수행한다.

이 mirror는 세그먼트당 CAN/IMU/pose 신호 전체를 제공하지만 `preview`(첫 프레임 1장)만 있고
전체 영상은 없다 — 그래서 이 스크립트는 "많은 route에서 dense label + class distribution +
threshold sensitivity + STEER_SIGN의 대규모 motion-signal 교차검증"까지만 다루고, 영상 육안
확인(overlay)은 다루지 않는다 — 그 부분은 여전히 사람이 직접 회전이 있는 세그먼트의 영상을
봐야 한다(AUTONOMOUS_RESEARCH_AGENT.md §15의 Human Review Condition).

실행:
    python -m src.data.comma2k19.analyze_demo64 \
        --parquet-dir data/external/comma2k19_demo64 \
        --out output/comma2k19_demo64
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .labeling import LabelConfig, accel_from_speed, accel_label, steer_label
from .report import class_distribution_report
from .sync import Comma2k19Segment, build_10hz_grid, interpolate

# route id = 세그먼트 식별자에서 "<device-hash>|<datetime>" 부분(뒤의 "/<segment-번호>" 제외).
# EXPERIMENT_DESIGN.md §9-6의 route-level split 설계와 동일한 규칙.
def route_id(segment_id: str) -> str:
    return segment_id.rsplit("/", 1)[0]


def _stack(col) -> np.ndarray:
    """parquet에서 읽은 object-dtype 배열(각 원소가 리스트/배열)을 2-D ndarray로 변환."""
    return np.stack([np.asarray(v, dtype=np.float64) for v in col])


def load_segment(row: pd.Series) -> Comma2k19Segment:
    log = row["log"]
    speed_v = _stack(log["processed_log__CAN__speed__value"]).reshape(-1)
    steer_v = np.asarray(log["processed_log__CAN__steering_angle__value"], dtype=np.float64).reshape(-1)
    frame_times = np.asarray(log["global_pose__frame_times"], dtype=np.float64).reshape(-1)
    return Comma2k19Segment.from_arrays(
        row["segment_id"],
        speed_t=log["processed_log__CAN__speed__t"],
        speed_v=speed_v,
        steer_t=log["processed_log__CAN__steering_angle__t"],
        steer_v=steer_v,
        frame_times=frame_times,
    )


def gyro_cross_check(row: pd.Series, segment: Comma2k19Segment) -> dict:
    """steering_angle과 IMU gyro_z(요레이트)의 부호 관계를 이 세그먼트에서 측정한다.

    EXPERIMENT_DESIGN.md §9-3와 동일한 방법 — FRD 좌표계에서 +gyro_z = 우회전이므로,
    steering_angle이 gyro_z와 반대 부호일 때 "steering_angle>0 = LEFT"가 성립한다.
    """
    log = row["log"]
    gyro_t = np.asarray(log["processed_log__IMU__gyro__t"], dtype=np.float64).reshape(-1)
    gyro_xyz = _stack(log["processed_log__IMU__gyro__value"])
    gyro_z = gyro_xyz[:, 2]

    t0 = max(segment.steer_t[0], gyro_t[0])
    t1 = min(segment.steer_t[-1], gyro_t[-1])
    mask = (segment.steer_t >= t0) & (segment.steer_t <= t1)
    st, sv = segment.steer_t[mask], segment.steer_v[mask]
    if len(st) < 5:
        return {"n": len(st), "corr": np.nan, "n_clear": 0, "opposite_sign_rate": np.nan}
    gz = np.interp(st, gyro_t, gyro_z)

    corr = float(np.corrcoef(sv, gz)[0, 1]) if np.std(sv) > 0 and np.std(gz) > 0 else float("nan")
    clear = (np.abs(sv) > 1.0) & (np.abs(gz) > np.deg2rad(1.0))
    n_clear = int(clear.sum())
    opposite_rate = float(np.mean(np.sign(sv[clear]) != np.sign(gz[clear]))) if n_clear else float("nan")
    return {"n": len(st), "corr": corr, "n_clear": n_clear, "opposite_sign_rate": opposite_rate}


def analyze(parquet_dir: Path, config: LabelConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or LabelConfig.load()
    df = pd.read_parquet(parquet_dir, engine="pyarrow")

    label_tables = []
    gyro_rows = []
    failures = []

    for _, row in df.iterrows():
        seg_id = row["segment_id"]
        try:
            segment = load_segment(row)
            grid = build_10hz_grid(segment)
            speed, steer = interpolate(segment, grid)
            accel = accel_from_speed(grid, speed, config)
            table = pd.DataFrame(
                {
                    "segment_id": seg_id,
                    "route_id": route_id(seg_id),
                    "sample_index": np.arange(len(grid)),
                    "speed_mps": speed,
                    "accel_mps2": accel,
                    "steering_angle_deg": steer,
                    "accel_label": accel_label(speed, accel, config),
                    "steer_label": steer_label(steer, config),
                }
            )
            label_tables.append(table)

            gyro_result = gyro_cross_check(row, segment)
            gyro_result["segment_id"] = seg_id
            gyro_result["route_id"] = route_id(seg_id)
            gyro_rows.append(gyro_result)
        except Exception as exc:  # noqa: BLE001 - 수집해서 한꺼번에 보고
            failures.append({"segment_id": seg_id, "error": repr(exc)})

    labels = pd.concat(label_tables, ignore_index=True) if label_tables else pd.DataFrame()
    gyro_df = pd.DataFrame(gyro_rows)
    failures_df = pd.DataFrame(failures)
    return {"labels": labels, "gyro_check": gyro_df, "failures": failures_df, "raw": df}


def main() -> None:
    parser = argparse.ArgumentParser(description="comma2k19 demo64 parquet -> EXP-S3-LABEL-002 분석")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    result = analyze(args.parquet_dir)

    labels = result["labels"]
    labels.to_csv(args.out / "labels_10hz_all_segments.csv", index=False, encoding="utf-8-sig")

    n_segments = labels["segment_id"].nunique()
    n_routes = labels["route_id"].nunique()
    print(f"세그먼트 {n_segments}개 (route {n_routes}개), 총 {len(labels)} sample")

    if not result["failures"].empty:
        print("\n실패한 세그먼트:")
        print(result["failures"].to_string(index=False))

    report = class_distribution_report(labels)
    report.to_csv(args.out / "class_distribution_all_segments.csv", index=False, encoding="utf-8-sig")
    print("\n=== 전체 64개 세그먼트 class distribution ===")
    print(report.to_string(index=False))

    gyro_df = result["gyro_check"]
    gyro_df.to_csv(args.out / "gyro_cross_check.csv", index=False, encoding="utf-8-sig")
    valid = gyro_df.dropna(subset=["corr"])
    print(f"\n=== gyro_z 교차검증 ({len(valid)}/{len(gyro_df)}개 세그먼트) ===")
    print(f"corr 평균: {valid['corr'].mean():.3f}  (범위 {valid['corr'].min():.3f} ~ {valid['corr'].max():.3f})")
    print(f"corr > 0(steering_angle과 같은 부호)인 세그먼트: {(valid['corr'] > 0).sum()}개")
    print(f"corr < 0(steering_angle과 반대 부호)인 세그먼트: {(valid['corr'] < 0).sum()}개")
    total_clear = int(valid["n_clear"].sum())
    weighted_opposite = (
        (valid["opposite_sign_rate"] * valid["n_clear"]).sum() / total_clear if total_clear else float("nan")
    )
    print(f"뚜렷한 구간(|steer|>1deg,|gyro_z|>1deg/s) 전체 {total_clear}개, 가중평균 반대부호율: {weighted_opposite:.3f}")


if __name__ == "__main__":
    main()
