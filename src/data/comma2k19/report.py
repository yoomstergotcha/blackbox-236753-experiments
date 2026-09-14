"""accel_label/steer_label 클래스 분포 리포트 — Macro-F1 클래스 불균형 대응 설계용."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .labeling import ACCEL_LABELS, STEER_LABELS


def class_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    """df는 accel_label, steer_label 컬럼을 가진 10Hz 라벨 테이블.

    steer는 accel_label == 'STOPPED'인 행을 제외한 분포도 함께 계산한다
    (COMPETITION_GUIDE.md §8: STOPPED 구간은 steer 채점에서 제외되지만 값은 출력해야 함).
    """
    accel_counts = Counter(df["accel_label"])
    steer_all_counts = Counter(df["steer_label"])
    steer_scored_counts = Counter(df.loc[df["accel_label"] != "STOPPED", "steer_label"])

    n = len(df)
    n_scored = sum(steer_scored_counts.values())
    rows = []
    for label in ACCEL_LABELS:
        count = accel_counts.get(label, 0)
        rows.append({"group": "accel_label", "label": label, "count": count, "ratio": count / n if n else 0.0})
    for label in STEER_LABELS:
        count = steer_all_counts.get(label, 0)
        rows.append(
            {"group": "steer_label(all)", "label": label, "count": count, "ratio": count / n if n else 0.0}
        )
    for label in STEER_LABELS:
        count = steer_scored_counts.get(label, 0)
        rows.append(
            {
                "group": "steer_label(scored, STOPPED 제외)",
                "label": label,
                "count": count,
                "ratio": count / n_scored if n_scored else 0.0,
            }
        )
    return pd.DataFrame(rows)
