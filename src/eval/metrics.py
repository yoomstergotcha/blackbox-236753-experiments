"""로컬 검증용 평가 하네스.

COMPETITION_GUIDE.md에 명시된 공식 채점 방식(§2.5, §4, §8)을 최대한 그대로 구현한다.
submit.zip에는 포함하지 않는다 — 실험 간 점수를 비교하기 위한 개발 도구다.

핵심 규칙 (가이드 기준):
- Stage 1: ORIGINAL/RERECORDED 두 클래스에 대한 Macro-F1.
- Stage 2: collision/entry는 프레임 번호를 해당 영상의 fps로 초 단위로 환산한 뒤
  |pred - gt| <= 0.3초 이면 정답. 내부 가중치는 collision 0.35 / entry 0.35 /
  evasion_space 0.15 / entry_side 0.15.
- Stage 3: accel_label Macro-F1(0.7) + steer_label Macro-F1(0.3). steer는
  정답 accel_label이 STOPPED인 행을 채점에서 제외하지만, 제출값 자체는 해당
  행에도 존재해야 한다(§8).
- Macro-F1은 데이터에 실제 등장한 클래스가 아니라 "정의된 전체 클래스 집합"
  기준으로 계산한다(§8) — 이 모듈은 항상 고정된 labels 목록을 넘겨 계산한다.
- 누락/NaN/음수 프레임/범위 초과/미정의 클래스값은 오답으로 처리한다(§4) —
  예외를 던지지 않고 "맞은 것으로 셀 수 없는 값"으로 처리해 하네스가 계속
  동작하게 한다.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

_INVALID = "__INVALID__"

STAGE1_LABELS: list[str] = ["ORIGINAL", "RERECORDED"]
STAGE2_EVASION_LABELS: list[int] = [0, 1]
STAGE2_ENTRY_SIDE_LABELS: list[str] = ["LEFT", "RIGHT"]
STAGE3_ACCEL_LABELS: list[str] = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STAGE3_STEER_LABELS: list[str] = ["LEFT", "STRAIGHT", "RIGHT"]

STAGE_WEIGHTS = {"stage1": 0.20, "stage2": 0.40, "stage3": 0.40}


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    """정의된 전체 클래스 집합(labels) 기준 Macro-F1."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def _sanitize(values: Iterable, labels: Sequence) -> list:
    """labels에 속하지 않는 값(결측·오타·정의되지 않은 값)을 오답 sentinel로 치환."""
    label_set = set(labels)
    out = []
    for v in values:
        try:
            out.append(v if v in label_set else _INVALID)
        except TypeError:
            out.append(_INVALID)
    return out


def score_stage1(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Stage 1 — ORIGINAL/RERECORDED Macro-F1.

    gt_df / pred_df 필수 컬럼: ID, answer
    """
    merged = gt_df[["ID", "answer"]].rename(columns={"answer": "gt"}).merge(
        pred_df[["ID", "answer"]].rename(columns={"answer": "pred"}), on="ID", how="left"
    )
    pred = _sanitize(merged["pred"].tolist(), STAGE1_LABELS)
    f1 = macro_f1(merged["gt"].tolist(), pred, STAGE1_LABELS)
    return {
        "stage1_score": f1,
        "n": len(merged),
        "n_invalid": sum(1 for p in pred if p == _INVALID),
    }


def _hit_within_tolerance(pred_frame, gt_frame, fps, tol_seconds: float, max_frame: float | None = None) -> bool:
    if pd.isna(pred_frame):
        return False
    try:
        pred_frame = float(pred_frame)
    except (TypeError, ValueError):
        return False
    if pred_frame < 0:
        return False
    if max_frame is not None and pred_frame > max_frame:
        return False
    if not fps or fps <= 0:
        return False
    return abs(pred_frame - float(gt_frame)) / fps <= tol_seconds


def score_stage2(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    fps_map: Mapping[str, float],
    max_frame_map: Mapping[str, float] | None = None,
    tol_seconds: float = 0.3,
) -> dict:
    """Stage 2 — collision/entry timestamp + evasion_space/entry_side.

    gt_df / pred_df 필수 컬럼: ID, collision_frame, entry_frame, evasion_space, entry_side
    fps_map: {video_id: fps} — 프레임 번호를 초 단위로 환산하는 데 필요.
    max_frame_map: {video_id: 마지막 프레임 번호} — 지정하면 범위를 벗어난 예측을 오답 처리.
    """
    required = {"collision_frame", "entry_frame", "evasion_space", "entry_side"}
    missing = required - set(gt_df.columns)
    if missing:
        raise ValueError(f"gt_df에 다음 컬럼이 없습니다: {sorted(missing)}")

    merged = gt_df.merge(pred_df, on="ID", how="left", suffixes=("_gt", "_pred"))
    n = len(merged)

    collision_hits = entry_hits = 0
    evasion_true: list = []
    evasion_pred: list = []
    side_true: list = []
    side_pred: list = []
    missing_fps_ids: list = []

    for row in merged.itertuples(index=False):
        video_id = row.ID
        fps = fps_map.get(video_id)
        if not fps or fps <= 0:
            missing_fps_ids.append(video_id)
        max_frame = None if max_frame_map is None else max_frame_map.get(video_id)

        if _hit_within_tolerance(row.collision_frame_pred, row.collision_frame_gt, fps, tol_seconds, max_frame):
            collision_hits += 1
        if _hit_within_tolerance(row.entry_frame_pred, row.entry_frame_gt, fps, tol_seconds, max_frame):
            entry_hits += 1

        evasion_true.append(row.evasion_space_gt)
        evasion_pred.append(row.evasion_space_pred)
        side_true.append(row.entry_side_gt)
        side_pred.append(row.entry_side_pred)

    evasion_pred = _sanitize(evasion_pred, STAGE2_EVASION_LABELS)
    side_pred = _sanitize(side_pred, STAGE2_ENTRY_SIDE_LABELS)

    collision_acc = collision_hits / n if n else 0.0
    entry_acc = entry_hits / n if n else 0.0
    evasion_f1 = macro_f1(evasion_true, evasion_pred, STAGE2_EVASION_LABELS)
    side_f1 = macro_f1(side_true, side_pred, STAGE2_ENTRY_SIDE_LABELS)

    score = 0.35 * collision_acc + 0.35 * entry_acc + 0.15 * evasion_f1 + 0.15 * side_f1
    result = {
        "stage2_score": score,
        "collision_acc": collision_acc,
        "entry_acc": entry_acc,
        "evasion_macro_f1": evasion_f1,
        "entry_side_macro_f1": side_f1,
        "n": n,
    }
    if missing_fps_ids:
        result["missing_fps_ids"] = sorted(set(missing_fps_ids))
    return result


def score_stage3(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Stage 3 — accel_label(0.7) + steer_label(0.3, STOPPED 행 제외).

    gt_df / pred_df 필수 컬럼: ID, sample_index, accel_label, steer_label
    """
    required = {"accel_label", "steer_label"}
    missing = required - set(gt_df.columns)
    if missing:
        raise ValueError(f"gt_df에 다음 컬럼이 없습니다: {sorted(missing)}")

    merged = gt_df.merge(pred_df, on=["ID", "sample_index"], how="left", suffixes=("_gt", "_pred"))

    accel_true = merged["accel_label_gt"].tolist()
    accel_pred = _sanitize(merged["accel_label_pred"].tolist(), STAGE3_ACCEL_LABELS)
    accel_f1 = macro_f1(accel_true, accel_pred, STAGE3_ACCEL_LABELS)

    steer_mask = merged["accel_label_gt"] != "STOPPED"
    steer_true = merged.loc[steer_mask, "steer_label_gt"].tolist()
    steer_pred = _sanitize(merged.loc[steer_mask, "steer_label_pred"].tolist(), STAGE3_STEER_LABELS)
    steer_f1 = macro_f1(steer_true, steer_pred, STAGE3_STEER_LABELS)

    stopped_missing_steer = int(merged.loc[~steer_mask, "steer_label_pred"].isna().sum())

    score = 0.7 * accel_f1 + 0.3 * steer_f1
    return {
        "stage3_score": score,
        "accel_macro_f1": accel_f1,
        "steer_macro_f1": steer_f1,
        "n": len(merged),
        "n_steer_scored": int(steer_mask.sum()),
        "stopped_missing_steer_label": stopped_missing_steer,
    }


def total_score(stage1: dict, stage2: dict, stage3: dict) -> float:
    return (
        STAGE_WEIGHTS["stage1"] * stage1["stage1_score"]
        + STAGE_WEIGHTS["stage2"] * stage2["stage2_score"]
        + STAGE_WEIGHTS["stage3"] * stage3["stage3_score"]
    )


def _self_test() -> None:
    """손으로 계산한 값과 대조하는 sanity check. `python -m src.eval.metrics`로 실행."""
    f1 = macro_f1(["A", "A", "B", "B"], ["A", "B", "B", "B"], ["A", "B"])
    assert abs(f1 - 0.733333) < 1e-4, f1

    gt1 = pd.DataFrame({"ID": ["v1", "v2"], "answer": ["ORIGINAL", "RERECORDED"]})
    pred1 = pd.DataFrame({"ID": ["v1", "v2"], "answer": ["ORIGINAL", None]})
    r1 = score_stage1(pred1, gt1)
    assert abs(r1["stage1_score"] - 0.5) < 1e-6, r1
    assert r1["n_invalid"] == 1, r1

    gt2 = pd.DataFrame(
        {
            "ID": ["v1", "v2"],
            "collision_frame": [100, 50],
            "entry_frame": [80, 40],
            "evasion_space": [1, 0],
            "entry_side": ["LEFT", "RIGHT"],
        }
    )
    pred2 = pd.DataFrame(
        {
            "ID": ["v1", "v2"],
            "collision_frame": [109, 41],  # v1: 9프레임/30fps=0.3s(정답), v2: 9/30=0.3s(정답)
            "entry_frame": [90, 40],  # v1: 10/30=0.333s(오답), v2: 0s(정답)
            "evasion_space": [1, 1],  # v1 정답, v2 오답
            "entry_side": ["LEFT", "RIGHT"],  # 둘 다 정답
        }
    )
    r2 = score_stage2(pred2, gt2, fps_map={"v1": 30, "v2": 30})
    assert abs(r2["collision_acc"] - 1.0) < 1e-9, r2
    assert abs(r2["entry_acc"] - 0.5) < 1e-9, r2
    assert abs(r2["entry_side_macro_f1"] - 1.0) < 1e-9, r2
    assert abs(r2["stage2_score"] - 0.725) < 1e-6, r2

    gt3 = pd.DataFrame(
        {
            "ID": ["v1"] * 4,
            "sample_index": [0, 1, 2, 3],
            "accel_label": ["CONSTANT", "STOPPED", "ACCELERATING", "CONSTANT"],
            "steer_label": ["STRAIGHT", "STRAIGHT", "LEFT", "RIGHT"],
        }
    )
    pred3 = pd.DataFrame(
        {
            "ID": ["v1"] * 4,
            "sample_index": [0, 1, 2, 3],
            "accel_label": ["CONSTANT", "STOPPED", "DECELERATING", "CONSTANT"],
            "steer_label": ["STRAIGHT", "STRAIGHT", "LEFT", "STRAIGHT"],
        }
    )
    r3 = score_stage3(pred3, gt3)
    assert abs(r3["accel_macro_f1"] - 0.5) < 1e-6, r3
    assert abs(r3["steer_macro_f1"] - 0.555556) < 1e-4, r3
    assert r3["stopped_missing_steer_label"] == 0, r3
    assert abs(r3["stage3_score"] - 0.516667) < 1e-4, r3

    total = total_score(r1, r2, r3)
    assert abs(total - (0.20 * 0.5 + 0.40 * 0.725 + 0.40 * r3["stage3_score"])) < 1e-9

    print("모든 self-test 통과")
    print(f"  stage1={r1['stage1_score']:.4f} stage2={r2['stage2_score']:.4f} stage3={r3['stage3_score']:.4f}")
    print(f"  total={total:.4f}")


if __name__ == "__main__":
    _self_test()
