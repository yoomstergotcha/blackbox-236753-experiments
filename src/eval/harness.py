"""저장된 예측 CSV와 정답 CSV로 로컬 점수를 계산하는 CLI.

pred/gt CSV는 predict_stageX가 반환하는 것과 동일한 컬럼 스키마여야 한다
(COMPETITION_GUIDE.md §13).

사용 예:
    python -m src.eval.harness --stage 1 --pred stage1_pred.csv --gt stage1_gt.csv
    python -m src.eval.harness --stage 2 --pred stage2_pred.csv --gt stage2_gt.csv \
        --video-dir data/stage2/videos
    python -m src.eval.harness --stage 3 --pred stage3_pred.csv --gt stage3_gt.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import metrics

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}


def _video_id_path_map(video_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(video_dir.iterdir()) if p.suffix.lower() in _VIDEO_EXTS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage별 로컬 점수 계산")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None, help="Stage 2 fps 계산용 원본 영상 폴더")
    parser.add_argument("--tol", type=float, default=0.3, help="Stage 2 시간 허용 오차(초), 기본 0.3")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred)
    gt_df = pd.read_csv(args.gt)

    if args.stage == 1:
        result = metrics.score_stage1(pred_df, gt_df)
    elif args.stage == 2:
        if args.video_dir is None:
            parser.error("--stage 2에는 --video-dir가 필요합니다 (fps 계산용)")
        from .video_stats import build_fps_and_frame_count

        video_paths = _video_id_path_map(args.video_dir)
        fps_map, max_frame_map = build_fps_and_frame_count(video_paths)
        result = metrics.score_stage2(pred_df, gt_df, fps_map, max_frame_map, tol_seconds=args.tol)
    else:
        result = metrics.score_stage3(pred_df, gt_df)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
