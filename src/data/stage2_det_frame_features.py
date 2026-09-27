"""YOLO 박스 캐시(output/stage2_ccd_det) → 프레임별 8-d 검출 특징을 기존 특징 npz에 'det' 키로 추가한 새 캐시 디렉토리 생성.
det[t] = [최대 차량 박스 면적, 그 박스 하단 y, 그 박스 중심 x 편차, 차량 박스 수, 면적 합, 3프레임 전 대비 최대 면적 증가, 박스 쌍 최소 중심거리(1=없음), 박스 쌍 최대 IoU]
실행: python -m src.data.stage2_det_frame_features [--src output/stage2_ccd_feats] [--det output/stage2_ccd_det] [--out output/stage2_ccd_feats_det]
"""
import argparse
from pathlib import Path
import numpy as np

VEH = {2, 3, 5, 7}


def frame_feats(det: np.ndarray, n: int, conf_min: float = 0.3) -> np.ndarray:
    f = np.zeros((n, 8), np.float32); f[:, 6] = 1.0
    boxes = [[] for _ in range(n)]
    for fr, x1, y1, x2, y2, cf, cl in det:
        i = int(fr)
        if int(cl) in VEH and cf >= conf_min and 0 <= i < n:
            boxes[i].append((x1, y1, x2, y2))
    for i, bs in enumerate(boxes):
        if not bs:
            continue
        b = np.array(bs); a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); j = int(a.argmax())
        f[i, 0], f[i, 1], f[i, 2], f[i, 3], f[i, 4] = a[j], b[j, 3], abs((b[j, 0] + b[j, 2]) / 2 - 0.5), len(bs), a.sum()
        if len(bs) > 1:
            c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1); d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(len(bs)) * 9
            f[i, 6] = d.min()
            ix1, iy1 = np.maximum(b[:, None, 0], b[None, :, 0]), np.maximum(b[:, None, 1], b[None, :, 1]); ix2, iy2 = np.minimum(b[:, None, 2], b[None, :, 2]), np.minimum(b[:, None, 3], b[None, :, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None); iou = inter / (a[:, None] + a[None] - inter + 1e-6); np.fill_diagonal(iou, 0); f[i, 7] = iou.max()
    f[3:, 5] = f[3:, 0] - f[:-3, 0]
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("output/stage2_ccd_feats"))
    ap.add_argument("--det", type=Path, default=Path("output/stage2_ccd_det"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_ccd_feats_det"))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); miss = 0
    for p in sorted(args.src.glob("*.npz")):
        z = dict(np.load(p)); n = z["signals"].shape[0]; dp = args.det / p.name
        if dp.is_file():
            z["det"] = frame_feats(np.load(dp)["det"], n)
        else:
            miss += 1; z["det"] = np.zeros((n, 8), np.float32); z["det"][:, 6] = 1.0
        np.savez(args.out / p.name, **z)
    print("done", len(list(args.out.glob('*.npz'))), "files | det missing", miss)


if __name__ == "__main__":
    main()
