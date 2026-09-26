"""EXP-S2-DET-001 — CCD ego 클립 전 프레임에 YOLOv8(COCO) 차량 검출을 돌려 캐시한다.

물리적 정의 기반 충돌/진입 검출(접촉 = 피해차량이 가장 가까워지는 순간, 진입 = 피해차량 박스가 내 차선 영역에 처음 들어오는 순간)의 재료.
출력: <out>/<vid>.npz — det (M,7) float32 [frame, x1, y1, x2, y2, conf, cls] (정규화 좌표 0~1), n_frames.
차량 클래스: car 2, motorcycle 3, bus 5, truck 7 (COCO). conf ≥ 0.2.
실행: python -m src.train.stage2_ccd_detect --weights data/external/hf_models/yolo/yolov8s.pt --out output/stage2_ccd_det
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

VEHICLE = {2, 3, 5, 7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("data/external/ccd"))
    ap.add_argument("--vids", type=Path, default=Path("output/ccd_ego_vids.txt"))
    ap.add_argument("--weights", type=Path, default=Path("data/external/hf_models/yolo/yolov8s.pt"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_ccd_det"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)
    vids = [l.strip() for l in open(args.vids)]
    done = 0
    for vid in vids:
        dst = args.out / f"{vid}.npz"
        if dst.is_file():
            done += 1
            continue
        cap = cv2.VideoCapture(str(args.videos / f"{vid}.mp4"))
        frames = []
        while True:
            ok, b = cap.read()
            if not ok:
                break
            frames.append(b)
        cap.release()
        rows = []
        for s in range(0, len(frames), 32):
            batch = frames[s : s + 32]
            res = model.predict(batch, imgsz=args.imgsz, conf=0.2, verbose=False, device=0 if torch.cuda.is_available() else "cpu", half=torch.cuda.is_available())
            for j, r in enumerate(res):
                h, w = batch[j].shape[:2]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for k in range(len(xyxy)):
                    if int(cls[k]) in VEHICLE:
                        rows.append([s + j, xyxy[k, 0] / w, xyxy[k, 1] / h, xyxy[k, 2] / w, xyxy[k, 3] / h, conf[k], cls[k]])
        np.savez(dst, det=np.array(rows, np.float32).reshape(-1, 7), n_frames=np.int64(len(frames)))
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(vids)}", flush=True)
    print("done", done)


if __name__ == "__main__":
    main()
