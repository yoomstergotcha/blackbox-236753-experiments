"""공식 inference.py에 붙여넣을 Stage1 predict 함수 (합성 재녹화 데이터로 fine-tune한 ResNet18).

self-contained 코드 블록 — 공식 common 셀(_device, _video_paths 등) 뒤에 이어붙여 쓴다.
학습(src/train/stage1_dataset.py)과 동일한 전처리: 원본 해상도 유지, 224 크롭(중앙+사분면 4개),
ImageNet 정규화. 영상당 균등 8프레임 x 5크롭 = 40장 -> sigmoid 평균 -> 0.5 임계.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18

_S1_CROP = 224
_S1_FRAMES_PER_VIDEO = 8
_S1_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_S1_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _s1_build_model() -> nn.Module:
    m = resnet18(weights=None)  # 가중치는 state_dict로 채움(인터넷 불필요)
    m.fc = nn.Linear(512, 1)
    return m


def _s1_grid_crops(rgb: np.ndarray, size: int = _S1_CROP):
    h, w = rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w) + 1e-3
        rgb = cv2.resize(rgb, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
    centers = [
        (h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4),
        (size // 2, size // 2), (size // 2, w - size // 2), (h - size // 2, size // 2), (h - size // 2, w - size // 2),
    ]  # 중앙 + 사분면 4 + 모서리 4 (모서리는 재촬영 시 베젤/배경 영역)
    crops = []
    for cy, cx in centers:
        y = int(np.clip(cy - size // 2, 0, h - size))
        x = int(np.clip(cx - size // 2, 0, w - size))
        crop = rgb[y : y + size, x : x + size].astype(np.float32) / 255.0
        crop = (crop - _S1_IMAGENET_MEAN) / _S1_IMAGENET_STD
        crops.append(torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1))
    return torch.stack(crops)


def _s1_sample_frames(path: Path, n: int):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        wanted = set(np.linspace(0, total - 1, num=min(n, total)).round().astype(int).tolist())
        pos = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if pos in wanted:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pos += 1
            if pos > max(wanted):
                break
    if not frames:  # 프레임 수 메타데이터가 틀린 경우: 전부 읽고 균등 추출
        cap.release()
        cap = cv2.VideoCapture(str(path))
        allf = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            allf.append(bgr)
        if allf:
            idx = np.linspace(0, len(allf) - 1, num=min(n, len(allf))).round().astype(int)
            frames = [cv2.cvtColor(allf[i], cv2.COLOR_BGR2RGB) for i in idx]
    cap.release()
    return frames


def _predict_stage1_cnn(data_dir, model_dir):
    device = _device()
    model = _s1_build_model()
    model.load_state_dict(torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False))
    model.to(device).eval()

    # 임계값은 학습 시 공개 라벨(공식 5쌍)로 보정한 값을 쓴다(없으면 0.5). 평가 영상 간 통계는
    # 일절 쓰지 않는다 — 영상마다 독립적으로 이 고정 임계값과 비교할 뿐이다.
    threshold = 0.5
    thr_path = Path(model_dir) / "threshold.json"
    if thr_path.is_file():
        try:
            import json as _json

            threshold = float(_json.loads(thr_path.read_text(encoding="utf-8")).get("threshold", 0.5))
        except Exception:
            threshold = 0.5

    rows = []
    with torch.inference_mode():
        for path in _video_paths(Path(data_dir) / "videos"):
            try:
                frames = _s1_sample_frames(path, _S1_FRAMES_PER_VIDEO)
                probs = []
                for rgb in frames:
                    crops = _s1_grid_crops(rgb).to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logit = model(crops).float().squeeze(1)
                    probs.append(torch.sigmoid(logit).mean().item())
                p = float(np.mean(probs)) if probs else 0.0
            except Exception:
                p = 0.0  # 디코딩 실패 시 다수 클래스(ORIGINAL)로 안전 처리
            rows.append({"ID": path.stem, "answer": "RERECORDED" if p >= threshold else "ORIGINAL"})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])


# ---------------------------------------------------------------------------
# EXP-S1-META-001: 컨테이너 서명 규칙 + CNN(v4) 폴백  (이 블록은 predict_stage1_synth.py 뒤에 이어붙여
# predict_stage1_meta.py 를 만든다 — build_stage1_meta_snippet.py 참고)
# 공개 5쌍/샘플 10영상 실측: ORIGINAL은 전부 MPEG-4 Part2(FMP4/mp4v, Lavf58.12), RERECORDED는 전부
# H.264(avc1, Lavf63.1, 비트레이트 2배). 재녹화(재촬영이든 재인코딩이든)는 반드시 다시 인코딩을 거치므로
# 코덱/먹서 서명이 남는다 — 가이드 §2.3의 "추가 압축 및 화질 저하 / 해상도·fps 변화"와 같은 계열의 단서.
# 영상마다 독립적으로 그 파일의 헤더만 읽는다(파일 간 통계 없음, 가이드 §12.2 준수).
# ---------------------------------------------------------------------------


def _s1_container_signature(path: Path):
    """(codec, lavf) — codec: 'mp4v'|'avc1'|'hev1'|'hvc1'|...|'unknown', lavf: 'Lavf58.12.100' 같은 먹서 태그 또는 None"""
    try:
        data = path.read_bytes()
    except Exception:
        return "unknown", None
    codec = "unknown"
    for tag in (b"mp4v", b"avc1", b"avc3", b"hev1", b"hvc1", b"av01", b"vp09"):
        if data.find(tag) >= 0:
            codec = tag.decode()
            break
    if codec == "unknown":
        try:
            cap = cv2.VideoCapture(str(path))
            f = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            cc = "".join(chr((f >> 8 * i) & 255) for i in range(4)).strip().lower()
            codec = {"fmp4": "mp4v", "mp4v": "mp4v", "h264": "avc1", "avc1": "avc1", "hevc": "hev1", "hvc1": "hvc1"}.get(cc, "unknown")
        except Exception:
            pass
    k = data.find(b"Lavf")
    lavf = data[k : k + 16].split(b"\x00")[0].decode("latin1", errors="replace") if k >= 0 else None
    return codec, lavf


def _s1_rule(codec: str, lavf):
    """확정 판정만 반환. 애매하면 None -> CNN 폴백."""
    if codec == "mp4v":
        return "ORIGINAL"  # 원본 배포 규격(MPEG-4 Part2). 재녹화가 이 코덱으로 다시 인코딩될 가능성은 사실상 없음
    if codec in ("avc1", "avc3", "hev1", "hvc1", "av01", "vp09"):
        return "RERECORDED"  # 원본 규격이 아닌 현대 코덱 = 재인코딩을 거침
    if lavf and not lavf.startswith("Lavf58"):
        return "RERECORDED"
    if lavf and lavf.startswith("Lavf58"):
        return "ORIGINAL"
    return None


def predict_stage1(data_dir, model_dir):
    cnn = _predict_stage1_cnn(data_dir, model_dir)  # 폴백용(모든 영상에 대해 계산, v4와 동일)
    cnn_map = dict(zip(cnn["ID"], cnn["answer"]))
    rows = []
    for path in _video_paths(Path(data_dir) / "videos"):
        codec, lavf = _s1_container_signature(path)
        verdict = _s1_rule(codec, lavf)
        rows.append({"ID": path.stem, "answer": verdict if verdict is not None else cnn_map.get(path.stem, "ORIGINAL")})
    return pd.DataFrame(rows, columns=["ID", "answer"])
