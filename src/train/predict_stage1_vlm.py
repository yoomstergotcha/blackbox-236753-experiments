"""공식 inference.py에 붙여넣을 Stage1 predict 함수 — VLM(Qwen2-VL-2B-Instruct, Apache-2.0) zero-shot '화면 재촬영' 판별.

영상당 K프레임을 뽑아 "직접 촬영(DIRECT) vs 화면을 찍은 것(SCREEN)" log-prob 차의 평균을 고정 임계와 비교한다(파일 간 통계 없음).
가중치는 model/stage2/qwen2vl/ 를 공유(model_dir/qwen2vl 가 있으면 우선). 로딩·추론 실패 시 ORIGINAL.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

_S1V_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".3gp"}
_S1V_K, _S1V_W = 3, 896
_S1V_THR = -0.5  # CCD 원본 p90 −0.75 / 합성 capture 중앙값 +0.27; 로컬 검증(CCD 원본 vs 합성 재촬영)으로 보정한 고정 상수
_S1V_Q = ("Is this image a direct recording from a car dashboard camera, or is it a photo of a screen (a monitor, TV or phone display) "
          "that is showing dashcam footage? Look for screen edges or bezels, moire or pixel-grid patterns, reflections or glare on a "
          "display surface, or a tilted perspective of a display. Answer with exactly one word: DIRECT or SCREEN.")


def _s1v_load(model_dir, device):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    for d in (Path(model_dir) / "qwen2vl", Path(model_dir).parent / "stage2" / "qwen2vl"):
        if d.is_dir():
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            return Qwen2VLForConditionalGeneration.from_pretrained(d, torch_dtype=dtype).to(device).eval(), AutoProcessor.from_pretrained(d)
    return None


def _s1v_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(max(n * 0.2, 0), max(n * 0.8, 0), _S1V_K).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        s = _S1V_W / w
        bgr = cv2.resize(bgr, (_S1V_W, max(28, int(round(h * s / 28)) * 28)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def _s1v_score(vlm, device, img):
    model, proc = vlm
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _S1V_Q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    n_prompt = proc(text=[text], images=[img], return_tensors="pt")["input_ids"].shape[1]
    lp = []
    for cand in ("SCREEN", "DIRECT"):
        full = proc(text=[text + cand], images=[img], return_tensors="pt").to(device)
        logits = model(**full).logits[0].float()
        ids = full["input_ids"][0]
        lp.append(torch.log_softmax(logits[n_prompt - 1 : -1], -1).gather(1, ids[n_prompt:, None]).sum().item())
    return lp[0] - lp[1]


def predict_stage1(data_dir, model_dir):
    device = _device()
    video_dir = Path(data_dir) / "videos"
    if not video_dir.is_dir():
        video_dir = Path(data_dir)
    try:
        vlm = _s1v_load(model_dir, device)
    except Exception:
        vlm = None
    rows = []
    with torch.inference_mode():
        for path in sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in _S1V_VIDEO_EXT):
            answer = "ORIGINAL"
            if vlm is not None:
                try:
                    frames = _s1v_frames(path)
                    if frames:
                        score = float(np.mean([_s1v_score(vlm, device, f) for f in frames]))
                        answer = "RERECORDED" if score > _S1V_THR else "ORIGINAL"
                except Exception:
                    answer = "ORIGINAL"
            rows.append({"ID": path.stem, "answer": answer})
    del vlm
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])
