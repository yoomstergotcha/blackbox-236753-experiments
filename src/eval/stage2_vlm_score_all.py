"""배포 VLM 절차(predict_stage2_fixed3_vlm 상수·프롬프트 동일)로 CCD ego 클립 전부의 side/evasion 로짓을 계산해 캐시.
출력: output/s2_vlm_scores_all.csv — vid, side_score(LEFT−RIGHT), eva_img(YES−NO, 충돌 프레임 이미지), eva_vid(YES−NO, 비디오)
실행: python -m src.eval.stage2_vlm_score_all [--at gt|pred]
"""
import argparse, sys, time, cv2, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, ".")
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from src.train.train_stage2_collision import load_ann

MD = Path("data/external/hf_models/Qwen2-VL-2B-Instruct"); NF, W, SPAN = 8, 448, 14
Q_SIDE = "This is a dashcam video from the ego car, ending at the moment it collides with another vehicle. From which side of the screen did that other vehicle come into the ego car's path? Answer with exactly one word: LEFT or RIGHT."
Q_EVA_IMG = "This dashcam image shows the moment the camera car collides with another vehicle. Is there an open lane or free road space right next to the camera car where it could have steered to avoid the crash? Answer with exactly one word: YES or NO."
Q_EVA_VID = "This is a dashcam video ending at a collision. Just before the collision, was there empty road space to the left or right of the camera car (no other vehicle, wall, barrier or curb blocking it)? Answer with exactly one word: YES or NO."


def frame(bgr):
    h, w = bgr.shape[:2]; s = W / w
    return cv2.cvtColor(cv2.resize(bgr, (W, max(28, int(round(h * s / 28)) * 28))), cv2.COLOR_BGR2RGB)


def read_video(vid):
    cap = cv2.VideoCapture(f"data/external/ccd/{vid}.mp4"); fr = []
    while True:
        ok, b = cap.read()
        if not ok: break
        fr.append(b)
    return fr


@torch.inference_mode()
def logprobs(model, proc, media, question, cands, kind="video"):
    msgs = [{"role": "user", "content": [{"type": kind}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {"videos": [media]} if kind == "video" else {"images": [media]}
    n_prompt = proc(text=[text], return_tensors="pt", **kw)["input_ids"].shape[1]; out = []
    for cand in cands:
        full = proc(text=[text + cand], return_tensors="pt", **kw).to("cuda")
        logits = model(**full).logits[0].float(); ids = full["input_ids"][0]
        out.append(torch.log_softmax(logits[n_prompt - 1: -1], -1).gather(1, ids[n_prompt:, None]).sum().item())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--at", default="gt"); ap.add_argument("--out", default="output/s2_vlm_scores_all.csv"); a = ap.parse_args()
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.ego & (ann.onset >= 0)]
    at = {v: int(o) for v, o in zip(ann.vid, ann.onset)}
    if a.at == "pred":
        z = np.load("output/s2_ens11_oof_permodel.npz"); E = z["S"].mean(0); n = E.shape[1]
        at.update({v: int(np.argmax(E[i][: n - 3])) for i, v in enumerate(z["vids"])})
    done = set()
    if Path(a.out).is_file(): done = set(pd.read_csv(a.out, dtype={"vid": str}).vid)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MD, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MD)
    t0 = time.time(); rows = []
    for i, (v, c) in enumerate(sorted(at.items())):
        if v in done: continue
        fr = read_video(v)
        if not fr: continue
        c = min(c, len(fr) - 1); idx = np.linspace(max(c - SPAN, 0), min(c + 1, len(fr) - 1), NF).round().astype(int)
        video = np.stack([frame(fr[j]) for j in idx])
        ls = logprobs(model, proc, video, Q_SIDE, ["LEFT", "RIGHT"]); l1 = logprobs(model, proc, frame(fr[c]), Q_EVA_IMG, ["YES", "NO"], "image"); l2 = logprobs(model, proc, video, Q_EVA_VID, ["YES", "NO"])
        rows.append(dict(vid=v, at=c, side_score=ls[0] - ls[1], eva_img=l1[0] - l1[1], eva_vid=l2[0] - l2[1]))
        if len(rows) % 25 == 0:
            print(f"{len(rows)} scored, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if Path(a.out).is_file(): df = pd.concat([pd.read_csv(a.out, dtype={"vid": str}), df])
    df.to_csv(a.out, index=False); print("done", len(df), "rows ->", a.out)


if __name__ == "__main__":
    main()
