"""공식 베이스라인 노트북 2종([Baseline_Train], [Baseline_Inference])의 코드를 그대로
스크립트로 재현해 순서대로 실행한다 — 학습 -> inference.py 생성 -> smoke-test 추론 ->
submit.zip 생성까지 한 번에 수행.

노트북을 열지 않고도 오늘 제출 가능한 베이스라인 zip을 만들기 위한 용도. 노트북 셀의
코드를 그대로 옮겼을 뿐 로직을 바꾸지 않았다 — 두 노트북 파일 자체는 그대로 둔다.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import mvit_v2_s

ROOT = Path.cwd()
DATA = ROOT / "data"
MODEL = ROOT / "model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(os.getenv("EPOCHS", "1"))

SIZE = 224
S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]
torch.manual_seed(20260825)
random.seed(20260825)


# ===========================================================================
# [Baseline_Train] 1. 라이브러리·모델 구조·학습함수
# ===========================================================================
def _video_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise ValueError(f"cannot decode: {path}")
    return out


def _crop_tensor(rgb, size=224):
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (nh - size) // 2, (nw - size) // 2
    return torch.from_numpy(rgb[y : y + size, x : x + size].copy()).permute(2, 0, 1).float() / 255


def _clip(path, n=16, center=None):
    frames = _video_frames(path)
    total = len(frames)
    if center is None:
        idx = np.linspace(0, total - 1, n).round().astype(int)
    else:
        idx = np.clip(center - n // 2 + np.arange(n), 0, total - 1)
    x = torch.stack([_crop_tensor(frames[int(i)]) for i in idx], 1)
    return x, total


class Stage1MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mvit_v2_s(weights=None)
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, x):
        return self.net(x)


class Stage2Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def logits(self, x):
        h, _ = self.r(x)
        return self.tc(h).squeeze(-1), self.te(h).squeeze(-1), h

    def forward(self, x):
        collision, entry, h = self.logits(x)
        ci, ei = collision.argmax(1), entry.argmax(1)
        b = torch.arange(len(h), device=h.device)
        return ci, ei, self.scene(torch.cat([h[b, ci], h[b, ei]], 1))


class Stage3MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dim = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dim, 4)
        self.steer = nn.Linear(dim, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.accel(z), self.steer(z)


def fit_stage1():
    out = MODEL / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "stage1/labels.csv")
    model = Stage1MViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    for _ in range(EPOCHS):
        model.train()
        for r in df.sample(frac=1, random_state=20260825).itertuples():
            x, _ = _clip(DATA / "stage1" / r.path, 16)
            x = (x - S1_MEAN) / S1_STD
            y = torch.tensor([0 if r.label == "ORIGINAL" else 1], device=DEVICE)
            loss = nn.functional.cross_entropy(model(x[None].to(DEVICE)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save({"model": model.net.state_dict(), "size": 224, "frames": 16}, out / "best.pt")


def _resnet_backbone():
    try:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        print("경고: ImageNet 가중치를 받지 못해 weights=None으로 진행합니다.")
        model = resnet18(weights=None)
    return model


def fit_stage2():
    out = MODEL / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "stage2/labels.csv")
    backbone = _resnet_backbone()
    torch.save(backbone.state_dict(), out / "resnet18-f37072fd.pth")
    backbone.fc = nn.Identity()
    backbone.to(DEVICE).eval()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    sequences = []
    with torch.inference_mode():
        for r in df.itertuples():
            frames = _video_frames(DATA / "stage2" / r.path)
            batches = []
            for start in range(0, len(frames), 64):
                x = torch.stack([transform(Image.fromarray(a)) for a in frames[start : start + 64]]).to(DEVICE)
                batches.append(backbone(x).float().cpu())
            sequences.append((torch.cat(batches), min(int(r.t_collision), len(frames) - 1)))
    temporal = Stage2Temporal().to(DEVICE)
    opt = torch.optim.AdamW(temporal.parameters(), 2e-4)
    for _ in range(max(1, EPOCHS)):
        temporal.train()
        for seq, target in sequences:
            collision, _, _ = temporal.logits(seq[None].to(DEVICE))
            loss = nn.functional.cross_entropy(collision, torch.tensor([target], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save({"model": temporal.state_dict()}, out / "best.pt")


def fit_stage3():
    out = MODEL / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "stage3/labels.csv")
    amap = {"ACCELERATING": 0, "DECELERATING": 1, "CONSTANT": 2, "STOPPED": 3}
    smap = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
    model = Stage3MViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    for _ in range(EPOCHS):
        model.train()
        for r in df.itertuples():
            x, _ = _clip(DATA / "stage3/videos" / f"{r.ID}.mp4", 16, int(r.frame_index))
            x = (x - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]
            a, s = model(x[None].to(DEVICE))
            loss = nn.functional.cross_entropy(a, torch.tensor([amap[r.accel_label]], device=DEVICE))
            loss += nn.functional.cross_entropy(s, torch.tensor([smap[r.steer_label]], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save({"model": model.state_dict()}, out / "best.pt")


def run_training():
    print("device:", DEVICE)
    fit_stage1()
    print("Stage 1 완료")
    fit_stage2()
    print("Stage 2 완료")
    fit_stage3()
    print("Stage 3 완료")
    for p in sorted((ROOT / "model").rglob("*")):
        if p.is_file():
            print(p.relative_to(ROOT), f"{p.stat().st_size / 1024**2:.1f} MB")


# ===========================================================================
# [Baseline_Inference] 5. inference.py 생성 (노트북 자체를 파싱해 BASELINE_INFERENCE_PART 셀 추출)
# ===========================================================================
def build_inference_py():
    notebook_path = ROOT / "[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb"
    inference_path = ROOT / "inference.py"
    if not notebook_path.is_file():
        raise FileNotFoundError(f"현재 추론 노트북을 찾을 수 없습니다: {notebook_path}")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    inference_parts = []
    marker = "# " + "BASELINE_INFERENCE_PART"
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        if marker in cell_source:
            inference_parts.append(cell_source)

    if len(inference_parts) != 4:
        raise RuntimeError(f"inference.py 생성에 필요한 코드 셀은 4개여야 합니다. 현재: {len(inference_parts)}개")

    inference_source = "\n\n".join(part.rstrip() for part in inference_parts) + "\n"
    tree = ast.parse(inference_source, filename="inference.py")
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {"predict_stage1", "predict_stage2", "predict_stage3"}
    missing = sorted(required - defined)
    if missing:
        raise RuntimeError("필수 추론 함수가 없습니다: " + str(missing))

    inference_path.write_text(inference_source, encoding="utf-8")

    # 주의: importlib.util.module_from_spec로 동적 로딩하면 Windows의 spawn 기반
    # multiprocessing(DataLoader num_workers>0)에서 워커 프로세스가 이 모듈을 다시
    # import하지 못해 PicklingError가 난다(실제 inference.py 코드의 결함이 아니라
    # 이 smoke-test 방식과 Windows 조합의 문제 — Linux 평가서버는 fork를 쓰므로
    # 이 문제가 없다). 그래서 여기서는 실제 제출되는 inference.py를 그대로 두고,
    # 검증 스크립트에서만 정상적인 이름 기반 import를 사용한다.
    import sys

    sys.path.insert(0, str(ROOT))
    sys.modules.pop("inference", None)
    import inference as module  # noqa: PLC0415

    importlib.reload(module)
    print("생성 완료:", inference_path)
    print("확인된 함수:", sorted(required))
    return module


# ===========================================================================
# [Baseline_Inference] 7. 공개 예제 평가 입력 생성 (smoke test)
# ===========================================================================
def build_smoke_dir():
    smoke_dir = ROOT / "sample_evaluation_data"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    (smoke_dir / "stage1" / "videos").mkdir(parents=True)
    (smoke_dir / "stage2" / "images").mkdir(parents=True)
    (smoke_dir / "stage3" / "videos").mkdir(parents=True)

    for label, folder in [("O", "original"), ("R", "rerecorded")]:
        for index, path in enumerate(sorted((DATA / "stage1" / folder).glob("*")), 1):
            target = smoke_dir / "stage1" / "videos" / f"SAMPLE_S1_{label}_{index:03d}{path.suffix.lower()}"
            shutil.copy2(path, target)

    for index, video_path in enumerate(sorted((DATA / "stage2" / "videos").glob("*")), 1):
        frame_dir = smoke_dir / "stage2" / "images" / f"SAMPLE_S2_{index:03d}"
        frame_dir.mkdir()
        capture = cv2.VideoCapture(str(video_path))
        frame_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            cv2.imwrite(str(frame_dir / f"frame_{frame_index:06d}.jpg"), image)
            frame_index += 1
        capture.release()

    for path in sorted((DATA / "stage3" / "videos").glob("*")):
        shutil.copy2(path, smoke_dir / "stage3" / "videos" / path.name)

    print("예제 평가 입력 생성 완료:", smoke_dir)
    return smoke_dir


def run_smoke_predictions(module, smoke_dir):
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    stage1_pred = module.predict_stage1(smoke_dir / "stage1", MODEL / "stage1")
    stage2_pred = module.predict_stage2(smoke_dir / "stage2", MODEL / "stage2")
    stage3_pred = module.predict_stage3(smoke_dir / "stage3", MODEL / "stage3")

    for stage, frame in {"stage1": stage1_pred, "stage2": stage2_pred, "stage3": stage3_pred}.items():
        output_path = output_dir / f"{stage}_submission.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"{stage}: {len(frame):,}행 -> {output_path}")
        print(frame.head())


# ===========================================================================
# [Baseline_Inference] 9. submit.zip 생성 및 최종 검사
# ===========================================================================
def build_submit_zip():
    inference_path = ROOT / "inference.py"
    submit_path = ROOT / "submit.zip"
    inference_source = inference_path.read_text(encoding="utf-8")

    tree = ast.parse(inference_source, filename="inference.py")
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {"predict_stage1", "predict_stage2", "predict_stage3"}
    missing = sorted(required - defined)
    if missing:
        raise RuntimeError("inference.py 생성 실패: 필수 추론 함수가 없습니다: " + str(missing))

    if submit_path.exists():
        submit_path.unlink()

    with zipfile.ZipFile(submit_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("inference.py", inference_source)
        archive.write(ROOT / "requirements.txt", "requirements.txt")
        for model_path in sorted(MODEL.rglob("*")):
            if model_path.is_file():
                archive.write(model_path, model_path.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(submit_path) as archive:
        names = archive.namelist()
        zipped_inference = archive.read("inference.py").decode("utf-8")

    required_files = {
        "inference.py",
        "requirements.txt",
        "model/stage1/best.pt",
        "model/stage2/best.pt",
        "model/stage2/resnet18-f37072fd.pth",
        "model/stage3/best.pt",
    }
    missing_files = sorted(required_files - set(names))
    if missing_files:
        raise RuntimeError("제출 ZIP 필수 파일 누락: " + str(missing_files))

    zipped_tree = ast.parse(zipped_inference, filename="submit.zip/inference.py")
    zipped_functions = {node.name for node in zipped_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing_in_zip = sorted(required - zipped_functions)
    if missing_in_zip:
        raise RuntimeError("ZIP 내부 inference.py 함수 누락: " + str(missing_in_zip))

    print(f"생성 완료: {submit_path}")
    print(f"압축 크기: {submit_path.stat().st_size / 1024**3:.3f} GB")
    print("ZIP 내부 파일:")
    for name in names:
        print(" -", name)


if __name__ == "__main__":
    run_training()
    module = build_inference_py()
    required_models = [
        MODEL / "stage1" / "best.pt",
        MODEL / "stage2" / "best.pt",
        MODEL / "stage2" / "resnet18-f37072fd.pth",
        MODEL / "stage3" / "best.pt",
    ]
    missing = [str(p) for p in required_models if not p.is_file()]
    if missing:
        raise FileNotFoundError("학습이 먼저 필요합니다: " + str(missing))
    smoke_dir = build_smoke_dir()
    run_smoke_predictions(module, smoke_dir)
    build_submit_zip()
