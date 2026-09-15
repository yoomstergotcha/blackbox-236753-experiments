"""제출 후보 2 — Stage1/2는 공식 베이스라인 그대로, Stage3만 comma2k19로 학습한
모델로 교체한 submit_v2.zip을 만든다.

전제조건:
  - run_baseline_pipeline.py를 먼저 실행해 model/stage1, model/stage2가 준비돼 있어야 한다.
  - src.train.train_stage3_multiroute --model resnet 로 학습한 best.pt가 있어야 한다
    (EXP-S3-BASE-005, output/exp_s3_base_005/best.pt).

SUBMISSION_STRATEGY.md §3 원칙: 한 번에 한 Stage만 바꾼다 - Stage1/2는 baseline과
100% 동일해야 LB 델타가 Stage3 변경분만 반영한다.
"""
from __future__ import annotations

import ast
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

STAGE3_CHECKPOINT = ROOT / "output" / "exp_s3_base_005" / "best.pt"
INFERENCE_NOTEBOOK = ROOT / "[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb"
STAGE3_SNIPPET = ROOT / "src" / "train" / "predict_stage3_comma2k19.py"


def build_inference_v2() -> Path:
    import json

    notebook = json.loads(INFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    marker = "# " + "BASELINE_INFERENCE_PART"
    parts = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            parts.append(src)
    if len(parts) != 4:
        raise RuntimeError(f"공식 추론 노트북 셀 개수가 예상과 다릅니다: {len(parts)}")

    common, stage1, stage2, _official_stage3 = parts
    stage3 = STAGE3_SNIPPET.read_text(encoding="utf-8")

    inference_source = "\n\n".join(part.rstrip() for part in [common, stage1, stage2, stage3]) + "\n"

    tree = ast.parse(inference_source, filename="inference.py")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {"predict_stage1", "predict_stage2", "predict_stage3"}
    missing = sorted(required - defined)
    if missing:
        raise RuntimeError(f"필수 함수 누락: {missing}")

    out_path = ROOT / "inference_v2.py"
    out_path.write_text(inference_source, encoding="utf-8")
    print("생성 완료:", out_path)
    print("확인된 함수:", sorted(required))
    return out_path


def stage_model_files() -> Path:
    model_dir = ROOT / "model_v2"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    for stage in ("stage1", "stage2"):
        src = ROOT / "model" / stage
        dst = model_dir / stage
        if not src.is_dir():
            raise FileNotFoundError(f"{src} 없음 - run_baseline_pipeline.py를 먼저 실행하세요")
        shutil.copytree(src, dst)

    if not STAGE3_CHECKPOINT.is_file():
        raise FileNotFoundError(f"{STAGE3_CHECKPOINT} 없음 - EXP-S3-BASE-005 학습을 먼저 실행하세요")
    (model_dir / "stage3").mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAGE3_CHECKPOINT, model_dir / "stage3" / "best.pt")
    print("모델 파일 준비 완료:", model_dir)
    return model_dir


def smoke_test(inference_path: Path, model_dir: Path) -> None:
    # 주의: importlib.util 동적 로딩은 Windows spawn 기반 multiprocessing
    # (predict_stage1의 DataLoader num_workers=4)에서 PicklingError를 낸다
    # (run_baseline_pipeline.py에서 이미 겪은 문제 - Linux 평가서버에서는 fork라
    # 문제 없음). 여기서는 파일로 저장된 inference_v2.py를 이름으로 정상 import한다.
    import importlib

    sys.modules.pop("inference_v2", None)
    module = importlib.import_module("inference_v2")
    importlib.reload(module)

    smoke_dir = ROOT / "sample_evaluation_data"
    if not smoke_dir.exists():
        raise FileNotFoundError(f"{smoke_dir} 없음 - run_baseline_pipeline.py의 smoke-test 생성 단계를 먼저 실행하세요")

    out_dir = ROOT / "output" / "submission_v2_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_pred = module.predict_stage1(smoke_dir / "stage1", model_dir / "stage1")
    stage2_pred = module.predict_stage2(smoke_dir / "stage2", model_dir / "stage2")
    stage3_pred = module.predict_stage3(smoke_dir / "stage3", model_dir / "stage3")

    for name, df in [("stage1", stage1_pred), ("stage2", stage2_pred), ("stage3", stage3_pred)]:
        path = out_dir / f"{name}_submission.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df):,}행 -> {path}")
        print(df.head())


def build_zip(inference_path: Path, model_dir: Path) -> None:
    submit_path = ROOT / "submit_v2.zip"
    inference_source = inference_path.read_text(encoding="utf-8")

    if submit_path.exists():
        submit_path.unlink()

    with zipfile.ZipFile(submit_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("inference.py", inference_source)
        archive.write(ROOT / "requirements.txt", "requirements.txt")
        for model_path in sorted(model_dir.rglob("*")):
            if model_path.is_file():
                archive.write(model_path, ("model/" + model_path.relative_to(model_dir).as_posix()))

    with zipfile.ZipFile(submit_path) as archive:
        names = archive.namelist()
    required = {
        "inference.py",
        "requirements.txt",
        "model/stage1/best.pt",
        "model/stage2/best.pt",
        "model/stage2/resnet18-f37072fd.pth",
        "model/stage3/best.pt",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"제출 ZIP 필수 파일 누락: {missing}")

    print(f"생성 완료: {submit_path}")
    print(f"압축 크기: {submit_path.stat().st_size / 1024**3:.3f} GB")
    for name in names:
        print(" -", name)


if __name__ == "__main__":
    inference_path = build_inference_v2()
    model_dir = stage_model_files()
    smoke_test(inference_path, model_dir)
    build_zip(inference_path, model_dir)
