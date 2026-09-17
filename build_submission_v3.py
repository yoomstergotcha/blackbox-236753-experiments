"""제출 후보 3 — Stage1을 합성 재녹화로 학습한 ResNet18로 교체. Stage2는 공식 baseline,
Stage3는 제출 2(comma2k19 ResNet18-head)와 동일하게 유지해 LB 델타가 Stage1 변경분만 반영되게 한다.

전제조건: model_v2/ (build_submission_v2.py 산출), output/exp_s1_synth_001/best.pt (train_stage1).
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

STAGE1_CHECKPOINT = ROOT / "output" / "exp_s1_synth_001" / "best.pt"
INFERENCE_NOTEBOOK = ROOT / "[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb"
STAGE1_SNIPPET = ROOT / "src" / "train" / "predict_stage1_synth.py"
STAGE3_SNIPPET = ROOT / "src" / "train" / "predict_stage3_comma2k19.py"


def build_inference() -> Path:
    notebook = json.loads(INFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    marker = "# " + "BASELINE_INFERENCE_PART"
    parts = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code" and marker in "".join(c.get("source", []))]
    if len(parts) != 4:
        raise RuntimeError(f"공식 추론 노트북 셀 개수가 예상과 다릅니다: {len(parts)}")
    common, _official_stage1, stage2, _official_stage3 = parts
    source = "\n\n".join(p.rstrip() for p in [common, STAGE1_SNIPPET.read_text(encoding="utf-8"), stage2, STAGE3_SNIPPET.read_text(encoding="utf-8")]) + "\n"
    tree = ast.parse(source, filename="inference.py")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted({"predict_stage1", "predict_stage2", "predict_stage3"} - defined)
    if missing:
        raise RuntimeError(f"필수 함수 누락: {missing}")
    out = ROOT / "inference_v3.py"
    out.write_text(source, encoding="utf-8")
    print("생성 완료:", out)
    return out


def stage_model_files() -> Path:
    model_dir = ROOT / "model_v3"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    src_v2 = ROOT / "model_v2"
    for stage in ("stage2", "stage3"):
        if not (src_v2 / stage).is_dir():
            raise FileNotFoundError(f"{src_v2 / stage} 없음 - build_submission_v2.py를 먼저 실행하세요")
        shutil.copytree(src_v2 / stage, model_dir / stage)
    if not STAGE1_CHECKPOINT.is_file():
        raise FileNotFoundError(f"{STAGE1_CHECKPOINT} 없음 - train_stage1을 먼저 실행하세요")
    (model_dir / "stage1").mkdir(parents=True)
    shutil.copy2(STAGE1_CHECKPOINT, model_dir / "stage1" / "best.pt")
    print("모델 파일 준비 완료:", model_dir)
    return model_dir


def smoke_test(model_dir: Path) -> None:
    import importlib

    sys.modules.pop("inference_v3", None)
    module = importlib.import_module("inference_v3")
    smoke_dir = ROOT / "sample_evaluation_data"
    out_dir = ROOT / "output" / "submission_v3_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn, sub in [("stage1", module.predict_stage1, "stage1"), ("stage2", module.predict_stage2, "stage2"), ("stage3", module.predict_stage3, "stage3")]:
        df = fn(smoke_dir / sub, model_dir / sub)
        df.to_csv(out_dir / f"{name}_submission.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df):,}행")
        print(df.head(10) if name == "stage1" else df.head(3))


def build_zip(inference_path: Path, model_dir: Path) -> None:
    submit_path = ROOT / "submit_v3.zip"
    if submit_path.exists():
        submit_path.unlink()
    with zipfile.ZipFile(submit_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("inference.py", inference_path.read_text(encoding="utf-8"))
        archive.write(ROOT / "requirements.txt", "requirements.txt")
        for p in sorted(model_dir.rglob("*")):
            if p.is_file():
                archive.write(p, "model/" + p.relative_to(model_dir).as_posix())
    with zipfile.ZipFile(submit_path) as archive:
        names = archive.namelist()
    required = {"inference.py", "requirements.txt", "model/stage1/best.pt", "model/stage2/best.pt", "model/stage2/resnet18-f37072fd.pth", "model/stage3/best.pt"}
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"제출 ZIP 필수 파일 누락: {missing}")
    print(f"생성 완료: {submit_path} ({submit_path.stat().st_size / 1024**2:.0f} MB)")
    for n in names:
        print(" -", n)


if __name__ == "__main__":
    inference_path = build_inference()
    model_dir = stage_model_files()
    smoke_test(model_dir)
    build_zip(inference_path, model_dir)
