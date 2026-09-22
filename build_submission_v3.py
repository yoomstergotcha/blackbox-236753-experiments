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

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage1-ckpt", type=Path, default=ROOT / "output" / "exp_s1_synth_005" / "best.pt")
_ap.add_argument("--tag", default="v3", help="산출물 접미사: inference_<tag>.py, model_<tag>/, submit_<tag>.zip")
_ap.add_argument("--stage1-mode", choices=["synth", "baseline"], default="synth", help="baseline=공식 Stage1 코드/모델 그대로(model_v2/stage1)")
_ap.add_argument("--stage3-ckpt", type=Path, default=None, help="지정하면 Stage3 best.pt 교체(기본: 제출2와 동일 model_v2/stage3)")
_ap.add_argument("--stage1-snippet", type=Path, default=None, help="Stage1 추론 snippet 교체(기본 predict_stage1_synth.py)")
_ap.add_argument("--stage2-snippet", type=Path, default=None, help="Stage2 추론 snippet 교체(기본: 공식 baseline 셀)")
_ap.add_argument("--stage3-extra-ckpt", type=Path, nargs="*", default=[], help="Stage3 앙상블 멤버 best.pt들 (model/stage3/best_1.pt ...로 복사)")
_ap.add_argument("--stage2-extra", type=Path, nargs="*", default=[], help="model/stage2/ 에 추가 복사할 파일(학습 localizer fold*.pt 등)")
_ap.add_argument("--stage2-extra-dir", type=Path, nargs="*", default=[], help="model/stage2/<dirname>/ 로 통째로 복사할 디렉터리(VLM 스냅샷 등)")
_ap.add_argument("--stage3-snippet", type=Path, default=None, help="Stage3 추론 snippet 교체(기본 predict_stage3_comma2k19.py)")
_args = _ap.parse_args()
STAGE1_CHECKPOINT = _args.stage1_ckpt
TAG = _args.tag
INFERENCE_NOTEBOOK = ROOT / "[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb"
STAGE1_SNIPPET = _args.stage1_snippet or (ROOT / "src" / "train" / "predict_stage1_synth.py")
STAGE3_SNIPPET = _args.stage3_snippet or (ROOT / "src" / "train" / "predict_stage3_comma2k19.py")


def build_inference() -> Path:
    notebook = json.loads(INFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    marker = "# " + "BASELINE_INFERENCE_PART"
    parts = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code" and marker in "".join(c.get("source", []))]
    if len(parts) != 4:
        raise RuntimeError(f"공식 추론 노트북 셀 개수가 예상과 다릅니다: {len(parts)}")
    common, official_stage1, stage2, _official_stage3 = parts
    stage1_src = official_stage1 if _args.stage1_mode == "baseline" else STAGE1_SNIPPET.read_text(encoding="utf-8")
    stage2_src = _args.stage2_snippet.read_text(encoding="utf-8") if _args.stage2_snippet is not None else stage2
    source = "\n\n".join(p.rstrip() for p in [common, stage1_src, stage2_src, STAGE3_SNIPPET.read_text(encoding="utf-8")]) + "\n"
    tree = ast.parse(source, filename="inference.py")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted({"predict_stage1", "predict_stage2", "predict_stage3"} - defined)
    if missing:
        raise RuntimeError(f"필수 함수 누락: {missing}")
    out = ROOT / f"inference_{TAG}.py"
    out.write_text(source, encoding="utf-8")
    print("생성 완료:", out)
    return out


def stage_model_files() -> Path:
    model_dir = ROOT / f"model_{TAG}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    src_v2 = ROOT / "model_v2"
    for stage in ("stage2", "stage3"):
        if not (src_v2 / stage).is_dir():
            raise FileNotFoundError(f"{src_v2 / stage} 없음 - build_submission_v2.py를 먼저 실행하세요")
        shutil.copytree(src_v2 / stage, model_dir / stage)
    if _args.stage3_ckpt is not None:
        shutil.copy2(_args.stage3_ckpt, model_dir / "stage3" / "best.pt")
    for i, extra in enumerate(_args.stage3_extra_ckpt, 1):
        shutil.copy2(extra, model_dir / "stage3" / f"best_{i}.pt")
    for extra in _args.stage2_extra:
        shutil.copy2(extra, model_dir / "stage2" / extra.name)
    for d in _args.stage2_extra_dir:
        shutil.copytree(d, model_dir / "stage2" / d.name, ignore=shutil.ignore_patterns(".cache", "*.md", "*.incomplete"))
    if _args.stage1_mode == "baseline":
        shutil.copytree(src_v2 / "stage1", model_dir / "stage1")
    else:
        if not STAGE1_CHECKPOINT.is_file():
            raise FileNotFoundError(f"{STAGE1_CHECKPOINT} 없음 - train_stage1을 먼저 실행하세요")
        (model_dir / "stage1").mkdir(parents=True)
        shutil.copy2(STAGE1_CHECKPOINT, model_dir / "stage1" / "best.pt")
        thr = STAGE1_CHECKPOINT.parent / "threshold.json"
        if thr.is_file():
            shutil.copy2(thr, model_dir / "stage1" / "threshold.json")
    print("모델 파일 준비 완료:", model_dir)
    return model_dir


def smoke_test(model_dir: Path) -> None:
    import importlib

    sys.modules.pop(f"inference_{TAG}", None)
    module = importlib.import_module(f"inference_{TAG}")
    smoke_dir = ROOT / "sample_evaluation_data"
    out_dir = ROOT / "output" / f"submission_{TAG}_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn, sub in [("stage1", module.predict_stage1, "stage1"), ("stage2", module.predict_stage2, "stage2"), ("stage3", module.predict_stage3, "stage3")]:
        df = fn(smoke_dir / sub, model_dir / sub)
        df.to_csv(out_dir / f"{name}_submission.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df):,}행")
        print(df.head(10) if name == "stage1" else df.head(3))


def build_zip(inference_path: Path, model_dir: Path) -> None:
    submit_path = ROOT / f"submit_{TAG}.zip"
    if submit_path.exists():
        submit_path.unlink()
    with zipfile.ZipFile(submit_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("inference.py", inference_path.read_text(encoding="utf-8"))
        archive.write(ROOT / "requirements.txt", "requirements.txt")
        for p in sorted(model_dir.rglob("*")):
            if p.is_file():
                big = p.stat().st_size > 64 * 1024**2  # 대형 가중치(safetensors 등)는 압축 생략(시간 절약, 압축률 거의 0)
                archive.write(p, "model/" + p.relative_to(model_dir).as_posix(), compress_type=zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED)
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
