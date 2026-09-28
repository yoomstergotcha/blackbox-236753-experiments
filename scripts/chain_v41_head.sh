#!/usr/bin/env bash
# v41 = v38(v33 + Stage2 혼합 w=0.5) + side 하이브리드 규칙(YOLOv8s 트랙 횡위치 → VLM 폴백). v38~v40 체인 종료 후 빌드·분할·릴리스·업로드.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/projects/BLACKBOX_LB || exit 1
GH=.tools/bin/gh.exe; REPO=yoomstergotcha/blackbox-236753-experiments; PY=.venv/Scripts/python.exe
TAG=v41; REL=submissions-2026-09-29
S3="--stage3-ckpt output/exp_s3_motion_007_s20260825/best.pt --stage3-extra-ckpt output/exp_s3_motion_007_s1/best.pt output/exp_s3_motion_007_s2/best.pt --stage3-snippet src/train/predict_stage3_motion_v33.py"
S1="--stage1-mode synth --stage1-ckpt output/exp_s1_synth_006/best.pt --stage1-snippet src/train/predict_stage1_vlm.py"
VLM="data/external/hf_models/Qwen2-VL-2B-Instruct=qwen2vl"
until grep -q "CHAIN_DONE" output/chain_v38_40.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) build $TAG"
$PY build_submission_v3.py --tag "$TAG" $S1 --stage2-snippet src/train/predict_stage2_fixed3_vlm_mix_head.py --stage2-extra output/s2_ens_v37/*.pt data/external/hf_models/yolo/yolov8s.pt --stage2-extra-dir "$VLM" $S3 > "output/build_$TAG.log" 2>&1
[ -f "submit_$TAG.zip" ] || { echo "BUILD FAIL"; tail -n 20 "output/build_$TAG.log"; exit 1; }
cmp -s output/submission_v33_smoke/stage3_submission.csv "output/submission_${TAG}_smoke/stage3_submission.csv" && echo "$TAG stage3 smoke == v33" || echo "$TAG stage3 smoke != v33"
echo "$TAG stage2 smoke:"; cat "output/submission_${TAG}_smoke/stage2_submission.csv"
MD5=$(md5sum "submit_$TAG.zip" | cut -c1-32); echo "$TAG md5 $MD5"
mkdir -p "output/parts_$TAG"; rm -f "output/parts_$TAG/submit_$TAG.zip.part"*; split -b 8m -d -a 3 "submit_$TAG.zip" "output/parts_$TAG/submit_$TAG.zip.part"; echo "$TAG parts $(ls "output/parts_$TAG" | grep -c part)"
"$GH" release view "$REL" --repo "$REPO" >/dev/null 2>&1 || "$GH" release create "$REL" --repo "$REPO" --title "Submissions 2026-09-29 ($TAG)" --notes "8MB parts: cat submit_vNN.zip.part* > submit_vNN.zip ; md5sum -c MD5SUMS.txt" 2>&1 | tail -1
REL=$REL bash scripts/upload_parts.sh "$TAG" "/c/projects/BLACKBOX_LB/submit_$TAG.zip" "$MD5" > "output/parts_$TAG/upload.log" 2>&1
grep -E "on release|DONE" "output/parts_$TAG/upload.log"; echo "CHAIN41_DONE $(date +%H:%M:%S)"
