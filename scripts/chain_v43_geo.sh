#!/usr/bin/env bash
# v43 = v33 + Stage2 기하 혼합(배포 11 + 물리+기하 3-seed, w=0.5, YOLOv8s stride 검출) + side 하이브리드·evasion −0.5.
# 이어서 v42(v33 + head, 혼합 없음) 빌드·업로드, 마지막으로 v41 업로드 재개.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd /c/projects/BLACKBOX_LB || exit 1
GH=.tools/bin/gh.exe; REPO=yoomstergotcha/blackbox-236753-experiments; PY=.venv/Scripts/python.exe
S3="--stage3-ckpt output/exp_s3_motion_007_s20260825/best.pt --stage3-extra-ckpt output/exp_s3_motion_007_s1/best.pt output/exp_s3_motion_007_s2/best.pt --stage3-snippet src/train/predict_stage3_motion_v33.py"
S1="--stage1-mode synth --stage1-ckpt output/exp_s1_synth_006/best.pt --stage1-snippet src/train/predict_stage1_vlm.py"
VLM="data/external/hf_models/Qwen2-VL-2B-Instruct=qwen2vl"

# 체크포인트 스테이징: m* 배포 55 + g* 기하 15 (n* 제외)
CK=output/s2_ens_v43; rm -rf "$CK"; mkdir -p "$CK"; cp output/s2_ens_v26/m*.pt "$CK"/
i=0; for d in output/exp_s2_geo_s_001_s20260825 output/exp_s2_geo_s_001_s1 output/exp_s2_geo_s_001_s2; do i=$((i+1)); for k in 0 1 2 3 4; do cp "$d/fold$k.pt" "$CK/g$(printf %02d $i)_geo_s_fold$k.pt"; done; done
echo "staged $(ls "$CK" | wc -l) ckpts (geo $(ls "$CK" | grep -c '^g'))"

build_upload() {  # TAG SNIPPET CKGLOB REL TITLE
  local TAG=$1 SNIP=$2 CKG=$3 REL=$4 TITLE=$5
  echo "$(date +%H:%M:%S) build $TAG"
  $PY build_submission_v3.py --tag "$TAG" $S1 --stage2-snippet "$SNIP" --stage2-extra $CKG data/external/hf_models/yolo/yolov8s.pt --stage2-extra-dir "$VLM" $S3 > "output/build_$TAG.log" 2>&1
  [ -f "submit_$TAG.zip" ] || { echo "BUILD FAIL $TAG"; tail -n 20 "output/build_$TAG.log"; return 1; }
  cmp -s output/submission_v33_smoke/stage3_submission.csv "output/submission_${TAG}_smoke/stage3_submission.csv" && echo "$TAG stage3 smoke == v33" || echo "$TAG stage3 smoke != v33"
  echo "$TAG stage2 smoke:"; cat "output/submission_${TAG}_smoke/stage2_submission.csv"
  local MD5; MD5=$(md5sum "submit_$TAG.zip" | cut -c1-32); echo "$TAG md5 $MD5"
  mkdir -p "output/parts_$TAG"; rm -f "output/parts_$TAG/submit_$TAG.zip.part"*; split -b 8m -d -a 3 "submit_$TAG.zip" "output/parts_$TAG/submit_$TAG.zip.part"; echo "$TAG parts $(ls "output/parts_$TAG" | grep -c part)"
  "$GH" release view "$REL" --repo "$REPO" >/dev/null 2>&1 || "$GH" release create "$REL" --repo "$REPO" --title "$TITLE" --notes "8MB parts: cat submit_vNN.zip.part* > submit_vNN.zip ; md5sum -c MD5SUMS.txt" 2>&1 | tail -1
  REL=$REL bash scripts/upload_parts.sh "$TAG" "/c/projects/BLACKBOX_LB/submit_$TAG.zip" "$MD5" > "output/parts_$TAG/upload.log" 2>&1
  grep -E "on release|DONE" "output/parts_$TAG/upload.log"
}

build_upload v43 src/train/predict_stage2_fixed3_vlm_geo.py "output/s2_ens_v43/*.pt" submissions-2026-09-29c "Submissions 2026-09-29 (c) (v43)"
build_upload v42 src/train/predict_stage2_fixed3_vlm_mix_head.py "output/s2_ens_v26/*.pt" submissions-2026-09-29b "Submissions 2026-09-29 (b) (v42)"
echo "$(date +%H:%M:%S) resume v41 upload"
REL=submissions-2026-09-29 bash scripts/upload_parts.sh v41 /c/projects/BLACKBOX_LB/submit_v41.zip 94471307b54e18fa6f2e534b5ffe9ac5 > output/parts_v41/upload2.log 2>&1
grep -E "on release|DONE" output/parts_v41/upload2.log
echo "CHAIN43_DONE $(date +%H:%M:%S)"
