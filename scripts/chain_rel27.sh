#!/usr/bin/env bash
# 새 릴리스(submissions-2026-09-27)에 v30 → v29 → v28 순서로 단일 워커 업로드
cd /c/projects/BLACKBOX_LB || exit 1
export REL=submissions-2026-09-27
bash scripts/upload_parts.sh v30 /c/projects/BLACKBOX_LB/submit_v30.zip c0b106a1d48e31e62956e75d096f2650 > output/parts_v30/upload2.log 2>&1
bash scripts/upload_parts.sh v29 /c/projects/BLACKBOX_LB/submit_v29.zip 3e144a4f22bd44e717d797158b7722ab > output/parts_v29/upload2.log 2>&1
bash scripts/upload_parts.sh v28 /c/projects/BLACKBOX_LB/submit_v28.zip 72f85df29fcaf376d3ba5847afb8327c > output/parts_v28/upload.log 2>&1
echo CHAIN_DONE
