#!/usr/bin/env bash
# v28 업로드 완료 후 v31 → v32 (릴리스 submissions-2026-09-27)
cd /c/projects/BLACKBOX_LB || exit 1
until grep -q DONE output/parts_v28/upload.log 2>/dev/null; do sleep 60; done
export REL=submissions-2026-09-27
bash scripts/upload_parts.sh v31 /c/projects/BLACKBOX_LB/submit_v31.zip bd56bfb4be84ab24ee5a8a6c2cf78a1d > output/parts_v31/upload.log 2>&1
bash scripts/upload_parts.sh v32 /c/projects/BLACKBOX_LB/submit_v32.zip 5cb680ec87d4b1fcaff120b606d62345 > output/parts_v32/upload.log 2>&1
echo CHAIN_DONE
