#!/usr/bin/env bash
# v33 업로드 완료 후 v34를 릴리스 submissions-2026-09-28b 에 단일 워커로 업로드
cd /c/projects/BLACKBOX_LB || exit 1
until grep -q DONE output/parts_v33/upload.log 2>/dev/null; do sleep 60; done
REL=submissions-2026-09-28b bash scripts/upload_parts.sh v34 /c/projects/BLACKBOX_LB/submit_v34.zip c30e81929e2f2c267342549cd9c2a283 > output/parts_v34/upload.log 2>&1
echo CHAIN_DONE
