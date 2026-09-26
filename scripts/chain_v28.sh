#!/usr/bin/env bash
# v27 업로드 완료(DONE) 후 v28 업로드 워커 2개(정순/역순) 실행
cd /c/projects/BLACKBOX_LB || exit 1
until grep -q DONE output/parts_v27/upload.log 2>/dev/null || grep -q DONE output/parts_v27/upload_rev.log 2>/dev/null; do sleep 60; done
bash scripts/upload_parts.sh v28 /c/projects/BLACKBOX_LB/submit_v28.zip 72f85df29fcaf376d3ba5847afb8327c > output/parts_v28/upload.log 2>&1 &
sleep 20
bash scripts/upload_parts_rev.sh v28 /c/projects/BLACKBOX_LB/submit_v28.zip 72f85df29fcaf376d3ba5847afb8327c > output/parts_v28/upload_rev.log 2>&1
wait
