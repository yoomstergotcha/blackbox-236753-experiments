#!/usr/bin/env bash
# v37 빌드 완료 대기 → 8MB 조각 분할 → 릴리스 submissions-2026-09-28d 생성 → 단일 워커 업로드(재개 가능)
cd /c/projects/BLACKBOX_LB || exit 1
GH=.tools/bin/gh.exe; REPO=yoomstergotcha/blackbox-236753-experiments; REL=submissions-2026-09-28d
until grep -q "생성 완료" output/build_v37.log 2>/dev/null && [ -f submit_v37.zip ]; do sleep 60; done
sleep 30
MD5=$(md5sum submit_v37.zip | cut -c1-32); echo "md5 $MD5"
mkdir -p output/parts_v37; rm -f output/parts_v37/submit_v37.zip.part*
split -b 8m -d -a 3 submit_v37.zip output/parts_v37/submit_v37.zip.part
echo "parts $(ls output/parts_v37 | grep -c part)"
"$GH" release view "$REL" --repo "$REPO" >/dev/null 2>&1 || "$GH" release create "$REL" --repo "$REPO" --title "Submissions 2026-09-28 (d) (v37)" --notes "8MB parts: cat submit_vNN.zip.part* > submit_vNN.zip ; md5sum -c MD5SUMS.txt" 2>&1 | tail -1
REL=$REL bash scripts/upload_parts.sh v37 /c/projects/BLACKBOX_LB/submit_v37.zip "$MD5" > output/parts_v37/upload.log 2>&1
echo CHAIN_DONE
