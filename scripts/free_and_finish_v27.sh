#!/usr/bin/env bash
# 릴리스 submissions-2026-09-25 에서 v25 조각(557개)을 삭제해 자산 한도(1000)를 비운 뒤 v27 업로드를 재개한다.
cd /c/projects/BLACKBOX_LB || exit 1
GH=.tools/bin/gh.exe; REPO=yoomstergotcha/blackbox-236753-experiments; REL=submissions-2026-09-25
$GH release view $REL --repo $REPO --json assets -q '.assets[] | select(.name | startswith("submit_v25.zip.part")) | .name' > output/parts_v25/.to_delete
n=$(wc -l < output/parts_v25/.to_delete); echo "deleting $n v25 assets"; i=0
while read -r a; do i=$((i+1)); for t in 1 2 3 4 5; do $GH release delete-asset $REL "$a" --repo $REPO -y >/dev/null 2>&1 && break || sleep 5; done; [ $((i % 50)) = 0 ] && echo "deleted $i/$n"; done < output/parts_v25/.to_delete
echo "delete done $(date +%H:%M:%S)"
REL=$REL bash scripts/upload_parts.sh v27 /c/projects/BLACKBOX_LB/submit_v27.zip 7192ec6d6ec52affa1840274ad457f93
