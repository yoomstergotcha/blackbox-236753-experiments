#!/usr/bin/env bash
# upload_parts.sh 의 역순 변형: 조각을 뒤에서부터 올린다(정순 워커와 동시에 돌려 병렬 업로드).
# 사용: bash scripts/upload_parts_rev.sh <tag> <zip 경로> <md5>
set -u
TAG=$1; ZIP=$2; MD5=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd); GH="$ROOT/.tools/bin/gh.exe"; REPO=yoomstergotcha/blackbox-236753-experiments; REL=${REL:-submissions-2026-09-25}
DIR="$ROOT/output/parts_$TAG"; cd "$DIR" || exit 1
for round in 1 2 3 4 5; do
  "$GH" release view "$REL" --repo "$REPO" --json assets -q '.assets[] | "\(.name) \(.size)"' > "$DIR/.assets_rev" 2>/dev/null || { sleep 30; continue; }
  missing=0
  for f in $(ls -r submit_${TAG}.zip.part*); do
    sz=$(stat -c %s "$f")
    grep -q "^$f $sz$" "$DIR/.assets_rev" && continue
    missing=$((missing+1))
    for t in 1 2 3 4 5 6; do "$GH" release upload "$REL" "$f" --repo "$REPO" --clobber >/dev/null 2>&1 && { echo "$(date +%H:%M:%S) ok $f"; break; } || sleep 10; done
  done
  echo "round $round: uploaded/retried $missing"
  [ "$missing" = 0 ] && break
done
"$GH" release download "$REL" --repo "$REPO" --pattern MD5SUMS.txt -O MD5SUMS_rev.txt --clobber 2>/dev/null
grep -q "submit_${TAG}.zip" MD5SUMS_rev.txt || { echo "submit_${TAG}.zip $MD5" >> MD5SUMS_rev.txt; cp MD5SUMS_rev.txt MD5SUMS.txt; for t in 1 2 3 4 5; do "$GH" release upload "$REL" MD5SUMS.txt --repo "$REPO" --clobber >/dev/null 2>&1 && { echo "uploaded MD5SUMS.txt"; break; } || sleep 8; done; }
"$GH" release view "$REL" --repo "$REPO" --json assets -q ".assets[] | select(.name | startswith(\"submit_${TAG}.zip\")) | .size" | awk -v z=$(stat -c %s "$ZIP") '{n++; s+=$1} END {print "'"$TAG"' on release:", n, "parts,", s, "bytes, zip", z, (s==z ? "MATCH" : "MISMATCH")}'
echo "DONE"
