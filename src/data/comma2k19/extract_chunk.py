"""comma2k19 raw_data Chunk_N.zip에서 필요한 segment만 골라 추출한다.

Chunk 전체(~9GB, ~200개 segment)를 다 풀 필요는 없다 — route 다양성과 class 다양성
확보에 필요한 만큼만 골라서 추출한다.

사용 예:
    # 1) zip 안에 어떤 route/segment가 있는지만 확인
    python -m src.data.comma2k19.extract_chunk --zip data/external/comma2k19_chunks/Chunk_1.zip --list

    # 2) 실제 추출 (route마다 최대 N개 segment)
    python -m src.data.comma2k19.extract_chunk --zip data/external/comma2k19_chunks/Chunk_1.zip \
        --out data/external/comma2k19_chunk1_subset --max-per-route 3 --max-routes 12
"""
from __future__ import annotations

import argparse
import re
import zipfile
from collections import defaultdict
from pathlib import Path

_SEGMENT_RE = re.compile(r"([^/]+\|[0-9-]+--[0-9-]+)/(\d+)/")


def list_segments(zip_path: Path) -> dict[str, list[str]]:
    """zip 안의 segment를 route별로 묶어서 반환: {route_id: [segment_id, ...]}"""
    routes: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _SEGMENT_RE.search(name)
            if m:
                route, seg_num = m.group(1), m.group(2)
                routes[route].add(f"{route}/{seg_num}")
    return {route: sorted(segs, key=lambda s: int(s.rsplit("/", 1)[1])) for route, segs in routes.items()}


def select_segments(routes: dict[str, list[str]], max_per_route: int, max_routes: int) -> list[str]:
    selected = []
    for route in sorted(routes)[:max_routes]:
        segs = routes[route]
        # route 안에서도 앞/중간/뒤로 퍼지게 골라 다양한 국면(가속/정지 등)을 잡을 확률을 높인다.
        if len(segs) <= max_per_route:
            picked = segs
        else:
            step = len(segs) / max_per_route
            picked = [segs[int(i * step)] for i in range(max_per_route)]
        selected.extend(picked)
    return selected


def extract_segments(zip_path: Path, out_dir: Path, segment_ids: list[str]) -> None:
    """segment_id는 "<route>/<segnum>" 형태. zip 안 실제 경로는 "Chunk_N/<route>/<segnum>/..."
    처럼 앞에 chunk 폴더가 붙어 있을 수 있어 끝부분 일치로 찾는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_markers = tuple(f"/{seg}/" for seg in segment_ids)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if any(marker in f"/{n}" for marker in wanted_markers)]
        print(f"{len(members)}개 파일 추출 중 ({len(segment_ids)}개 segment)...")
        zf.extractall(out_dir, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(description="comma2k19 Chunk zip에서 필요한 segment만 추출")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="추출 없이 route/segment 목록만 출력")
    parser.add_argument("--max-per-route", type=int, default=3)
    parser.add_argument("--max-routes", type=int, default=12)
    args = parser.parse_args()

    routes = list_segments(args.zip)
    total_segments = sum(len(v) for v in routes.values())
    print(f"이 zip: route {len(routes)}개, segment {total_segments}개")

    if args.list:
        for route, segs in sorted(routes.items()):
            print(f"  {route}: {len(segs)}개 segment")
        return

    if args.out is None:
        parser.error("--out이 필요합니다 (--list가 아니면)")

    selected = select_segments(routes, args.max_per_route, args.max_routes)
    print(f"선택된 segment {len(selected)}개 (route {min(args.max_routes, len(routes))}개 x 최대 {args.max_per_route}개):")
    for s in selected:
        print(" ", s)
    extract_segments(args.zip, args.out, selected)
    print(f"완료: {args.out}")


if __name__ == "__main__":
    main()
