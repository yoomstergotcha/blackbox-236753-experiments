"""Route 단위 group split — 같은 route(`<device-hash>|<datetime>`)의 segment가
train/validation에 동시에 들어가지 않도록 보장한다 (EXPERIMENT_DESIGN.md §9-6, 가이드 §22
Stage3 leakage 주의사항).
"""
from __future__ import annotations

import random


def route_id(segment_id: str) -> str:
    return segment_id.rsplit("/", 1)[0]


def group_train_val_split(
    segment_ids: list[str], val_ratio: float = 0.2, seed: int = 20260825
) -> tuple[list[str], list[str]]:
    """route 단위로 묶어서 val_ratio에 가깝게 train/val로 나눈다.

    같은 route의 모든 segment는 항상 같은 split에 들어간다 — segment 단위로 나누지 않는다.
    """
    routes: dict[str, list[str]] = {}
    for seg in segment_ids:
        routes.setdefault(route_id(seg), []).append(seg)

    route_list = sorted(routes.keys())
    rng = random.Random(seed)
    rng.shuffle(route_list)

    target_val = round(len(segment_ids) * val_ratio)
    val_segments: list[str] = []
    val_routes: set[str] = set()
    for route in route_list:
        if len(val_segments) >= target_val:
            break
        val_segments.extend(routes[route])
        val_routes.add(route)

    train_segments = [seg for seg in segment_ids if route_id(seg) not in val_routes]
    return train_segments, val_segments


def assert_no_route_leakage(train_segments: list[str], val_segments: list[str]) -> None:
    train_routes = {route_id(s) for s in train_segments}
    val_routes = {route_id(s) for s in val_segments}
    overlap = train_routes & val_routes
    if overlap:
        raise AssertionError(f"route leakage 발견: {overlap}")


def _self_test() -> None:
    segments = [f"routeA/{i}" for i in range(3)] + [f"routeB/{i}" for i in range(2)] + [
        f"routeC/{i}" for i in range(5)
    ]
    train, val = group_train_val_split(segments, val_ratio=0.3, seed=1)
    assert_no_route_leakage(train, val)
    assert set(train) | set(val) == set(segments)
    assert set(train) & set(val) == set()
    print("split self-test 통과:", len(train), "train /", len(val), "val")


if __name__ == "__main__":
    _self_test()
