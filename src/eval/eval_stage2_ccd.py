"""EXP-S2-CCD-001 — CCD(Car Crash Dataset, MIT) 1,500 클립으로 Stage2 충돌 검출 휴리스틱을 검증·튜닝한다.

정책(§21): 비공개 Stage2가 CCD 클립일 가능성이 있어 클립별 라벨로 모델을 *학습하지 않는다*. 여기서는
  (1) 현재 휴리스틱(predict_stage2_heuristic.py)의 ±0.3s(±3프레임) 적중률을 ego 관여 클립에서 측정하고,
  (2) 위치 사전정보(후반부 검색 등, 제출 12에서 LB -0.075)를 제외한 **스케일 불변 파라미터**(z 임계, 신호 조합,
      gap)만 소수 격자 탐색한다.
신호(shift/vert/diff)는 클립당 한 번 계산해 캐시한다(output/ccd_signals.npz).

실행:
    python -m src.eval.eval_stage2_ccd --videos data/external/ccd --ann data/external/ccd/Crash-1500.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_ann(path: Path) -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"(\d+),\[(.*?)\],(\d+),(\S+),(\w+),(\w+),(\w+)", line.strip())
        if not m:
            continue
        lab = np.array([int(x) for x in m.group(2).split(",")])
        rows.append({"vid": m.group(1), "onset": int(np.argmax(lab)) if lab.max() > 0 else -1, "ego": m.group(7) == "Yes", "timing": m.group(5), "weather": m.group(6)})
    return pd.DataFrame(rows)


def signals_for(video: Path, g):
    cap = cv2.VideoCapture(str(video))
    gray = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32))
    cap.release()
    if len(gray) < 3:
        return None
    shift, vert, diff = g["_s2_signals"](gray)
    return np.stack([shift, vert, diff])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("data/external/ccd"))
    ap.add_argument("--ann", type=Path, default=Path("data/external/ccd/Crash-1500.txt"))
    ap.add_argument("--cache", type=Path, default=Path("output/ccd_signals.npz"))
    ap.add_argument("--snippet", type=Path, default=Path("src/train/predict_stage2_heuristic.py"))
    args = ap.parse_args()
    g = {}
    exec(open(args.snippet, encoding="utf-8").read(), g)
    ann = load_ann(args.ann)
    vid2path = {p.stem: p for p in args.videos.rglob("*.mp4")}
    if args.cache.is_file():
        cache = dict(np.load(args.cache, allow_pickle=True))
    else:
        cache = {}
        for k, vid in enumerate(ann.vid):
            p = vid2path.get(vid)
            if p is None:
                continue
            s = signals_for(p, g)
            if s is not None:
                cache[vid] = s
            if (k + 1) % 100 == 0:
                print(f"signals {k + 1}/{len(ann)}", flush=True)
        np.savez(args.cache, **cache)
    ann = ann[ann.vid.isin(cache)]
    print(f"{len(ann)} clips with signals | ego {int(ann.ego.sum())} | public 5 excluded from tuning")
    tune = ann[ann.ego & ~ann.vid.isin({"000001", "000002", "000003", "000004", "000005"})]

    z = g["_s2_robust_z"]

    def detect(sig, tau, gap, mode):
        zs, zv, zd = z(sig[0]), z(sig[1]), z(sig[2])
        n = len(zv)
        base = {"vert": zv, "max3": np.maximum.reduce([zs, zv, zd]), "vert+diff": np.maximum(zv, zd), "sum3": zs + zv + zd}[mode]
        if base.max() < tau:
            comb = np.maximum.reduce([zs, zv, zd])
            return int(np.argmax(comb))
        i = int(np.argmax(base))
        while True:
            lo = max(i - gap, 1)
            prev = [j for j in range(lo, i) if base[j] > tau]
            if not prev:
                return i
            i = prev[0]

    def hit_rate(df, tau, gap, mode, tol=3):
        hits = [abs(detect(cache[v], tau, gap, mode) - o) <= tol for v, o in zip(df.vid, df.onset)]
        return float(np.mean(hits))

    print("현재(HEUR-001: vert, tau 5, gap 5):", f"ego {hit_rate(tune, 5.0, 5, 'vert'):.3f} | all {hit_rate(ann, 5.0, 5, 'vert'):.3f}")
    # 오차 분포(현재 설정)
    err = np.array([detect(cache[v], 5.0, 5, "vert") - o for v, o in zip(tune.vid, tune.onset)])
    print("오차(pred-onset) 분위:", {q: int(np.percentile(err, q)) for q in (5, 25, 50, 75, 95)}, "| |err|<=3:", f"{np.mean(np.abs(err) <= 3):.3f}", "| 조기(<-3):", f"{np.mean(err < -3):.3f}", "| 지연(>3):", f"{np.mean(err > 3):.3f}")
    rows = []
    for mode in ("vert", "max3", "vert+diff", "sum3"):
        for tau in (3.0, 4.0, 5.0, 6.0, 8.0):
            for gap in (3, 5, 8):
                rows.append((mode, tau, gap, hit_rate(tune, tau, gap, mode)))
    res = pd.DataFrame(rows, columns=["mode", "tau", "gap", "hit"]).sort_values("hit", ascending=False)
    print(res.head(12).to_string(index=False))
    best = res.iloc[0]
    print("best:", dict(best), "| all clips:", f"{hit_rate(ann, best.tau, int(best.gap), best['mode']):.3f}")
    # 전체 프레임 중 정답 위치 분포(참고): 사전정보로 쓰지 않음
    print("onset 분위(참고):", {q: int(np.percentile(tune.onset, q)) for q in (5, 50, 95)})


if __name__ == "__main__":
    main()
