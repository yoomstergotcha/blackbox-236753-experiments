"""검출 캐시(det dir) → 프레임별 접촉 기하 특징 'geo'(24-d)를 기존 특징 npz에 추가한 새 캐시 디렉토리.
실행: python -m src.data.stage2_geo_cache --det output/stage2_ccd_det_m --out output/stage2_ccd_feats_geo
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from src.eval.stage2_geometry import geometry_features, per_frame_from_det, GEO_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("output/stage2_ccd_feats"))
    ap.add_argument("--det", type=Path, default=Path("output/stage2_ccd_det_m"))
    ap.add_argument("--out", type=Path, default=Path("output/stage2_ccd_feats_geo"))
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); miss = 0
    for p in sorted(a.src.glob("*.npz")):
        z = dict(np.load(p)); n = z["signals"].shape[0]; dp = a.det / p.name
        if dp.is_file():
            z["geo"] = geometry_features(per_frame_from_det(np.load(dp)["det"], n, a.conf))
        else:
            miss += 1; z["geo"] = np.zeros((n, GEO_DIM), np.float32)
        np.savez(a.out / p.name, **z)
    print("done", len(list(a.out.glob("*.npz"))), "| det missing", miss, "->", a.out)


if __name__ == "__main__":
    main()
