"""CCD onset vs 첫 물리 접촉(first physical contact) 감사용 몽타주: ego 클립 N개, onset−4..onset+4 연속 9프레임(1프레임 단위), 4클립/장.
라벨: 첫 접촉 프레임의 onset 대비 오프셋(정수), 판단 불가면 공란. 실행: python -m src.data.stage2_contact_audit_montage --n 100 --out output/ccd_contact_audit_v1
"""
import argparse, sys, cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann

OFFS = list(range(-4, 5)); TW, TH = 213, 120


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", type=Path, default=Path("output/ccd_contact_audit_v1")); ap.add_argument("--seed", type=int, default=5); a = ap.parse_args()
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.ego & (ann.onset >= 4) & (ann.onset <= 45)]
    public = {"000001", "000002", "000003", "000004", "000005"}; pool = ann[~ann.vid.isin(public)].reset_index(drop=True)
    pick = pool.sample(n=min(a.n, len(pool)), random_state=a.seed).reset_index(drop=True); a.out.mkdir(parents=True, exist_ok=True); rows = []
    for m in range(0, len(pick), 4):
        grid = []
        for r in range(m, min(m + 4, len(pick))):
            vid, onset = pick.vid[r], int(pick.onset[r]); cap = cv2.VideoCapture(f"data/external/ccd/{vid}.mp4"); fr = []
            while True:
                ok, b = cap.read()
                if not ok: break
                fr.append(b)
            tiles = []
            for o in OFFS:
                j = int(np.clip(onset + o, 0, len(fr) - 1)); t = cv2.resize(fr[j], (TW, TH)); s = f"{'+' if o >= 0 else ''}{o}"
                cv2.putText(t, s, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(t, s, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if o else (0, 0, 255), 1)
                tiles.append(t)
            row = np.concatenate(tiles, 1); cv2.putText(row, f"row{r - m + 1} {vid}", (4, TH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3); cv2.putText(row, f"row{r - m + 1} {vid}", (4, TH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            grid.append(row); rows.append(dict(montage=f"m{m // 4:03d}", row=r - m + 1, vid=vid, onset=onset))
        cv2.imwrite(str(a.out / f"m{m // 4:03d}.jpg"), np.concatenate(grid, 0), [cv2.IMWRITE_JPEG_QUALITY, 82])
    pd.DataFrame(rows).to_csv(a.out / "index.csv", index=False); print("montages", (len(pick) + 3) // 4, "clips", len(pick), "->", a.out)


if __name__ == "__main__":
    main()
