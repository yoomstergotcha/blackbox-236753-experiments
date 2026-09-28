"""수동 라벨용 몽타주: 미라벨 CCD ego 클립 N개를 골라 4클립×6프레임(onset−9,−6,−4,−2,0,+2) 이미지로 저장.
출력: <out>/mNNN.jpg, <out>/index.csv (montage, row, vid, onset). 라벨 정의(공식): entry_side = 피해차량이 화면 어느 쪽에서 내 진로로 들어왔나(LEFT/RIGHT),
evasion_space = 충돌 직전 내 차 옆에 피할 수 있는 빈 차로/공간이 있었나(1/0).
실행: python -m src.data.stage2_label_montage --n 200 --out output/ccd_label_montages_v3
"""
import argparse, sys, cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann

OFFS = (-9, -6, -4, -2, 0, 2); TW, TH = 320, 180


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", type=Path, default=Path("output/ccd_label_montages_v3")); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
    ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); ann = ann[ann.ego & (ann.onset >= 0)]
    public = {"000001", "000002", "000003", "000004", "000005"}
    lab = pd.read_csv("output/ccd_manual_labels_all.csv", dtype={"vid": str}); labeled = set(lab.vid[lab.manual_side.notna() | lab.manual_evasion.notna()])
    pool = ann[~ann.vid.isin(public | labeled)].reset_index(drop=True)
    pool = pool[(pool.onset >= 12) & (pool.onset <= 46)]
    pick = pool.sample(n=min(a.n, len(pool)), random_state=a.seed).reset_index(drop=True)
    a.out.mkdir(parents=True, exist_ok=True); rows = []
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
                j = int(np.clip(onset + o, 0, len(fr) - 1)); t = cv2.resize(fr[j], (TW, TH))
                cv2.putText(t, f"{'+' if o >= 0 else ''}{o}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(t, f"{'+' if o >= 0 else ''}{o}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                tiles.append(t)
            row = np.concatenate(tiles, 1); cv2.putText(row, f"row{r - m + 1} {vid}", (6, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(row, f"row{r - m + 1} {vid}", (6, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            grid.append(row); rows.append(dict(montage=f"m{m // 4:03d}", row=r - m + 1, vid=vid, onset=onset))
        cv2.imwrite(str(a.out / f"m{m // 4:03d}.jpg"), np.concatenate(grid, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
    pd.DataFrame(rows).to_csv(a.out / "index.csv", index=False); print("montages", (len(pick) + 3) // 4, "clips", len(pick), "->", a.out)


if __name__ == "__main__":
    main()
