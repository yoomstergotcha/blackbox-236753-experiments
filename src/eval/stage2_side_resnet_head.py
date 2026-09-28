"""EXP-S2-SIDE-005 — 진입 방향을 풍부한 시각 표현으로 학습: 충돌 직전 프레임(onset−7..onset)의 좌/우 절반 ResNet18(ImageNet, 패키지 동봉) 512-d 평균 특징
+ 시간 차분 → 로지스틱 회귀/MLP, 431 라벨 5-fold. 비교: 하이브리드 규칙 0.639.
실행: python -m src.eval.stage2_side_resnet_head
"""
import glob, sys, cv2, numpy as np, pandas as pd, torch, torchvision
from pathlib import Path
from torch import nn
sys.path.insert(0, ".")
from src.train.train_stage2_collision import load_ann

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ann = load_ann(Path("data/external/ccd/Crash-1500.txt")); onset = {v: int(o) for v, o in zip(ann.vid, ann.onset) if o >= 0}
lab = pd.concat([pd.read_csv(f, dtype={"vid": str}) for f in ["output/ccd_manual_labels_all.csv"] + sorted(glob.glob("output/ccd_manual_labels_v*_part*.csv"))], ignore_index=True).drop_duplicates("vid")
lab = lab[lab.manual_side.notna() & lab.vid.isin(onset)].reset_index(drop=True); print("side labels", len(lab))
net = torchvision.models.resnet18(); net.load_state_dict(torch.load("model_v38/stage2/resnet18-f37072fd.pth", map_location="cpu")); net.fc = nn.Identity(); net = net.to(dev).eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


@torch.no_grad()
def feats(vid, c, k=8):
    cap = cv2.VideoCapture(f"data/external/ccd/{vid}.mp4"); fr = []
    while True:
        ok, b = cap.read()
        if not ok: break
        fr.append(b)
    idx = [int(np.clip(c - k + 1 + i, 0, len(fr) - 1)) for i in range(k)]
    tiles = []
    for i in idx:
        b = fr[i]; w = b.shape[1]; tiles += [prep(b[:, : w // 2]), prep(b[:, w // 2:]), prep(b)]
    f = net(torch.stack(tiles).to(dev)).cpu().numpy().reshape(k, 3, 512)  # (frame, [left,right,full], 512)
    L, R, Fu = f[:, 0], f[:, 1], f[:, 2]
    return np.concatenate([L.mean(0), R.mean(0), (L - R).mean(0), L[-1] - L[0], R[-1] - R[0], Fu.mean(0)])


cache = Path("output/s2_side_resnet_feats.npz")
if cache.is_file() and set(np.load(cache)["vids"]) >= set(lab.vid):
    z = np.load(cache); X = dict(zip(z["vids"], z["X"]))
else:
    X = {v: feats(v, onset[v]) for v in lab.vid}; np.savez_compressed(cache, vids=np.array(list(X)), X=np.stack(list(X.values())))
Xa = np.stack([X[v] for v in lab.vid]); y = (lab.manual_side == "LEFT").astype(int).to_numpy()


def mf1(y, p):
    f = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


def cv(Xs, y, model="lr", l2=1.0, seeds=(0, 1, 2), folds=5, hid=64, epochs=200):
    res = []
    for seed in seeds:
        rng = np.random.default_rng(seed); order = rng.permutation(len(y)); fold = np.zeros(len(y), int); fold[order] = np.arange(len(y)) % folds; p = np.zeros(len(y))
        for k in range(folds):
            tr, te = fold != k, fold == k; mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6; Xt = torch.from_numpy((Xs - mu) / sd).float().to(dev); yt = torch.from_numpy(y).float().to(dev)
            torch.manual_seed(seed); m = (nn.Linear(Xs.shape[1], 1) if model == "lr" else nn.Sequential(nn.Linear(Xs.shape[1], hid), nn.GELU(), nn.Dropout(0.5), nn.Linear(hid, 1))).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=l2); w = torch.tensor([(y[tr] == 0).mean() / (y[tr] == 1).mean()], device=dev)  # LEFT 다수 → 가중
            for _ in range(epochs):
                m.train(); lg = m(Xt[tr]).squeeze(-1); loss = nn.functional.binary_cross_entropy_with_logits(lg, yt[tr], pos_weight=1 / w); opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): p[te] = torch.sigmoid(m(Xt[te]).squeeze(-1)).cpu().numpy()
        res.append((mf1(y, (p > 0.5).astype(int)), ((p > 0.5).astype(int) == y).mean(), (p > 0.5).mean()))
    return np.mean([r[0] for r in res]), np.std([r[0] for r in res]), np.mean([r[1] for r in res]), np.mean([r[2] for r in res])


blocks = {"LR mean": slice(0, 512), "L-R diff": slice(1024, 1536), "L,R,L-R": slice(0, 1536), "all(3072)": slice(0, 3072), "full only": slice(2560, 3072)}
for name, sl in blocks.items():
    for model, l2 in (("lr", 1.0), ("lr", 10.0), ("mlp", 1.0)):
        f1, sd, acc, pl = cv(Xa[:, sl], y, model, l2); print(f"{name:10s} {model} l2={l2:<4} macroF1 {f1:.3f}±{sd:.3f} acc {acc:.3f} predLEFT {pl:.2f}")
