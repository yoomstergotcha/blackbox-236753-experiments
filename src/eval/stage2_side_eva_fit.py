"""전체 라벨로 side/evasion 로지스틱 회귀를 학습해 추론 스니펫에 심을 상수(mu, sd, w, b)를 JSON으로 출력.
입력: output/s2_side_eva_features.csv (stage2_side_eva_head.py가 저장). 실행: python -m src.eval.stage2_side_eva_fit --side-cols side_score lat has --eva-cols eva_img eva_vid occ
"""
import argparse, json, numpy as np, pandas as pd


def fit(X, y, l2=1.0, iters=2000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xt = np.c_[(X - mu) / sd, np.ones(len(y))]; w = np.zeros(Xt.shape[1])
    for _ in range(iters):
        z = Xt @ w; g = Xt.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + l2 * np.r_[w[:-1], 0] / len(y); w -= lr * g
    p = 1 / (1 + np.exp(-(Xt @ w))); return dict(mu=mu.tolist(), sd=sd.tolist(), w=w[:-1].tolist(), b=float(w[-1]), train_acc=float(((p > 0.5) == y).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--side-cols", nargs="+", default=["side_score"]); ap.add_argument("--eva-cols", nargs="+", default=["eva_img", "eva_vid"]); ap.add_argument("--out", default="output/s2_side_eva_lr.json"); a = ap.parse_args()
    F = pd.read_csv("output/s2_side_eva_features.csv", dtype={"vid": str})
    s = F[F.manual_side.notna()]; e = F[F.manual_evasion.notna()]
    out = {"side": {"cols": a.side_cols, **fit(s[a.side_cols].to_numpy(float), (s.manual_side == "LEFT").astype(int).to_numpy())},
           "evasion": {"cols": a.eva_cols, **fit(e[a.eva_cols].to_numpy(float), e.manual_evasion.astype(int).to_numpy())}}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
