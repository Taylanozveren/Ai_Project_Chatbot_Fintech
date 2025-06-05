# ─── src/08_threshold_tune_v6.py (güncel) ───
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import precision_recall_curve, roc_auc_score, fbeta_score
from pathlib import Path

BASE = [                     # 28 numeric + takvim
    "Open","High","Low","Close","Volume",
    "avg_sentiment","news_count","news_log1p",
    "MA7","MA30","RSI14","MACD_diff","ATR14",
    "BB_UP","BB_LO","BB_WIDTH","ROC10","VOL14",
    "lag_1_Close","lag_3_Close","lag_7_Close",
    "lag_1_avg_sentiment","lag_3_avg_sentiment",
    "sent_mean_3d","sent_mean_7d",
    "lag_ret_1d","lag_ret_2d",
    "dayofweek"
]
TARGET = "bin_h3_thr2"
TEST_START = np.datetime64("2024-01-01")

def tune(sym: str):
    df = pd.read_csv(f"data/processed/{sym}_features_ml_v6.csv",
                     parse_dates=["Date"])
    tr, te = df[df.Date < TEST_START], df[df.Date >= TEST_START]
    bst = lgb.train(dict(objective="binary", metric="auc",
                         learning_rate=.05, num_leaves=31, seed=42),
                    lgb.Dataset(tr[BASE], tr[TARGET]), 400)

    prob = bst.predict(te[BASE]); y = te[TARGET].values
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2*p*r/(p + r + 1e-9)
    k = f1.argmax()

    return dict(coin=sym,
                auc = round(roc_auc_score(y, prob), 3),
                best_thr = round(float(t[k]), 2),
                f1  = round(float(f1[k]), 3),
                prec = round(float(p[k]), 3),
                rec  = round(float(r[k]), 3),
                n_pos = int(y.sum()),
                n_test = len(y))

if __name__ == "__main__":
    rows = [tune(c) for c in ("btc", "eth")]
    Path("results").mkdir(exist_ok=True)
    fn = "results/threshold_tune_bin_h3_thr2.csv"
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(pd.DataFrame(rows))
    print(f"\n[✓] Kaydedildi → {fn}")
