"""
06_walkforward_v6.py
────────────────────
• 12 hedef (bin_h{1,3,5}_thr{2,3}) için
  3-adımlı walk-forward (2022/23/24 başlangıç)
• Çıktı : results/walk_metrics_v6.csv
  Konsolda pivot(AUC) + pivot(ACC)
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path

# ------------------- ortak sabitler -------------------
BASE = [      # 28 numeric + takvim
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
SPLITS  = [np.datetime64("2022-01-01"),
           np.datetime64("2023-01-01"),
           np.datetime64("2024-01-01")]
TARGETS = [f"bin_h{h}_thr{t}" for h in (1,3,5) for t in (2,3)]  # 6×2 = 12

# ------------------------------------------------------
def one_target(df: pd.DataFrame, col: str):
    """tek hedef, 3 pencere → [{step,auc,acc}, …]"""
    rows = []
    for i, cut in enumerate(SPLITS, 1):
        tr, te = df[df.Date < cut], df[df.Date >= cut]
        bst = lgb.train(
            dict(objective="binary", metric="auc",
                 learning_rate=.05, num_leaves=31, seed=42),
            lgb.Dataset(tr[BASE], label=tr[col]),
            num_boost_round=400
        )
        prob = bst.predict(te[BASE])
        rows.append(dict(step=i,
                         auc = roc_auc_score(te[col], prob),
                         acc = accuracy_score(te[col], prob > .5)))
    return rows

def run(sym: str):
    df = pd.read_csv(f"data/processed/{sym}_features_ml_v6.csv",
                     parse_dates=["Date"])
    rep = []
    for tgt in TARGETS:
        for r in one_target(df, tgt):
            rep.append(dict(coin=sym, target=tgt, **r))
    return rep

# ---------------- MAIN ----------------
if __name__ == "__main__":
    all_rows = []
    for coin in ("btc", "eth"):
        all_rows += run(coin)

    out = pd.DataFrame(all_rows)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/walk_metrics_v6.csv", index=False)

    print("\n=== AUC pivot ===")
    print(out.pivot_table("auc", ["coin", "target"], "step").round(3))
    print("\n=== ACC pivot ===")
    print(out.pivot_table("acc", ["coin", "target"], "step").round(3))
