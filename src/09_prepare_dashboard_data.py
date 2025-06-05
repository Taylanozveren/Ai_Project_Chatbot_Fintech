import numpy as np, pandas as pd, lightgbm as lgb, pickle, joblib
from pathlib import Path
BASE = [
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
TARGET = "bin_h3_thr2"       # şimdilik tek hedef
TEST_START = "2024-01-01"
OUT = Path("dashboard_data"); OUT.mkdir(exist_ok=True, parents=True)

def fit_and_predict(sym):
    df = pd.read_csv(f"data/processed/{sym}_features_ml_v6.csv",
                     parse_dates=["Date"])
    tr, te = df[df.Date < TEST_START], df[df.Date >= TEST_START]

    bst = lgb.train({"objective":"binary","metric":"auc",
                     "learning_rate":.05,"num_leaves":31,"seed":42},
                    lgb.Dataset(tr[BASE], tr[TARGET]),
                    num_boost_round=400)

    df["prob"] = bst.predict(df[BASE])            # tüm tarih için olasılık
    df.to_parquet(OUT / f"{sym}_full.parquet")    # dashboard hızlı okusun
    joblib.dump(bst,      OUT / f"{sym}_model.pkl")
    joblib.dump(tr[BASE].columns.tolist(), OUT / "feature_list.pkl")

for coin in ("btc","eth"):
    fit_and_predict(coin)
print("[✓] Dashboard verileri hazır → dashboard_data/")
