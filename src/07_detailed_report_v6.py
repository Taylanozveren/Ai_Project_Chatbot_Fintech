# ───────── src/07_detailed_report_v6.py ─────────
"""
Kullanım
--------
!python src/07_detailed_report_v6.py bin_h3_thr2
↳ results/final_report_bin_h3_thr2.csv oluşturur
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, classification_report)

# ─── 1) Hedef kolonu al ───
cli_args = [a for a in sys.argv[1:] if not a.startswith("-")]
TARGET   = cli_args[0] if cli_args else "bin_h3_thr2"
print(f"[i] Hedef sütun   : {TARGET}")

# ─── 2) Ortak sabitler ───
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
TEST_START = np.datetime64("2024-01-01")

# ─── 3) Tek coin rapor fonksiyonu ───
def analyse(sym: str) -> dict:
    df = pd.read_csv(f"data/processed/{sym}_features_ml_v6.csv",
                     parse_dates=["Date"])
    if TARGET not in df.columns:
        raise ValueError(f"❌ '{TARGET}' sütunu {sym.upper()} datasında yok!")

    tr, te = df[df.Date < TEST_START], df[df.Date >= TEST_START]

    bst = lgb.train(
        params=dict(objective="binary", metric="auc",
                    learning_rate=.05, num_leaves=31, seed=42),
        train_set=lgb.Dataset(tr[BASE], label=tr[TARGET]),
        num_boost_round=400
    )

    prob = bst.predict(te[BASE])
    pred = (prob > .5).astype(int)

    auc  = roc_auc_score(te[TARGET], prob)
    acc  = accuracy_score(te[TARGET], pred)
    cm   = confusion_matrix(te[TARGET], pred)
    cls  = classification_report(te[TARGET], pred,
                                 digits=3, output_dict=True)

    return dict(coin=sym, target=TARGET,
                auc=round(auc,3), acc=round(acc,3),
                prec=round(cls['1']['precision'],3),
                rec =round(cls['1']['recall'],3),
                f1  =round(cls['1']['f1-score'],3),
                tn=int(cm[0,0]), fp=int(cm[0,1]),
                fn=int(cm[1,0]), tp=int(cm[1,1]),
                n_test=len(te))

# ─── 4) Main ───
if __name__ == "__main__":
    rows = [analyse(c) for c in ("btc", "eth")]
    Path("results").mkdir(exist_ok=True)
    fn = f"results/final_report_{TARGET}.csv"
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(pd.DataFrame(rows).round(3))
    print(f"\n[✓] Kaydedildi → {fn}")
