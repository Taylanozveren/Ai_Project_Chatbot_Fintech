# ─────────────────────────────────────────────
# Feature-Engineering  v6
#  • yeni hedefler:
#      – logret_h{1|3|5}
#      – bin_h{1|3|5}_thr{2|3}
# ─────────────────────────────────────────────
import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------- teknik indikatör fonksiyonları ----------
def add_MA(df, w=(7, 30)):
    for k in w: df[f"MA{k}"] = df["Close"].rolling(k).mean()
    return df
def add_RSI(df, p=14):
    diff = df["Close"].diff()
    up   = diff.clip(lower=0).rolling(p).mean()
    down = (-diff.clip(upper=0)).rolling(p).mean()
    rs   = up / down.replace(0, np.nan)
    df[f"RSI{p}"] = 100 - 100 / (1 + rs)
    return df
def add_MACD(df, s=12, l=26, sg=9):
    ema_s = df["Close"].ewm(span=s, adjust=False).mean()
    ema_l = df["Close"].ewm(span=l, adjust=False).mean()
    macd  = ema_s - ema_l
    sig   = macd.ewm(span=sg, adjust=False).mean()
    df["MACD_diff"] = macd - sig
    return df
def add_ATR(df, p=14):
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()],
                   axis=1).max(axis=1)
    df[f"ATR{p}"] = tr.rolling(p).mean(); return df
def add_BB(df, w=20, k=2):
    m = df["Close"].rolling(w).mean()
    s = df["Close"].rolling(w).std()
    df["BB_UP"] = m+k*s; df["BB_LO"] = m-k*s
    df["BB_WIDTH"] = df["BB_UP"]-df["BB_LO"]; return df
def add_ROC(df, p=10):
    df[f"ROC{p}"] = df["Close"].pct_change(p); return df
def add_VOL(df, p=14):
    df[f"VOL{p}"] = df["Close"].pct_change().rolling(p).std(); return df
# -----------------------------------------------------

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

# -------- yeni hedef fonksiyonu --------
HORIZONS   = (1, 3, 5)          # gün
THRESHOLDS = (0.02, 0.03)       # 2 %, 3 %

def add_targets_grid(df):
    for h in HORIZONS:
        logret = np.log(df["Close"].shift(-h) / df["Close"])
        df[f"logret_h{h}"] = logret          # regresyon hedefi

        for thr in THRESHOLDS:
            col = f"bin_h{h}_thr{int(thr*100)}"
            df[col] = (logret > thr).astype(int)  # 1 = güçlü yükseliş
    return df

# -------- ana FE rutini --------
def fe_one(sym:str):
    root = os.getcwd()
    df   = (pd.read_csv(f"data/processed/merged_{sym}.csv", parse_dates=["Date"])
              .sort_values("Date").reset_index(drop=True))

    # sentiment doldur + log(news)
    df["avg_sentiment"] = df["avg_sentiment"].ffill(limit=30).fillna(0)
    df["news_log1p"]    = np.log1p(df["news_count"])

    # teknik indikatör
    df = (add_MA(df).pipe(add_RSI).pipe(add_MACD)
                  .pipe(add_ATR).pipe(add_BB)
                  .pipe(add_ROC).pipe(add_VOL))

    # lag/rolling
    for l in (1,3,7):
        df[f"lag_{l}_Close"] = df["Close"].shift(l)
        df[f"lag_{l}_avg_sentiment"] = df["avg_sentiment"].shift(l)
    for w in (3,7):
        df[f"sent_mean_{w}d"] = df["avg_sentiment"].rolling(w).mean()
    logret = np.log(df["Close"]).diff()
    df["lag_ret_1d"] = logret.shift(1); df["lag_ret_2d"] = logret.shift(2)
    df["dayofweek"]  = df["Date"].dt.dayofweek

    # hedefler
    df = add_targets_grid(df)

    clean = df.dropna().reset_index(drop=True)
    out   = f"data/processed/{sym}_features_ml_v6.csv"
    target_cols = [c for c in clean.columns if c.startswith(("logret_","bin_"))]
    clean[["Date"] + BASE + target_cols].to_csv(out, index=False)
    print(f"[✓] {sym.upper()} FE-v6  →  {out}  ({clean.shape[0]} satır)")

    # scaler
    scaler = StandardScaler().fit(clean[BASE])
    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open(f"models/{sym}_scaler_v6.pkl", "wb"))

# -----------------------------------------------------
if __name__ == "__main__":
    for coin in ("btc","eth"):
        fe_one(coin)
