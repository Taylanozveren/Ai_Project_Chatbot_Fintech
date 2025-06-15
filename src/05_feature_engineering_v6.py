# ─────────────────────────────────────────────
# Feature-Engineering  v6 - SMART NaN HANDLING
#  • Veri kalitesi korunur
#  • Son günler için akıllı imputation
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
    up = diff.clip(lower=0).rolling(p).mean()
    down = (-diff.clip(upper=0)).rolling(p).mean()
    rs = up / down.replace(0, np.nan)
    df[f"RSI{p}"] = 100 - 100 / (1 + rs)
    return df


def add_MACD(df, s=12, l=26, sg=9):
    ema_s = df["Close"].ewm(span=s, adjust=False).mean()
    ema_l = df["Close"].ewm(span=l, adjust=False).mean()
    macd = ema_s - ema_l
    sig = macd.ewm(span=sg, adjust=False).mean()
    df["MACD_diff"] = macd - sig
    return df


def add_ATR(df, p=14):
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()],
                   axis=1).max(axis=1)
    df[f"ATR{p}"] = tr.rolling(p).mean();
    return df


def add_BB(df, w=20, k=2):
    m = df["Close"].rolling(w).mean()
    s = df["Close"].rolling(w).std()
    df["BB_UP"] = m + k * s;
    df["BB_LO"] = m - k * s
    df["BB_WIDTH"] = df["BB_UP"] - df["BB_LO"];
    return df


def add_ROC(df, p=10):
    df[f"ROC{p}"] = df["Close"].pct_change(p);
    return df


def add_VOL(df, p=14):
    df[f"VOL{p}"] = df["Close"].pct_change().rolling(p).std();
    return df


# ---------- AKILLI İMPUTATION FONKSİYONU ----------
def smart_impute_recent(df, max_recent_days=3):
    """
    Son X günlük veri için akıllı doldurma:
    - Rolling indicators: son valid değerle doldur
    - Lag features: hesaplanabilirse hesapla
    - Sentiment: forward-fill
    """
    df_imp = df.copy()
    cutoff_date = df['Date'].max() - pd.Timedelta(days=max_recent_days)
    recent_mask = df['Date'] > cutoff_date

    # Son günler için imputation
    for idx in df_imp[recent_mask].index:
        # 1. Rolling/Technical indicators - son valid değerle doldur
        for col in ['MA7', 'MA30', 'RSI14', 'MACD_diff', 'ATR14', 'BB_UP', 'BB_LO', 'BB_WIDTH', 'ROC10', 'VOL14']:
            if pd.isna(df_imp.loc[idx, col]):
                # Son valid değeri al
                last_valid = df_imp.loc[:idx - 1, col].dropna()
                if len(last_valid) > 0:
                    df_imp.loc[idx, col] = last_valid.iloc[-1]

        # 2. Lag features - geriye bakarak hesapla
        for lag in [1, 3, 7]:
            lag_idx = idx - lag
            if lag_idx >= 0:
                # Close lag
                if pd.isna(df_imp.loc[idx, f'lag_{lag}_Close']):
                    df_imp.loc[idx, f'lag_{lag}_Close'] = df_imp.loc[lag_idx, 'Close']

                # Sentiment lag
                if pd.isna(df_imp.loc[idx, f'lag_{lag}_avg_sentiment']):
                    df_imp.loc[idx, f'lag_{lag}_avg_sentiment'] = df_imp.loc[lag_idx, 'avg_sentiment']

        # 3. Rolling sentiment means
        for window in [3, 7]:
            col = f'sent_mean_{window}d'
            if pd.isna(df_imp.loc[idx, col]):
                start_idx = max(0, idx - window + 1)
                window_data = df_imp.loc[start_idx:idx, 'avg_sentiment'].dropna()
                if len(window_data) > 0:
                    df_imp.loc[idx, col] = window_data.mean()

        # 4. Return lags
        if pd.isna(df_imp.loc[idx, 'lag_ret_1d']) and idx >= 1:
            if not pd.isna(df_imp.loc[idx - 1, 'Close']) and not pd.isna(df_imp.loc[max(0, idx - 2), 'Close']):
                ret = np.log(df_imp.loc[idx - 1, 'Close'] / df_imp.loc[max(0, idx - 2), 'Close'])
                df_imp.loc[idx, 'lag_ret_1d'] = ret

        if pd.isna(df_imp.loc[idx, 'lag_ret_2d']) and idx >= 2:
            if not pd.isna(df_imp.loc[idx - 2, 'Close']) and not pd.isna(df_imp.loc[max(0, idx - 3), 'Close']):
                ret = np.log(df_imp.loc[idx - 2, 'Close'] / df_imp.loc[max(0, idx - 3), 'Close'])
                df_imp.loc[idx, 'lag_ret_2d'] = ret

    return df_imp


# -----------------------------------------------------

BASE = [
    "Open", "High", "Low", "Close", "Volume",
    "avg_sentiment", "news_count", "news_log1p",
    "MA7", "MA30", "RSI14", "MACD_diff", "ATR14",
    "BB_UP", "BB_LO", "BB_WIDTH", "ROC10", "VOL14",
    "lag_1_Close", "lag_3_Close", "lag_7_Close",
    "lag_1_avg_sentiment", "lag_3_avg_sentiment",
    "sent_mean_3d", "sent_mean_7d",
    "lag_ret_1d", "lag_ret_2d",
    "dayofweek"
]

HORIZONS = (1, 3, 5)
THRESHOLDS = (0.02, 0.03)


def add_targets_grid(df):
    for h in HORIZONS:
        logret = np.log(df["Close"].shift(-h) / df["Close"])
        df[f"logret_h{h}"] = logret

        for thr in THRESHOLDS:
            col = f"bin_h{h}_thr{int(thr * 100)}"
            df[col] = (logret > thr).astype(int)
    return df


def fe_one(sym: str):
    root = os.getcwd()
    df = (pd.read_csv(f"data/processed/merged_{sym}.csv", parse_dates=["Date"])
          .sort_values("Date").reset_index(drop=True))

    print(f"[INFO] {sym.upper()} raw data: {len(df)} rows, {df['Date'].min()} to {df['Date'].max()}")

    # sentiment doldur + log(news)
    df["avg_sentiment"] = df["avg_sentiment"].ffill(limit=30).fillna(0)
    df["news_log1p"] = np.log1p(df["news_count"])

    # teknik indikatör
    df = (add_MA(df).pipe(add_RSI).pipe(add_MACD)
          .pipe(add_ATR).pipe(add_BB)
          .pipe(add_ROC).pipe(add_VOL))

    # lag/rolling
    for l in (1, 3, 7):
        df[f"lag_{l}_Close"] = df["Close"].shift(l)
        df[f"lag_{l}_avg_sentiment"] = df["avg_sentiment"].shift(l)
    for w in (3, 7):
        df[f"sent_mean_{w}d"] = df["avg_sentiment"].rolling(w).mean()
    logret = np.log(df["Close"]).diff()
    df["lag_ret_1d"] = logret.shift(1);
    df["lag_ret_2d"] = logret.shift(2)
    df["dayofweek"] = df["Date"].dt.dayofweek

    # hedefler (son 5 gün için NaN olacak - normal)
    df = add_targets_grid(df)

    print(f"[INFO] After FE: {len(df)} rows, latest: {df['Date'].max()}")

    # 🎯 AKILLI ÇÖZÜM: Son günler için smart imputation
    df_imputed = smart_impute_recent(df, max_recent_days=3)

    # NaN durumunu analiz et
    target_cols = [c for c in df.columns if c.startswith(("logret_", "bin_"))]
    feature_cols = ["Date"] + BASE + target_cols
    available_features = [col for col in feature_cols if col in df_imputed.columns]

    # İki aşamalı temizlik:
    # 1. Bulk data (30+ gün önce) - tam temiz olmalı
    bulk_cutoff = df_imputed['Date'].max() - pd.Timedelta(days=30)
    bulk_mask = df_imputed['Date'] <= bulk_cutoff
    bulk_data = df_imputed[bulk_mask].dropna(subset=BASE)

    # 2. Recent data (son 30 gün) - akılcı NaN toleransı
    recent_mask = df_imputed['Date'] > bulk_cutoff
    recent_data = df_imputed[recent_mask].copy()

    # Recent data için kritik feature'ları kontrol et
    critical_features = ['Close', 'Volume', 'avg_sentiment']  # Bunlar mutlaka olmalı
    recent_clean = recent_data.dropna(subset=critical_features)

    # Diğer feature'lar için %80 doluluk kuralı
    other_features = [f for f in BASE if f not in critical_features + ['Date']]
    for idx in recent_clean.index:
        row_completeness = recent_clean.loc[idx, other_features].notna().mean()
        if row_completeness < 0.8:  # %80'den az doluysa çıkar
            recent_clean = recent_clean.drop(idx)

    # Birleştir
    clean = pd.concat([bulk_data, recent_clean], ignore_index=True)
    clean = clean.sort_values('Date').reset_index(drop=True)

    print(f"[INFO] Bulk data (>{bulk_cutoff.date()}): {len(bulk_data)} rows")
    print(f"[INFO] Recent data: {len(recent_clean)} rows")
    print(f"[INFO] Final clean: {len(clean)} rows, latest: {clean['Date'].max()}")

    # Son kontrol: veri kalitesi raporu
    completeness = clean[BASE].notna().mean()
    low_quality = completeness[completeness < 0.95]
    if len(low_quality) > 0:
        print(f"[WARNING] Low completeness features: {dict(low_quality)}")

    out = f"data/processed/{sym}_features_ml_v6.csv"
    clean[available_features].to_csv(out, index=False)
    print(f"[✓] {sym.upper()} FE-v6 → {out} ({clean.shape[0]} rows)")

    # High-quality scaler
    scaler_data = clean[BASE].dropna()  # Scaler için tam temiz veri
    scaler = StandardScaler().fit(scaler_data)
    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open(f"models/{sym}_scaler_v6.pkl", "wb"))

    return clean


# -----------------------------------------------------
if __name__ == "__main__":
    for coin in ("btc", "eth"):
        result_df = fe_one(coin)
        print(f"\n{coin.upper()} QUALITY CHECK:")
        print(f"  📊 Rows: {len(result_df)}")
        print(f"  📅 Range: {result_df['Date'].min().date()} → {result_df['Date'].max().date()}")
        print(f"  💰 Latest: ${result_df['Close'].iloc[-1]:,.2f}")

        # Veri kalitesi metriği
        completeness = result_df[BASE].notna().mean().mean()
        print(f"  ✨ Quality: {completeness:.1%} complete")
        print("-" * 60)