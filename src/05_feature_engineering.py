# src/05_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


# ------------------------------------------------------------
# 1. "merged_btc.csv" ve "merged_eth.csv" dosyalarını mutlak yoldan oku
# ------------------------------------------------------------
def load_merged_data(symbol: str):
    """
    symbol: "btc" veya "eth"
    Bu fonksiyon, çalışma dizinini (os.getcwd()) proje kökü kabul ederek
    data/processed altındaki merged_{symbol}.csv dosyasını okur.
    """
    # 1) Proje kökü olarak çalışma dizinini alın
    project_root = os.getcwd()

    # 2) data/processed altındaki merged_{symbol}.csv dosyasına giden mutlak yolu oluşturun
    data_path = os.path.join(project_root, "data", "processed", f"merged_{symbol}.csv")

    # Konsolda hangi dosyanın okunacağını yazdırın
    print(f"[LOAD] {symbol.upper()} verisi okunacak dosya yolu: {data_path}")

    # 3) CSV’yi bu mutlak yoldan yükleyin
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------------------------------------
# 2. Teknik indikatör fonksiyonları
# ------------------------------------------------------------
def add_moving_averages(df: pd.DataFrame, windows=[7, 30]):
    """
    Her pencere boyutu (7, 30) için hareketli ortalama ekler.
    """
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_RSI(df: pd.DataFrame, period: int = 14):
    """
    14-günlük RSI hesaplar.
    """
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean()
    RS = roll_up / roll_down
    df[f"RSI{period}"] = 100 - (100 / (1 + RS))
    return df


def add_MACD(df: pd.DataFrame, span_short=12, span_long=26, span_signal=9):
    """
    EMA12 - EMA26 ve sinyal satırı (EMA9) farkı şeklinde MACD_diff ekler.
    """
    ema_short = df["Close"].ewm(span=span_short, adjust=False).mean()
    ema_long = df["Close"].ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    df["MACD_diff"] = macd - signal
    return df


def add_ATR(df: pd.DataFrame, period: int = 14):
    """
    Average True Range (ATR) hesaplar.
    True Range = max[(High-Low), abs(High-PrevClose), abs(Low-PrevClose)]
    ATR = rolling mean(True Range, period)
    """
    high_low = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift()).abs()
    low_prev = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df[f"ATR{period}"] = true_range.rolling(window=period).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2):
    """
    Bollinger Bands: orta bant = MA(window),
    üst bant = MA + num_std * std, alt bant = MA - num_std * std
    Ayrıca bant genişliğini (BB_WIDTH) ekleriz.
    """
    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    df["BB_UP"] = rolling_mean + (rolling_std * num_std)
    df["BB_LO"] = rolling_mean - (rolling_std * num_std)
    df["BB_WIDTH"] = df["BB_UP"] - df["BB_LO"]
    return df


# ------------------------------------------------------------
# 3. Lag özelliklerini ekleme
# ------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, lags=[1, 3, 7]):
    """
    Close ve avg_sentiment için belirlenen gecikmeleri (lag) ekler.
    Örneğin lag_1_Close = df['Close'].shift(1)
    """
    for l in lags:
        df[f"lag_{l}_Close"] = df["Close"].shift(l)
        df[f"lag_{l}_avg_sentiment"] = df["avg_sentiment"].shift(l)
    return df


# ------------------------------------------------------------
# 4. Hedef (target) oluşturma
# ------------------------------------------------------------
def add_target_columns(df: pd.DataFrame):
    """
    target_price: Ertesi günün kapanışı (Close.shift(-1))
    target_dir: 1 eğer ertesi günün kapanışı bir önceki güne göre yüksekse, 0 ise düşüş
    """
    df["target_price"] = df["Close"].shift(-1)
    df["target_dir"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


# ------------------------------------------------------------
# 5. Feature Engineering sürecini bir arada yürüten fonksiyon
# ------------------------------------------------------------
def run_feature_engineering(symbol: str):
    print(f"\n>>>>>>> {symbol.upper()} için Feature Engineering'e başlıyoruz <<<<<<<\n")

    # 5.1. Veriyi yükle (mutlak yol olarak cwd kullanılıyor)
    df = load_merged_data(symbol)
    print("Orijinal merged veri boyutu:", df.shape)
    print("\n--- İlk 3 satır (merged) ---")
    print(df.head(3).to_string(index=False))

    # 5.2. Teknik indikatörleri ekle
    df = add_moving_averages(df, windows=[7, 30])
    df = add_RSI(df, period=14)
    df = add_MACD(df, span_short=12, span_long=26, span_signal=9)
    df = add_ATR(df, period=14)
    df = add_bollinger_bands(df, window=20, num_std=2)
    print("\nTeknik indikatörlerin eklenmesinden sonra sütunlar şunlar:")
    print(df.columns.tolist())
    print("\n--- İlk 5 satır (indikatör sonrası) ---")
    print(df[[
        "Date", "Close", "MA7", "MA30", "RSI14",
        "MACD_diff", "ATR14", "BB_UP", "BB_LO", "BB_WIDTH"
    ]].head().to_string(index=False))

    # 5.3. Lag özelliklerini ekle
    df = add_lag_features(df, lags=[1, 3, 7])
    print("\n--- İlk 5 satır (lag özellikleri eklendikten sonra) ---")
    lag_cols = [col for col in df.columns if col.startswith("lag_")]
    print(df[["Date", "Close"] + lag_cols].head().to_string(index=False))

    # 5.4. Hedef değişkenlerini ekle
    df = add_target_columns(df)
    print("\n--- Son 3 satır (target eklendikten sonra) ---")
    print(df[["Date", "Close", "target_price", "target_dir"]].tail(3).to_string(index=False))

    # 5.5. NaN içeren satırları say ve göster (indikatör-lag kaynaklı)
    nan_counts = df.isna().sum()
    print("\nHer sütundaki NaN sayıları:")
    print(nan_counts[nan_counts > 0].to_string())

    # Dropna: İlk ~30-40 gün indikatörlerden dolayı NaN olacak, onları çıkar
    df_clean = df.dropna().reset_index(drop=True)
    print(f"\nNaN satırlar atıldıktan sonraki veri boyutu: {df_clean.shape}")

    # 5.6. İlgili feature ve target sütunlarını seçip yeni CSV’e kaydet
    feature_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "avg_sentiment", "news_count",
        "MA7", "MA30", "RSI14", "MACD_diff", "ATR14",
        "BB_UP", "BB_LO", "BB_WIDTH",
        "lag_1_Close", "lag_3_Close", "lag_7_Close",
        "lag_1_avg_sentiment", "lag_3_avg_sentiment"
    ]
    target_cols = ["target_price", "target_dir"]

    # CSV olarak kaydet (Date hariç, index=False), mutlak yol = cwd
    project_root = os.getcwd()
    out_csv = os.path.join(project_root, "data", "processed", f"{symbol}_features_ml.csv")
    df_clean[feature_cols + target_cols].to_csv(out_csv, index=False)
    print(f"\n{symbol.upper()} için feature+target içeren CSV oluşturuldu: {out_csv}")

    # 5.7. DL için MinMaxScaler ile ölçekleme
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = df_clean[feature_cols[1:]]  # Date hariç tüm sayısal feature'lar
    y_price = df_clean["target_price"].values.reshape(-1, 1)
    y_dir = df_clean["target_dir"].values.reshape(-1, 1)

    X_scaled = scaler.fit_transform(X)

    # NumPy dizilerini kaydet (mutlak yol = cwd)
    np.save(os.path.join(project_root, "data", "processed", f"{symbol}_X_scaled.npy"), X_scaled)
    np.save(os.path.join(project_root, "data", "processed", f"{symbol}_y_price.npy"), y_price)
    np.save(os.path.join(project_root, "data", "processed", f"{symbol}_y_dir.npy"), y_dir)
    print(f"\n{symbol.upper()} için numpy dizileri kaydedildi:")
    print(f"  - {os.path.join(project_root, 'data', 'processed', symbol + '_X_scaled.npy')}")
    print(f"  - {os.path.join(project_root, 'data', 'processed', symbol + '_y_price.npy')}")
    print(f"  - {os.path.join(project_root, 'data', 'processed', symbol + '_y_dir.npy')}")

    # Scaler'ı pickle olarak kaydet (mutlak yol = cwd)
    scaler_path = os.path.join(project_root, "models", f"{symbol}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"{symbol.upper()} için MinMaxScaler kaydedildi: {scaler_path}")

    print(f"\n>>>>>>> {symbol.upper()} Feature Engineering bitti <<<<<<<\n")


if __name__ == "__main__":
    # Aynı anda hem BTC hem ETH için çalıştır
    for coin in ["btc", "eth"]:
        run_feature_engineering(coin)
