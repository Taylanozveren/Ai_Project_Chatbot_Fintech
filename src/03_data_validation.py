# src/03_data_validation.py

import os
import pandas as pd
import numpy as np

def validate_merged_file(path):
    """
    1) Dosyanın var olup olmadığını, okunup okunamadığını kontrol eder.
    2) Eğer ilk satırda Date sütunu NaT ise bu satırı atar.
    3) Veri tiplerini, eksik değer sayılarını, değer aralıklarını kontrol eder.
    4) Tarih aralığını (min/max) ekrana yazdırır.
    5) Numeric sütunlarda geçersiz (NaN/inf) değer var mı bakar.
    6) “avg_sentiment” ve “news_count” mantıklı aralıkta mı ([-1,1] ve ≥0) incele.
    7) İlk 3 ve son 3 satırı ekrana yazdırır.
    """
    print(f"--- Validating file: {path} ---")

    # 1) Dosyanın varlığı
    if not os.path.isfile(path):
        print(f"  ⚠️ Hata: '{path}' bulunamadı.\n")
        return

    # 2) Dosyayı okuma
    try:
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False)
    except Exception as e:
        print(f"  ⚠️ Dosya okunurken hata oluştu: {e}\n")
        return

    print(f"  ✅ Başarıyla okundu. → İlk 5 satırı incele:")
    print(df.head(5).to_string(index=False))
    print()

    # 3) Eğer ilk satırda Date NaT ise at
    if df["Date"].isna().iloc[0]:
        print("  ℹ️ İlk satırda Date=NaT görünüyor. Bu satırı atıyoruz.")
        df = df.loc[df["Date"].notna()].reset_index(drop=True)
        print(f"  🔄 İlk satır atıldı, yeni shape: {df.shape}")
    else:
        print("  ℹ️ İlk satırda Date=NaT yok (başlık karışıklığı görünmüyor).")
    print()

    # 4) Veri tipleri ve eksik değer sayıları
    print("  → Veri Tipleri (dtypes) ve Eksik (NaN) Değer Sayıları:")
    dtypes = df.dtypes
    null_counts = df.isna().sum()
    for col in df.columns:
        dtype_str = str(dtypes[col])
        nan_count = null_counts[col]
        print(f"     • {col:15s} | dtype = {dtype_str:12s} | NaN count = {nan_count}")
    print()

    # 5) Tarih aralığı kontrolü
    date_min, date_max = df["Date"].min(), df["Date"].max()
    print(f"  → Tarih Aralığı: {date_min.date()}  →  {date_max.date()}  (toplam satır: {len(df)})")

    # Eksik gün kontrolü
    all_days = pd.date_range(start=date_min.date(), end=date_max.date(), freq="D")
    missing_days = set(all_days.date) - set(df["Date"].dt.date)
    print(f"  → Eksik fiyat+sentiment günü sayısı: {len(missing_days)}")
    if len(missing_days) > 0:
        print(f"     Örnek 5 eksik tarih: {sorted(list(missing_days))[:5]}")
    print()

    # 6) Numeric sütunlarda geçersiz (NaN/inf) değer kontrolü
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("  → Numeric sütunlarda geçersiz değer kontrolü:")
    for col in numeric_cols:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col]).sum()
        n_neg_inf = np.isneginf(df[col]).sum()
        print(f"     • {col:15s} | NaN={n_nan:4d} | +inf={n_inf:4d} | -inf={n_neg_inf:4d}")
    print()

    # 7) avg_sentiment aralığı kontrolü (VADER: -1 .. +1)
    if "avg_sentiment" in df.columns:
        min_s, max_s = df["avg_sentiment"].min(), df["avg_sentiment"].max()
        print(f"  → avg_sentiment aralığı: min={min_s:.4f}, max={max_s:.4f} (beklenti: [-1.0, +1.0])")
    else:
        print("  ℹ️ avg_sentiment sütunu bulunamadı.")

    # 8) news_count negatif mi kontrol et
    if "news_count" in df.columns:
        neg_news = (df["news_count"] < 0).sum()
        print(f"  → news_count negatif değer sayısı: {neg_news} (beklenti: 0)")
    else:
        print("  ℹ️ news_count sütunu bulunamadı.")
    print()

    # 9) İlk 3 ve son 3 satırı göster
    print("  → İlk 3 satır (temizlendikten sonra):")
    print(df.head(3).to_string(index=False))
    print()
    print("  → Son 3 satır (temizlendikten sonra):")
    print(df.tail(3).to_string(index=False))
    print("\n--- Dosya kontrolü tamamlandı ---\n\n")


if __name__ == "__main__":
    merged_btc_path = "data/processed/merged_btc.csv"
    merged_eth_path = "data/processed/merged_eth.csv"

    validate_merged_file(merged_btc_path)
    validate_merged_file(merged_eth_path)
