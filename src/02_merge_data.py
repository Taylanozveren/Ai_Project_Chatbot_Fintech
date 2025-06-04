# src/02_merge_data.py

import pandas as pd
import os

def merge_price_and_sentiment(price_csv, sentiment_csv, output_csv):
    """
    1) price_csv: data/raw/btc_prices.csv veya data/raw/eth_prices.csv
    2) sentiment_csv: data/processed/btc_daily_sentiment.csv veya data/processed/eth_daily_sentiment.csv
    3) output_csv: data/processed/merged_btc.csv veya data/processed/merged_eth.csv

    - Fiyat verisini ve günlük sentiment-etki verisini tarihe göre left join eder.
    - Eksik olan sentiment günlerini önceki günün sentiment’i ile doldurur (ffill);
      haber sayısı eksikse 0 atar.
    - Ortaya çıkan birleşik DataFrame’i CSV’ye kaydeder.
    """
    # ➊ CSV’leri oku
    price_df = pd.read_csv(price_csv, parse_dates=["Date"])
    sentiment_df = pd.read_csv(sentiment_csv, parse_dates=["date"])

    # ➋ Tarih sütunlarını normalleştir
    price_df["date"] = price_df["Date"].dt.date
    sentiment_df["date"] = sentiment_df["date"].dt.date

    # ➌ Birleştirme (left join)
    merged = pd.merge(price_df, sentiment_df, on="date", how="left")

    # ➍ Eksik sentiment hücrelerini doldur
    merged["avg_sentiment"] = merged["avg_sentiment"].fillna(method="ffill").fillna(0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    # ➎ Sonuç CSV olarak kaydet
    outdir = os.path.dirname(output_csv)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    merged.to_csv(output_csv, index=False)
    print(f"✅ Birleştirilmiş veri kaydedildi: {output_csv}")

if __name__ == "__main__":
    # BTC birleştirme
    merge_price_and_sentiment(
        "data/raw/btc_prices.csv",
        "data/processed/btc_daily_sentiment.csv",
        "data/processed/merged_btc.csv"
    )
    # ETH birleştirme
    merge_price_and_sentiment(
        "data/raw/eth_prices.csv",
        "data/processed/eth_daily_sentiment.csv",
        "data/processed/merged_eth.csv"
    )
