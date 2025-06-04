# src/raw_eda_checks.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def check_price_dates(price_csv):
    """
    price_csv: data/raw/btc_prices.csv veya data/raw/eth_prices.csv
    Fiyat datasındaki tarih aralığını ve eksik günleri kontrol eder.
    """
    df = pd.read_csv(price_csv, parse_dates=['Date'])

    # Eğer Date sütununda NaT varsa, o satırları atalım
    df = df[df['Date'].notna()]

    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    start, end = df['Date'].min(), df['Date'].max()
    print(f"+ {os.path.basename(price_csv)} aralığı: {start.date()} → {end.date()}  (toplam satır: {len(df)})")

    all_days = pd.date_range(start=start.date(), end=end.date(), freq='D')
    price_days = df['Date'].dt.normalize()
    missing = all_days.difference(price_days)
    print(f"  → Eksik fiyat günü sayısı: {len(missing)}")
    if len(missing) > 0:
        print(f"    Örnek eksik tarihler: {missing[:5].date.tolist()}")
    print()

def check_sentiment_dates(sentiment_csv, price_csv):
    """
    sentiment_csv: data/processed/btc_daily_sentiment.csv veya data/processed/eth_daily_sentiment.csv
    price_csv: data/raw/btc_prices.csv veya data/raw/eth_prices.csv
    Eksik sentiment günlerini tespit eder.
    """
    df_price = pd.read_csv(price_csv, parse_dates=['Date'])
    df_sent = pd.read_csv(sentiment_csv, parse_dates=['date'])

    # Her iki çerçevede de NaT içeren satırları atalım
    df_price = df_price[df_price['Date'].notna()]
    df_sent  = df_sent[df_sent['date'].notna()]

    df_price.sort_values('Date', inplace=True)
    df_sent.sort_values('date', inplace=True)

    price_days = set(df_price['Date'].dt.date)
    sent_days  = set(df_sent['date'].dt.date)

    missing_sent = sorted(price_days - sent_days)
    print(f"+ {os.path.basename(sentiment_csv)} satır sayısı (Geçersiz satırlar atıldıktan sonra): {len(df_sent)}")
    print(f"  → Fiyat var ama sentiment yok gün sayısı: {len(missing_sent)}")
    if len(missing_sent) > 0:
        print(f"    Örnek eksik tarihler: {missing_sent[:5]}")
    print()

def plot_sentiment_distribution(sentiment_csv):
    """
    sentiment_csv: data/processed/btc_daily_sentiment.csv veya eth_daily_sentiment.csv
    avg_sentiment histogramını çizer.
    """
    df = pd.read_csv(sentiment_csv, parse_dates=['date'])
    df = df[df['date'].notna()]  # Geçersiz (NaT) satırları at
    plt.figure(figsize=(6,4))
    plt.hist(df['avg_sentiment'], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"Histogram of avg_sentiment ({os.path.basename(sentiment_csv)})")
    plt.xlabel("avg_sentiment")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()

def check_news_data(news_csv):
    """
    news_csv: data/raw/btc_news.csv veya data/raw/eth_news.csv
    Haber datası hakkında özet istatistikler.
    """
    df = pd.read_csv(news_csv, parse_dates=['newsDatetime'])
    print(f"+ {os.path.basename(news_csv)} satır sayısı: {len(df)}")

    null_text_count = df['text'].isna().sum() + (df['text'].astype(str).str.strip() == "").sum()
    print(f"  → ‘text’ sütununda boş (NaN veya boş) satır sayısı: {null_text_count}")

    df = df[df['newsDatetime'].notna()]
    start, end = df['newsDatetime'].min(), df['newsDatetime'].max()
    print(f"  → Haber datası tarih aralığı: {start.date()} → {end.date()}")

    df['date_only'] = df['newsDatetime'].dt.date
    daily_counts = df.groupby('date_only').size()
    mean_count = daily_counts.mean()
    max_count = daily_counts.max()
    max_day   = daily_counts.idxmax()
    print(f"  → Ortalama haber sayısı/gün: {mean_count:.2f}")
    print(f"  → En fazla haber kaydı olan gün: {max_day} ({max_count} haber)")
    print(f"  → İlk haber tarihi: {start}, Son haber tarihi: {end}")
    print()

def main():
    print("===== BTC Fiyat & Sentiment & News Kontrolleri =====\n")
    check_price_dates("data/raw/btc_prices.csv")
    check_sentiment_dates("data/processed/btc_daily_sentiment.csv", "data/raw/btc_prices.csv")
    plot_sentiment_distribution("data/processed/btc_daily_sentiment.csv")
    check_news_data("data/raw/btc_news.csv")

    print("\n===== ETH Fiyat & Sentiment & News Kontrolleri =====\n")
    check_price_dates("data/raw/eth_prices.csv")
    check_sentiment_dates("data/processed/eth_daily_sentiment.csv", "data/raw/eth_prices.csv")
    plot_sentiment_distribution("data/processed/eth_daily_sentiment.csv")
    check_news_data("data/raw/eth_news.csv")

if __name__ == "__main__":
    main()
