# src/01_data_processing.py

import yfinance as yf
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

def fetch_price_data():
    """(1) Bitcoin ve Ethereum fiyat verilerini yfinance'tan indir ve CSV olarak kaydet."""
    print("▶️ 1) Fiyat verisi indiriliyor…")

    # 1.a) BTC-USD fiyat verisini çek
    btc = yf.download("BTC-USD", start="2018-01-01", end="2025-12-31")
    print(f"    BTC fiyat satır sayısı: {btc.shape[0]}, sütun sayısı: {btc.shape[1]}")
    btc.reset_index(inplace=True)
    if not os.path.isdir("data/raw"):
        os.makedirs("data/raw")
    btc.to_csv("data/raw/btc_prices.csv", index=False)
    print("    ✅ data/raw/btc_prices.csv kaydedildi.")

    # 1.b) ETH-USD fiyat verisini çek
    eth = yf.download("ETH-USD", start="2018-01-01", end="2025-12-31")
    print(f"    ETH fiyat satır sayısı: {eth.shape[0]}, sütun sayısı: {eth.shape[1]}")
    eth.reset_index(inplace=True)
    eth.to_csv("data/raw/eth_prices.csv", index=False)
    print("    ✅ data/raw/eth_prices.csv kaydedildi.\n")

def filter_coin_news():
    """
    (2) Ham haber setinden BTC ve ETH ile ilgili haberleri ayıkla,
    'text' sütununu oluştur ve CSV olarak kaydet.
    """
    print("▶️ 2) Haber verileri yükleniyor ve filtreleniyor…")
    news_path = "data/raw/crypto_news.csv"
    if not os.path.isfile(news_path):
        print(f"    ⚠️ Hata: {news_path} bulunamadı. Lütfen doğru klasöre kopyaladığınızdan emin olun.")
        return

    df = pd.read_csv(news_path)
    print(f"    🔍 Toplam ham haber satırı: {df.shape[0]}")
    # .head() gösterelim:
    print("    Örnek satırlar (ham CSV’den):")
    print(df.head(3).to_string(index=False))
    print()

    # 2.a) 'description' sütunu NULL ise boş string'e çevir
    df['description'] = df['description'].fillna("")

    # 2.b) Haber metni olarak başlık + açıklama birleştirmesi yap
    df['text'] = df['title'].astype(str) + " " + df['description'].astype(str)

    # 2.c) 'currencies' sütununda BTC/ETH içerip içermediğini kontrol et
    df_btc = df[df['currencies'].str.contains("BTC", na=False)].copy()
    df_eth = df[df['currencies'].str.contains("ETH", na=False)].copy()
    print(f"    BTC etiketli haber sayısı: {df_btc.shape[0]}")
    print(f"    ETH etiketli haber sayısı: {df_eth.shape[0]}")
    print()

    # 2.d) Filtrelenmiş DataFrame’in ilk birkaç satırını göster
    print("    Örnek BTC haberleri:")
    print(df_btc[['newsDatetime','text','currencies']].head(3).to_string(index=False))
    print("\n    Örnek ETH haberleri:")
    print(df_eth[['newsDatetime','text','currencies']].head(3).to_string(index=False))
    print()

    # 2.e) Sadece gerekli sütunları alarak kaydet
    df_btc = df_btc[['newsDatetime','text','currencies']]
    df_eth = df_eth[['newsDatetime','text','currencies']]

    # Klasör kontrolü
    if not os.path.isdir("data/raw"):
        os.makedirs("data/raw")

    df_btc.to_csv("data/raw/btc_news.csv", index=False)
    print("    ✅ data/raw/btc_news.csv kaydedildi.")
    df_eth.to_csv("data/raw/eth_news.csv", index=False)
    print("    ✅ data/raw/eth_news.csv kaydedildi.\n")

def compute_daily_sentiment(input_csv, output_csv):
    """
    (3) BTC/ETH haber CSV’lerinden günlük ortalama sentiment ve haber sayısını çıkar,
    ardından CSV olarak kaydet.
    """
    print(f"▶️ 3) {input_csv} dosyasından günlük sentiment hesaplanıyor…")
    if not os.path.isfile(input_csv):
        print(f"    ⚠️ Hata: {input_csv} bulunamadı.")
        return

    df = pd.read_csv(input_csv)
    print(f"    [OK] {input_csv} satır sayısı: {df.shape[0]}")
    # Tarihi datetime tipine dönüştür
    df['newsDatetime'] = pd.to_datetime(df['newsDatetime'])
    # Metin sütunu zaten oluşturulmuş, yalnızca boşsa doldur
    df['text'] = df['text'].fillna("")
    # VADER ile sentiment hesapla
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['text'].apply(lambda txt: sia.polarity_scores(str(txt))['compound'])
    print("    👍 Sentiment skoru hesaplandı. Örnek skorlar:")
    print(df[['newsDatetime','sentiment_score']].head(3).to_string(index=False))
    # Günlük tarihe düşür
    df['date'] = df['newsDatetime'].dt.date
    # Günlük ortalama sentiment ve haber sayısı
    daily = df.groupby('date').agg({
        'sentiment_score':'mean',
        'text':'count'
    }).rename(columns={'sentiment_score':'avg_sentiment','text':'news_count'}).reset_index()
    print("    🗓️ Günlük özet (ilk 3 gün):")
    print(daily.head(3).to_string(index=False))
    print()

    # Klasör kontrolü
    outdir = os.path.dirname(output_csv)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    daily.to_csv(output_csv, index=False)
    print(f"    ✅ {output_csv} kaydedildi.\n")

if __name__ == "__main__":
    # 1) Fiyat verilerini indir
    fetch_price_data()
    # 2) Haberleri BTC/ETH olarak filtrele
    filter_coin_news()
    # 3) Günlük sentiment’i hesapla (BTC ve ETH için)
    compute_daily_sentiment("data/raw/btc_news.csv", "data/processed/btc_daily_sentiment.csv")
    compute_daily_sentiment("data/raw/eth_news.csv", "data/processed/eth_daily_sentiment.csv")
