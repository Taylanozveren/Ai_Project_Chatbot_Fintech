# src/04_clean_merged.py

import pandas as pd
import os


def clean_and_save(merged_path):
    """
    - 'merged_btc.csv' veya 'merged_eth.csv' dosyasındaki
      fiyat/volume sütunlarını float/int'e çevirir.
    - Tarih sütununu datetime olarak bırakır.
    - Dönüştürülmüş DataFrame'i üzerine yazar.
    """
    print(f"🔍 Reading: {merged_path}")
    df = pd.read_csv(merged_path, parse_dates=["Date"])

    # Eğer ilk satırda Date==NaT varsa at
    if df["Date"].isna().iloc[0]:
        df = df[df["Date"].notna()].reset_index(drop=True)
        print("  ⚠️ İlk satırda Date=NaT bulundu; atıldı.")

    # Sayısal olması gereken sütunlar
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            # 'object' tipindeki sütunu önce string'ten ayıklayıp float'a çevir
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"  • {col} → numeric (örnek: {df[col].dtype})")

    # Ek olarak 'date' (küçük harfli) sütununu da datetime hale getir (zorunlu değil; burada kontrol amaçlı)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Aynı dosyaya tekrar yaz
    df.to_csv(merged_path, index=False)
    print(f"✅ Cleaned & saved: {merged_path}\n")


def main():
    files = ["data/processed/merged_btc.csv", "data/processed/merged_eth.csv"]
    for path in files:
        if os.path.isfile(path):
            clean_and_save(path)
        else:
            print(f"⚠️ Dosya bulunamadı: {path}")


if __name__ == "__main__":
    main()
