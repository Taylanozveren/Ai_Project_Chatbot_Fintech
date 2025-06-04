# src/06_validation_checks.py

import pandas as pd
import numpy as np
import os
import pickle

def main():
    # 1. Proje kök dizinini belirle (çalışma dizinini proje kökü kabul ediyoruz)
    project_root = os.getcwd()
    print("Proje kök dizini:", project_root)

    # ----------------------------------------------------------------------
    # 2. Oluşan CSV'lerin İlk Satırlarını Gözlemlemek
    # ----------------------------------------------------------------------
    btc_csv_path = os.path.join(project_root, "data", "processed", "btc_features_ml.csv")
    eth_csv_path = os.path.join(project_root, "data", "processed", "eth_features_ml.csv")

    if os.path.exists(btc_csv_path):
        btc_df = pd.read_csv(btc_csv_path, parse_dates=["Date"])
        print("\n=== BTC features CSV – İlk 5 satır ===")
        print(btc_df.head().to_string(index=False))
        print("\nBTC CSV sütun sayısı:", len(btc_df.columns))
    else:
        print("\nBTC features CSV bulunamadı:", btc_csv_path)

    if os.path.exists(eth_csv_path):
        eth_df = pd.read_csv(eth_csv_path, parse_dates=["Date"])
        print("\n=== ETH features CSV – İlk 5 satır ===")
        print(eth_df.head().to_string(index=False))
        print("\nETH CSV sütun sayısı:", len(eth_df.columns))
    else:
        print("\nETH features CSV bulunamadı:", eth_csv_path)

    # ----------------------------------------------------------------------
    # 3. Oluşan Numpy Dosyalarının Şekillerini Kontrol Etmek
    # ----------------------------------------------------------------------
    btc_X_path = os.path.join(project_root, "data", "processed", "btc_X_scaled.npy")
    btc_y_price_path = os.path.join(project_root, "data", "processed", "btc_y_price.npy")
    btc_y_dir_path = os.path.join(project_root, "data", "processed", "btc_y_dir.npy")

    eth_X_path = os.path.join(project_root, "data", "processed", "eth_X_scaled.npy")
    eth_y_price_path = os.path.join(project_root, "data", "processed", "eth_y_price.npy")
    eth_y_dir_path = os.path.join(project_root, "data", "processed", "eth_y_dir.npy")

    if os.path.exists(btc_X_path):
        X_btc = np.load(btc_X_path)
        y_price_btc = np.load(btc_y_price_path)
        y_dir_btc = np.load(btc_y_dir_path)
        print("\n--- BTC Numpy Dizileri Şekilleri ---")
        print("BTC X_scaled shape:", X_btc.shape)
        print("BTC y_price shape :", y_price_btc.shape)
        print("BTC y_dir shape   :", y_dir_btc.shape)
    else:
        print("\nBTC numpy dosyaları bulunamadı")

    if os.path.exists(eth_X_path):
        X_eth = np.load(eth_X_path)
        y_price_eth = np.load(eth_y_price_path)
        y_dir_eth = np.load(eth_y_dir_path)
        print("\n--- ETH Numpy Dizileri Şekilleri ---")
        print("ETH X_scaled shape:", X_eth.shape)
        print("ETH y_price shape  :", y_price_eth.shape)
        print("ETH y_dir shape    :", y_dir_eth.shape)
    else:
        print("\nETH numpy dosyaları bulunamadı")

    # ----------------------------------------------------------------------
    # 4. Oluşan Scaler Objelerini İncelemek (Opsiyonel)
    # ----------------------------------------------------------------------
    btc_scaler_path = os.path.join(project_root, "models", "btc_scaler.pkl")
    eth_scaler_path = os.path.join(project_root, "models", "eth_scaler.pkl")

    if os.path.exists(btc_scaler_path):
        with open(btc_scaler_path, "rb") as f:
            btc_scaler = pickle.load(f)
        print("\n--- BTC Scaler Bilgileri ---")
        print("BTC scaler feature_range:", btc_scaler.feature_range)
        print("BTC scaler data_min_   :", btc_scaler.data_min_)
        print("BTC scaler data_max_   :", btc_scaler.data_max_)
    else:
        print("\nBTC scaler dosyası bulunamadı:", btc_scaler_path)

    if os.path.exists(eth_scaler_path):
        with open(eth_scaler_path, "rb") as f:
            eth_scaler = pickle.load(f)
        print("\n--- ETH Scaler Bilgileri ---")
        print("ETH scaler feature_range:", eth_scaler.feature_range)
        print("ETH scaler data_min_    :", eth_scaler.data_min_)
        print("ETH scaler data_max_    :", eth_scaler.data_max_)
    else:
        print("\nETH scaler dosyası bulunamadı:", eth_scaler_path)

    print("\nTüm kontroller tamamlandı. Lütfen çıktıları inceleyin ve bana gönderin!")


if __name__ == "__main__":
    main()
