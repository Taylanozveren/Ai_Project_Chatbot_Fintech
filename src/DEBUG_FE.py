# dl/debug_models.py
"""
Model debugging ve data quality check scripti
CNN vs LSTM performans farkını analiz eder
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SEQ_DIR = BASE_DIR / "outputs"
MODEL_DIR = SEQ_DIR / "models"


def analyze_model_predictions():
    """Model tahminlerini detaylı analiz et"""

    # Veri yükle
    btc_data = np.load(SEQ_DIR / "btc_seq.npz")
    X, y_class = btc_data['X'], btc_data['y_class']

    print("=== VERİ ANALİZİ ===")
    print(f"X shape: {X.shape}")
    print(f"y_class shape: {y_class.shape}")
    print(f"y_class unique values: {np.unique(y_class, return_counts=True)}")
    print(f"Class distribution: {np.mean(y_class):.3f} positive")

    # Son test bölümünü al
    test_size = 500
    X_test = X[-test_size:]
    y_test = y_class[-test_size:]

    print(f"\nTEST SET:")
    print(f"Test size: {len(X_test)}")
    print(f"Test class dist: {np.mean(y_test):.3f} positive")

    # Model tahminleri
    models = {}
    for arch in ['lstm', 'cnn']:
        model_path = MODEL_DIR / f"btc_{arch}.h5"
        if model_path.exists():
            models[arch] = tf.keras.models.load_model(model_path)
            print(f"\n{arch.upper()} MODEL YÜKLENDİ")

    # Tahmin analizi
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test, verbose=0)
        if isinstance(pred, list):
            pred = pred[0]  # Multi-output durumu
        pred = pred.ravel()
        predictions[name] = pred

        print(f"\n{name.upper()} PREDICTIONS:")
        print(f"  Min: {pred.min():.6f}")
        print(f"  Max: {pred.max():.6f}")
        print(f"  Mean: {pred.mean():.6f}")
        print(f"  Std: {pred.std():.6f}")
        print(f"  >0.5: {np.sum(pred > 0.5)} ({np.sum(pred > 0.5) / len(pred):.3f})")
        print(f"  >0.1: {np.sum(pred > 0.1)} ({np.sum(pred > 0.1) / len(pred):.3f})")
        print(f"  <0.001: {np.sum(pred < 0.001)} ({np.sum(pred < 0.001) / len(pred):.3f})")

        # Histogram
        bins = [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
        hist, _ = np.histogram(pred, bins=bins)
        print(f"  Histogram: {dict(zip([f'{bins[i]:.3f}-{bins[i + 1]:.3f}' for i in range(len(bins) - 1)], hist))}")


def check_model_architecture():
    """Model mimarilerini karşılaştır"""

    for arch in ['lstm', 'cnn']:
        model_path = MODEL_DIR / f"btc_{arch}.h5"
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            print(f"\n=== {arch.upper()} ARCHITECTURE ===")
            model.summary()

            # Son layer aktivasyonu kontrol et
            last_layer = model.layers[-1]
            print(f"Last layer: {last_layer.name}")
            print(f"Last layer activation: {last_layer.activation}")

            # Compile settings
            print(f"Optimizer: {model.optimizer}")
            print(f"Loss: {model.loss}")


def compare_training_data():
    """Eğitim verisi kalitesini kontrol et"""

    # Feature verisi yükle
    df = pd.read_csv(BASE_DIR.parent / "data" / "processed" / "btc_features_ml_v6.csv")

    print("=== FEATURE DATA QUALITY ===")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # NaN analizi
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"\nNaN counts:")
        print(nan_counts[nan_counts > 0])

    # Infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    if inf_counts.sum() > 0:
        print(f"\nInfinite values:")
        print(inf_counts[inf_counts > 0])

    # Target distribution
    target_cols = [c for c in df.columns if c.startswith('bin_')]
    if target_cols:
        print(f"\nTarget distributions:")
        for col in target_cols:
            if col in df.columns:
                dist = df[col].value_counts().sort_index()
                print(f"  {col}: {dict(dist)}")


if __name__ == "__main__":
    print("🔍 MODEL DEBUG ANALIZI BAŞLADI")
    print("=" * 50)

    try:
        analyze_model_predictions()
    except Exception as e:
        print(f"Prediction analysis error: {e}")

    try:
        check_model_architecture()
    except Exception as e:
        print(f"Architecture analysis error: {e}")

    try:
        compare_training_data()
    except Exception as e:
        print(f"Training data analysis error: {e}")

    print("\n✅ ANALIZ TAMAMLANDI")