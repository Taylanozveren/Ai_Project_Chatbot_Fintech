# convert_savedmodel.py   (proje kökünde)
import tensorflow as tf
from pathlib import Path

ARCHS = ["cnn", "lstm", "lstm_mt"]
COINS = ["btc", "eth"]

MODEL_DIR = Path("dl/outputs/models")

for coin in COINS:
    for arch in ARCHS:
        src = MODEL_DIR / f"{coin}_{arch}.h5"
        if not src.exists():
            print(f"⏭  Bulunamadı: {src}")
            continue

        dst = MODEL_DIR / f"{coin}_{arch}_tf"   # klasör adı
        print(f"↻  Dönüştürülüyor {src.name}  →  {dst}/")

        model = tf.keras.models.load_model(src, compile=False)
        # Keras 3’te klasöre kaydetmek için .export() kullan
        model.export(dst)        # <-- EN KRİTİK SATIR
