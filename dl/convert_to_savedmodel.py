# convert_savedmodel.py  (proje kökü)
import tensorflow as tf
from pathlib import Path

COINS = ["btc", "eth"]
ARCHS = ["cnn", "lstm", "lstm_mt"]
MODEL_DIR = Path("dl/outputs/models")

for coin in COINS:
    for arch in ARCHS:
        h5  = MODEL_DIR / f"{coin}_{arch}.h5"
        if not h5.exists():
            print(f"⏭  Bulunamadı: {h5.name}")
            continue

        dst = MODEL_DIR / f"{coin}_{arch}_tf"   # klasör adı
        print(f"↻  Dönüştürülüyor {h5.name}  →  {dst}/")

        model = tf.keras.models.load_model(h5, compile=False)
        model.export(dst)          # ✅ Keras-3'te SavedModel export
