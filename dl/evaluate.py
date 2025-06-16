# dl/evaluate.py
"""
FULL RUN TERMINAL CODES FOR EVALUATING:
python dl/train.py --sym btc --arch cnn      --epochs 30 --batch 128
python dl/train.py --sym btc --arch lstm     --epochs 30 --batch 128
python dl/train.py --sym btc --arch lstm_mt  --epochs 30 --batch 128
python dl/train.py --sym eth --arch cnn      --epochs 30 --batch 128
python dl/train.py --sym eth --arch lstm     --epochs 30 --batch 128
python dl/train.py --sym eth --arch lstm_mt  --epochs 30 --batch 128

Evaluate DL models with walk-forward splits and compare performance.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error
)
from pathlib import Path

# ───────── Configuration ─────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data" / "processed"
SEQ_DIR    = BASE_DIR / "outputs"
MODEL_DIR  = SEQ_DIR / "models"
OUTPUT_DIR = BASE_DIR / "reports"

SPLITS     = [np.datetime64("2022-01-01"),
              np.datetime64("2023-01-01"),
              np.datetime64("2024-01-01")]

ARCHS      = ["lstm", "cnn", "lstm_mt"]
SYMBOLS    = ["btc", "eth"]
THRESHOLD  = 0.50   # classification cut-off
# ─────────────────────────────────

# ---------- Data helpers ----------
def load_sequence(sym: str):
    """Return X, y_class, y_multi from .npz"""
    arr = np.load(SEQ_DIR / f"{sym}_seq.npz")
    X        = arr["X"]
    y_class  = arr["y_class"]
    y_multi  = {k: arr[k] for k in arr.files if k.startswith("y_") and k != "y_class"}
    return X, y_class, y_multi


def load_dates(sym: str, seq_len: int):
    df = pd.read_csv(DATA_DIR / f"{sym}_features_ml_v6.csv", parse_dates=["Date"])
    return df["Date"].values[seq_len:]


def load_model(sym: str, arch: str):
    """
    SavedModel klasörünü (‘btc_lstm_tf’ vb.) yükler.
    Modeller compile=False ile only-inference modunda açılır.
    """
    model_path = MODEL_DIR / f"{sym}_{arch}_tf"      # *** tek fark burası ***
    if not model_path.exists():
        print(f"[!] Model not found: {model_path}, skipping {sym} {arch}")
        return None
    return tf.keras.models.load_model(model_path, compile=False)
# ----------------------------------

def evaluate_model(sym: str, arch: str):
    model = load_model(sym, arch)
    if model is None:
        return []

    X, y_class, y_multi = load_sequence(sym)
    dates = load_dates(sym, seq_len=X.shape[1])
    results = []

    for step, split in enumerate(SPLITS, start=1):
        mask   = dates >= split
        X_test = X[mask]
        if X_test.size == 0:
            continue

        preds = model.predict(X_test, verbose=0)
        if not isinstance(preds, list):
            preds = [preds]

        # ---- single-output (lstm / cnn) ----
        if arch in {"lstm", "cnn"}:
            y_test = y_class[mask]
            prob   = preds[0].ravel()
            pred   = (prob > THRESHOLD).astype(int)
            results.append({
                "coin": sym, "arch": arch, "horizon": "h3", "step": step,
                "auc": round(roc_auc_score(y_test, prob), 3),
                "accuracy": round(accuracy_score(y_test, pred), 3),
                "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, pred, zero_division=0), 3),
                "mse": None, "mae": None, "n_test": int(len(y_test))
            })
        # ---- multi-output (lstm_mt) ----
        else:
            # first three outputs = classification (h1,h3,h5)
            for arr, horizon in zip(preds[:3], ["h1", "h3", "h5"]):
                y_test = y_multi[f"y_{horizon}"][mask]
                prob   = arr.ravel()
                pred   = (prob > THRESHOLD).astype(int)
                results.append({
                    "coin": sym, "arch": arch, "horizon": horizon, "step": step,
                    "auc": round(roc_auc_score(y_test, prob), 3),
                    "accuracy": round(accuracy_score(y_test, pred), 3),
                    "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                    "recall": round(recall_score(y_test, pred, zero_division=0), 3),
                    "mse": None, "mae": None, "n_test": int(len(y_test))
                })
            # last two outputs = regression (r3,r5)
            for arr, horizon in zip(preds[3:], ["r3", "r5"]):
                y_true = y_multi[f"y_{horizon}"][mask]
                y_pred = arr.ravel()
                valid  = ~(np.isnan(y_true) | np.isnan(y_pred))
                mse = mae = n = None
                if valid.any():
                    mse = round(mean_squared_error(y_true[valid], y_pred[valid]), 3)
                    mae = round(mean_absolute_error(y_true[valid], y_pred[valid]), 3)
                    n   = int(valid.sum())
                results.append({
                    "coin": sym, "arch": arch, "horizon": horizon, "step": step,
                    "auc": None, "accuracy": None, "precision": None, "recall": None,
                    "mse": mse, "mae": mae, "n_test": n
                })
    return results

# ---------- CLI ----------
if __name__ == "__main__":
    all_res = []
    for sym in SYMBOLS:
        for arch in ARCHS:
            all_res.extend(evaluate_model(sym, arch))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "dl_walkforward.csv"
    pd.DataFrame(all_res).to_csv(out_csv, index=False)

    print("\n=== CLASSIFICATION METRICS (AUC, ACC, PREC, REC) ===")
    print(pd.DataFrame(all_res)[lambda d: d.horizon.isin(["h1", "h3", "h5"])]
          .to_string(index=False))

    print("\n=== REGRESSION METRICS (MSE, MAE) ===")
    print(pd.DataFrame(all_res)[lambda d: d.horizon.isin(["r3", "r5"])]
          .to_string(index=False))

    print(f"\n[✓] Walk-forward evaluation saved → {out_csv}")
