# dl/evaluate.py

"""
Evaluate DL models with walk-forward splits and compare performance.
Supports single-output (lstm, cnn) and multi-output (lstm_mt) architectures.
Generates AUC, accuracy, precision, recall for classification and MSE/MAE for regression.
Handles missing model files gracefully.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error
)
from pathlib import Path

# --------------------
# Configuration
# --------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data" / "processed"
SEQ_DIR    = BASE_DIR / "outputs"
MODEL_DIR  = SEQ_DIR / "models"
OUTPUT_DIR = BASE_DIR / "reports"
# walk-forward splits
SPLITS     = [
    np.datetime64("2022-01-01"),
    np.datetime64("2023-01-01"),
    np.datetime64("2024-01-01")
]
# architectures to evaluate
ARCHS      = ["lstm", "cnn", "lstm_mt"]
SYMBOLS    = ["btc", "eth"]
THRESHOLD  = 0.5  # decision threshold for classification

# --------------------
# Data loading
# --------------------
def load_sequence(sym):
    """Load X and all y arrays (classification + regression) from .npz"""
    arr = np.load(SEQ_DIR / f"{sym}_seq.npz")
    X = arr['X']
    y_class = arr['y_class']
    y_multi = {}
    # classification
    for key in ['y_h1','y_h3','y_h5']:
        if key in arr: y_multi[key] = arr[key]
    # regression
    for key in ['y_r3','y_r5']:
        if key in arr: y_multi[key] = arr[key]
    return X, y_class, y_multi

def load_dates(sym, seq_len):
    df = pd.read_csv(DATA_DIR / f"{sym}_features_ml_v6.csv", parse_dates=["Date"])
    return df['Date'].values[seq_len:]

def load_model(sym, arch):
    model_path = MODEL_DIR / f"{sym}_{arch}.h5"
    if not model_path.exists():
        print(f"[!] Model file not found: {model_path}, skipping {sym} {arch}.")
        return None
    return tf.keras.models.load_model(model_path)

# --------------------
# Evaluation
# --------------------
def evaluate_model(sym, arch):
    model = load_model(sym, arch)
    if model is None:
        return []

    X, y_class, y_multi = load_sequence(sym)
    dates = load_dates(sym, seq_len=X.shape[1])

    results = []
    for step, split in enumerate(SPLITS, start=1):
        mask = dates >= split
        X_test = X[mask]
        if len(X_test)==0:
            continue

        preds = model.predict(X_test)
        # ensure list
        if not isinstance(preds, list):
            preds = [preds]

        # single-output (tek çıkışlı) modeller
        if arch in ['lstm','cnn']:
            y_test = y_class[mask]
            prob   = preds[0].ravel()
            pred   = (prob > THRESHOLD).astype(int)
            results.append({
                'coin': sym, 'arch': arch, 'horizon': 'h3', 'step': step,
                'auc': round(roc_auc_score(y_test, prob),3),
                'accuracy': round(accuracy_score(y_test, pred),3),
                'precision': round(precision_score(y_test, pred, zero_division=0),3),
                'recall': round(recall_score(y_test, pred, zero_division=0),3),
                'mse': None, 'mae': None,
                'n_test': int(len(y_test))
            })

        # multi-output modeller
        else:
            # 0:h1, 1:h3, 2:h5, 3:r3, 4:r5
            # classification
            for arr, horizon in zip(preds[:3], ['h1','h3','h5']):
                y_test = y_multi[f'y_{horizon}'][mask]
                prob   = arr.ravel()
                pred   = (prob > THRESHOLD).astype(int)
                results.append({
                    'coin': sym, 'arch': arch, 'horizon': horizon, 'step': step,
                    'auc': round(roc_auc_score(y_test, prob),3),
                    'accuracy': round(accuracy_score(y_test, pred),3),
                    'precision': round(precision_score(y_test, pred, zero_division=0),3),
                    'recall': round(recall_score(y_test, pred, zero_division=0),3),
                    'mse': None, 'mae': None,
                    'n_test': int(len(y_test))
                })
            # regression
            for arr, horizon in zip(preds[3:], ['r3','r5']):
                y_true = y_multi[f'y_{horizon}'][mask]
                y_pred = arr.ravel()
                results.append({
                    'coin': sym, 'arch': arch, 'horizon': horizon, 'step': step,
                    'auc': None, 'accuracy': None, 'precision': None, 'recall': None,
                    'mse': round(mean_squared_error(y_true, y_pred),3),
                    'mae': round(mean_absolute_error(y_true, y_pred),3),
                    'n_test': int(len(y_true))
                })

    return results

# --------------------
# Main execution
# --------------------
if __name__ == '__main__':
    all_results = []
    for sym in SYMBOLS:
        for arch in ARCHS:
            all_results.extend(evaluate_model(sym, arch))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'dl_walkforward.csv'

    df = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)

    # ayrı ayrı bölümler halinde konsola yaz
    df_clf = df[df['horizon'].isin(['h1','h3','h5'])]
    df_reg = df[df['horizon'].isin(['r3','r5'])]

    print("\n=== CLASSIFICATION METRICS (AUC, ACC, PREC, REC) ===")
    print(df_clf.to_string(index=False))

    print("\n=== REGRESSION METRICS (MSE, MAE) ===")
    print(df_reg.to_string(index=False))

    print(f"\n[✓] Walk-forward evaluation saved to {out_path}")
