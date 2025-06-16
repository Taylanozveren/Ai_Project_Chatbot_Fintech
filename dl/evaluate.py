"""
Walk-forward evaluation for all DL models
Run after every re-train to update dl/reports/dl_walkforward.csv
----------------------------------------------------------------
Example full training suite (optional):
python dl/train.py --sym btc --arch cnn      --epochs 30 --batch 128
python dl/train.py --sym btc --arch lstm     --epochs 30 --batch 128
python dl/train.py --sym btc --arch lstm_mt  --epochs 30 --batch 128
python dl/train.py --sym eth --arch cnn      --epochs 30 --batch 128
python dl/train.py --sym eth --arch lstm     --epochs 30 --batch 128
python dl/train.py --sym eth --arch lstm_mt  --epochs 30 --batch 128
"""

from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error,
)

# ───────── paths & consts ────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data" / "processed"
SEQ_DIR    = BASE_DIR / "outputs"
MODEL_DIR  = SEQ_DIR / "models"
OUTPUT_DIR = BASE_DIR / "reports"

ARCHS      = ["lstm", "cnn", "lstm_mt"]
SYMBOLS    = ["btc", "eth"]
SPLITS     = [np.datetime64("2022-01-01"),
              np.datetime64("2023-01-01"),
              np.datetime64("2024-01-01")]
THRESHOLD  = 0.50           # prob > thr  → long
# ──────────────────────────────────────────────────────────────────────

# ---------- helpers --------------------------------------------------
def load_sequence(sym: str):
    """X  |  y_class  |  y_multi  döndürür"""
    arr      = np.load(SEQ_DIR / f"{sym}_seq.npz")
    X        = arr["X"]
    y_class  = arr["y_class"]
    y_multi  = {k: arr[k] for k in arr.files if k.startswith("y_") and k != "y_class"}
    return X, y_class, y_multi


def load_dates(sym: str, seq_len: int):
    df = pd.read_csv(DATA_DIR / f"{sym}_features_ml_v6.csv", parse_dates=["Date"])
    return df["Date"].values[seq_len:]


def load_model(sym: str, arch: str):
    """
    Önce .h5 checkpoint'ini dene. Yoksa TFSMLayer ile SavedModel klasörünü yükle.
    """
    h5 = MODEL_DIR / f"{sym}_{arch}.h5"
    if h5.exists():
        return tf.keras.models.load_model(h5, compile=False)

    sm = MODEL_DIR / f"{sym}_{arch}_tf"
    if sm.exists():
        # inference-only layer – compile gerekmez
        from keras.layers import TFSMLayer
        return TFSMLayer(str(sm), call_endpoint="serve")

    print(f"[!] Model not found for {sym} {arch}")
    return None
# ---------------------------------------------------------------------

def evaluate_model(sym: str, arch: str):
    model = load_model(sym, arch)
    if model is None:
        return []

    X, y_class, y_multi = load_sequence(sym)
    dates               = load_dates(sym, seq_len=X.shape[1])

    res = []
    for step, split in enumerate(SPLITS, start=1):
        mask   = dates >= split
        X_test = X[mask]
        if X_test.size == 0:
            continue

        preds = model(X_test, training=False)  # works for both layer & model
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        # ── single-output (lstm / cnn) ────────────────────────────────
        if arch in {"lstm", "cnn"}:
            y_true = y_class[mask]
            prob   = preds[0].numpy().ravel()
            pred   = (prob > THRESHOLD).astype(int)
            res.append(dict(
                coin=sym, arch=arch, horizon="h3", step=step,
                auc=round(roc_auc_score(y_true, prob), 3),
                accuracy=round(accuracy_score(y_true, pred), 3),
                precision=round(precision_score(y_true, pred, zero_division=0), 3),
                recall=round(recall_score(y_true, pred, zero_division=0), 3),
                mse=None, mae=None, n_test=int(len(y_true))
            ))
        # ── multi-task (lstm_mt) ─────────────────────────────────────
        else:
            # ilk 3 çıktı = classification (h1,h3,h5)
            for arr, hz in zip(preds[:3], ["h1", "h3", "h5"]):
                prob   = arr.numpy().ravel()
                y_true = y_multi[f"y_{hz}"][mask]
                pred   = (prob > THRESHOLD).astype(int)
                res.append(dict(
                    coin=sym, arch=arch, horizon=hz, step=step,
                    auc=round(roc_auc_score(y_true, prob), 3),
                    accuracy=round(accuracy_score(y_true, pred), 3),
                    precision=round(precision_score(y_true, pred, zero_division=0), 3),
                    recall=round(recall_score(y_true, pred, zero_division=0), 3),
                    mse=None, mae=None, n_test=int(len(y_true))
                ))
            # son 2 çıktı = regression (r3,r5)
            for arr, hz in zip(preds[3:], ["r3", "r5"]):
                y_t = y_multi[f"y_{hz}"][mask]
                y_p = arr.numpy().ravel()
                valid = ~(np.isnan(y_t) | np.isnan(y_p))
                mse = mae = n = None
                if valid.any():
                    mse = round(mean_squared_error(y_t[valid], y_p[valid]), 3)
                    mae = round(mean_absolute_error(y_t[valid], y_p[valid]), 3)
                    n   = int(valid.sum())
                res.append(dict(
                    coin=sym, arch=arch, horizon=hz, step=step,
                    auc=None, accuracy=None, precision=None, recall=None,
                    mse=mse, mae=mae, n_test=n
                ))
    return res

# ───────── main CLI ─────────
if __name__ == "__main__":
    all_res = []
    for sym in SYMBOLS:
        for arch in ARCHS:
            all_res += evaluate_model(sym, arch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "dl_walkforward.csv"
    pd.DataFrame(all_res).to_csv(out_csv, index=False)

    cls = pd.DataFrame(all_res).query("horizon in ['h1','h3','h5']")
    reg = pd.DataFrame(all_res).query("horizon in ['r3','r5']")

    print("\n=== CLASSIFICATION METRICS ===")
    print(cls.to_string(index=False))
    print("\n=== REGRESSION METRICS ===")
    print(reg.to_string(index=False))
    print(f"\n[✓] Walk-forward saved → {out_csv}")
