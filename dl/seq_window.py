import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --------------------
# Determine BASE_DIR robustly for both script and interactive use
# --------------------
def get_base_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        cwd = Path.cwd()
        if (cwd / "data" / "processed").exists():
            return cwd
        elif (cwd.parent / "data" / "processed").exists():
            return cwd.parent
        else:
            return cwd

BASE_DIR   = get_base_dir()
SRC_PATH   = BASE_DIR / "data" / "processed"
OUT_PATH   = BASE_DIR / "dl" / "outputs"
SCALER_DIR = BASE_DIR / "dl" / "scalers"

# --------------------
# Parameters
# --------------------
SEQ_LEN       = 60               # sequence length in days
TARGET_CLASS  = 'bin_h3_thr2'    # baseline single-output (3-day / 2%)
TARGETS_MULTI = {
    'h1': 'bin_h1_thr2',         # 1-day / 2%
    'h3': 'bin_h3_thr2',         # 3-day / 2%
    'h5': 'bin_h5_thr2'          # 5-day / 2%
}
TARGET_R3     = 'logret_h3'      # 3-day log-return regression
TARGET_R5     = 'logret_h5'      # 5-day log-return regression

# --------------------
# Function: make_windows
# --------------------
def make_windows(sym: str):
    """
    Builds sliding windows for classification and regression:
      - Binary: h1, h3, h5
      - Regression: r3 (logret_h3), r5 (logret_h5)
    Saves arrays to compressed .npz and scaler parameters.
    """
    df = pd.read_csv(SRC_PATH / f"{sym}_features_ml_v6.csv", parse_dates=["Date"])

    # debug: sütun listesini ve feature sayısını yazdıralım
    print(f"\n[{sym}] CSV sütunları ({len(df.columns)}): {df.columns.tolist()}")

    # features: exclude Date, classification & regression targets
    exclude = ['Date', TARGET_CLASS] + list(TARGETS_MULTI.values()) + [TARGET_R3, TARGET_R5]
    feat_cols = [c for c in df.columns if c not in exclude]
    print(f"[{sym}] feature olarak kullanılacak sütun sayısı: {len(feat_cols)}\n")
    X_raw = df[feat_cols].values

    # classification targets
    y_class = df[TARGET_CLASS].values
    y_h1    = df[TARGETS_MULTI['h1']].values
    y_h3    = df[TARGETS_MULTI['h3']].values
    y_h5    = df[TARGETS_MULTI['h5']].values
    # regression targets
    y_r3    = df[TARGET_R3].values
    y_r5    = df[TARGET_R5].values

    # scale features
    scaler   = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    # sliding windows
    X_seq, yc_seq, y1_seq, y3_seq, y5_seq, yr3_seq, yr5_seq = [], [], [], [], [], [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        end = i + SEQ_LEN
        X_seq.append(X_scaled[i:end])
        yc_seq.append(y_class[end])
        y1_seq.append(y_h1[end])
        y3_seq.append(y_h3[end])
        y5_seq.append(y_h5[end])
        yr3_seq.append(y_r3[end])
        yr5_seq.append(y_r5[end])

    # convert to arrays
    X_seq    = np.array(X_seq, dtype=np.float32)
    yc_seq   = np.array(yc_seq, dtype=np.float32)
    y1_seq   = np.array(y1_seq, dtype=np.float32)
    y3_seq   = np.array(y3_seq, dtype=np.float32)
    y5_seq   = np.array(y5_seq, dtype=np.float32)
    yr3_seq  = np.array(yr3_seq, dtype=np.float32)
    yr5_seq  = np.array(yr5_seq, dtype=np.float32)

    # save sequences
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH / f"{sym}_seq.npz",
        X=X_seq,
        y_class=yc_seq,
        y_h1=y1_seq,
        y_h3=y3_seq,
        y_h5=y5_seq,
        y_r3=yr3_seq,
        y_r5=yr5_seq
    )

    # save scaler params
    SCALER_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SCALER_DIR / f"{sym}_scaler_mean.npy", scaler.mean_)
    np.save(SCALER_DIR / f"{sym}_scaler_var.npy", scaler.var_)

    print(f"[✓] {sym.upper()} sequence tensors created:")
    print(f"    X.shape={X_seq.shape}")
    print(f"    y_class={yc_seq.shape}, y_h1={y1_seq.shape}, y_h3={y3_seq.shape}, y_h5={y5_seq.shape}")
    print(f"    y_r3={yr3_seq.shape}, y_r5={yr5_seq.shape}\n")

# --------------------
# Main entrypoint
# --------------------
if __name__ == '__main__':
    for coin in ("btc", "eth"):
        make_windows(coin)
