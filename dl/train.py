"""
Script to train DL models on crypto sequence data.
Supports single-output (lstm, cnn) and multi-output (lstm_mt) architectures,
hem classification hem regression başlıkları.
Usage:
    python dl/train.py --sym btc --arch lstm --epochs 30 --batch 128
    python dl/train.py --sym eth --arch lstm_mt --epochs 30 --batch 128
"""
import argparse
import numpy as np
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# model builder’lar
from lstm import build_lstm
from cnn import build_cnn
from lstm_multitask import build_lstm_multitask

# ---- Paths ----
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'outputs'
MODELS_DIR  = DATA_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Data loader ----
def load_data(sym, arch):
    """
    .npz içinden X ve uygun y’leri yükler.
    Single-output için y_class, multi-task için h1,h3,h5,r3,r5.
    """
    arr = np.load(DATA_DIR / f"{sym}_seq.npz")
    X = arr['X']
    if arch == 'lstm_mt':
        y = {
            'h1': arr['y_h1'],
            'h3': arr['y_h3'],
            'h5': arr['y_h5'],
            'r3': arr['y_r3'],
            'r5': arr['y_r5']
        }
    else:
        y = arr['y_class']
    return X, y

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sym', choices=['btc','eth'], required=True)
    parser.add_argument('--arch', choices=['lstm','cnn','lstm_mt'], default='lstm')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()

    # load data
    X, y = load_data(args.sym, args.arch)
    n       = len(X)
    idx     = int(0.8 * n)
    X_train, X_val = X[:idx], X[idx:]
    if args.arch == 'lstm_mt':
        y_train = {k: v[:idx] for k, v in y.items()}
        y_val   = {k: v[idx:] for k, v in y.items()}
    else:
        y_train, y_val = y[:idx], y[idx:]

    # select model
    if args.arch == 'lstm':
        model = build_lstm(input_shape=X_train.shape[1:])
    elif args.arch == 'cnn':
        model = build_cnn(input_shape=X_train.shape[1:])
    else:
        model = build_lstm_multitask(input_shape=X_train.shape[1:])

    # callbacks (checkpoint extension .h5 olarak)
    ckpt_file = f"{args.sym}_{args.arch}.h5"
    ckpt = MODELS_DIR / ckpt_file
    monitor = 'val_h3_auc_h3' if args.arch == 'lstm_mt' else 'val_auc'

    callbacks = [
        EarlyStopping(patience=5, monitor=monitor, mode='max', restore_best_weights=True),
        ModelCheckpoint(filepath=str(ckpt),
                        save_best_only=True,
                        monitor=monitor,
                        mode='max')
    ]

    # train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks
    )

    # ─────────── ✅ DÜZELTİLEN KISIM ────────────
    saved_dir = MODELS_DIR / f"{args.sym}_{args.arch}_tf"  # klasör adı
    model.export(saved_dir)  # <──  model.save(...) DEĞİL!   🚀
    # ────────────────────────────────────────────

    print(f"[✓] Trained {args.arch.upper()} for {args.sym.upper()} → "
          f".h5: {ckpt.name}  |  SavedModel: {saved_dir.name}")


if __name__ == '__main__':
    main()
