# optuna_dl_threshold.py – DL threshold & hold tuner
"""
Run Optuna to find the best (thr, hold) combo for each DL strategy.
Usage examples
--------------
# tüm coinler için 150’er deneme
python dl/optuna_dl_threshold.py

# sadece BTC, 300 deneme
python dl/optuna_dl_threshold.py --coin btc --trials 300
"""
import argparse
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from backtest_helper import backtest

# ────────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ────────────────────────────────────────────────────────────────────────────────
DL_OUT_DIR  = Path("dl/outputs")
DATA_DIR    = Path("dashboard_data")
RESULT_DIR  = Path("results")
RESULT_DIR.mkdir(exist_ok=True)
COINS       = ["btc", "eth"]          # genişletmek istersen buraya ekle
SEQ_CACHE   = {}

def load_price_df(coin: str) -> pd.DataFrame:
    fp = DATA_DIR / f"{coin}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return (pd.read_parquet(fp, columns=["Date", "Close"])
              .sort_values("Date").reset_index(drop=True))

def get_seq_len(coin: str) -> int:
    if coin not in SEQ_CACHE:
        seq_path = DL_OUT_DIR / f"{coin}_seq.npz"
        if not seq_path.exists():
            raise FileNotFoundError(seq_path)
        SEQ_CACHE[coin] = int(np.load(seq_path)["X"].shape[1])
    return SEQ_CACHE[coin]

def build_bt_frame(coin: str) -> pd.DataFrame:
    prob_path = DL_OUT_DIR / f"{coin}_dl_prob.npy"
    if not prob_path.exists():
        raise FileNotFoundError(prob_path)
    prob      = np.load(prob_path)
    price_df  = load_price_df(coin)
    seq_len   = get_seq_len(coin)
    start_idx = seq_len - 1                         # 1. tahminin hizalandığı satır
    if len(price_df) < start_idx + len(prob):
        raise ValueError("Price df < predictions — alignment error.")
    df_bt = price_df.iloc[start_idx:start_idx + len(prob)].copy()
    df_bt["prob"] = prob[:len(df_bt)]
    return df_bt

def objective_factory(df: pd.DataFrame):
    def _objective(trial: optuna.Trial) -> float:
        thr  = trial.suggest_float("thr", 0.00, 0.25)
        hold = trial.suggest_int("hold", 1, 7)
        equity = backtest(df, thr=thr, hold=hold)["Equity"].iloc[-1]
        return -float(equity)        # minimize → maximize equity
    return _objective

def run_optuna_for_coin(coin: str, n_trials: int):
    df_bt  = build_bt_frame(coin)
    study  = optuna.create_study(direction="minimize",
                                 study_name=f"dl-{coin}",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(df_bt), n_trials=n_trials, show_progress_bar=True)

    best = study.best_params | {"equity": -study.best_value}
    out_path = RESULT_DIR / f"dl_thr_hold_{coin}.csv"
    pd.DataFrame([best]).to_csv(out_path, index=False)
    print(f"[✓] {coin.upper()} best → {best}   ⇒  {out_path}")

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",   default="all", help="'btc' / 'eth' / 'all'")
    parser.add_argument("--trials", type=int, default=150, help="Optuna trials per coin")
    args = parser.parse_args()

    target_coins = COINS if args.coin.lower() == "all" else [args.coin.lower()]
    for c in target_coins:
        print(f"\n──── Optimizing {c.upper()} ({args.trials} trials) ────")
        run_optuna_for_coin(c, args.trials)
