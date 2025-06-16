# backtest_helper.py
import numpy as np
import pandas as pd

TEST_START = "2022-01-01"       # OOS başlangıcı
RET_CAP    = 0.15               # Günlük ±%15 limit
EQ_CAP     = 50                 # Equity upper-guard: bh * 50

def backtest(
        df: pd.DataFrame,
        thr: float = 0.02,
        fee: float = 0.0005,
        hold: int = 3,
        oos_start: str = TEST_START,
) -> pd.DataFrame:
    """
    Long/Flat strateji (overlap-fix’li) back-test.
        • Sinyal günü LONG açılır, + (hold-1) gün tutulur
        • Aynı sinyal periyodu içinde yeniden LONG açılmaz (overlap yok)
        • Pozisyon açılışında tek yön komisyon (fee) uygulanır
        • OOS periyodu: oos_start ve sonrası
        • Günlük getiri ±RET_CAP içinde kırpılır, equity bh*EQ_CAP’i aşamaz
    """
    if df.empty:
        return df.assign(Equity=1.0, **{"Buy&Hold": 1.0})

    # — OOS dilimini seç —
    df = df.loc[df["Date"] >= oos_start].reset_index(drop=True)

    # 1) ham long sinyali
    raw = (df["prob"].values > thr).astype(int)

    # 2) overlap-fix: hold boyunca tekrar sinyal açma
    sig = np.zeros_like(raw)
    i = 0
    while i < len(raw):
        if raw[i]:
            sig[i:i+hold] = 1
            i += hold        # pozisyon süresi kadar atla
        else:
            i += 1

    # 3) günlük getiriler
    ret = df["Close"].pct_change().fillna(0).clip(-RET_CAP, RET_CAP).values

    # 4) strateji getirisi
    enter = (np.diff(np.insert(sig, 0, 0)) == 1).astype(int)
    strat_ret = sig * ret - fee * enter

    # 5) kümülatif
    equity  = np.cumprod(1 + strat_ret)
    bh      = np.cumprod(1 + ret)
    equity  = np.minimum(equity, bh * EQ_CAP)      # güvenlik tavanı

    return pd.DataFrame({
        "Date": df["Date"].values,
        "Equity": equity,
        "Buy&Hold": bh
    })
