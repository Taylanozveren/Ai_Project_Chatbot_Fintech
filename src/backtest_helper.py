import pandas as pd
import numpy as np

def backtest(
        df: pd.DataFrame,
        thr: float = 0.02,
        fee: float = 0.0005,
        hold: int = 3          # ← kaç gün pozisyonda kalalım?
) -> pd.DataFrame:
    """
    Basit long/flat strateji back-test'i.

    Parameters
    ----------
    df   : DataFrame  -> en az ["Date", "Close", "prob"] sütunları
    thr  : float      -> long sinyali eşiği  (prob > thr)
    fee  : float      -> tek yön komisyon (örn. 0.0005 = 5 bps)
    hold : int        -> sinyalden sonra pozisyonu kaç gün elde tut (horizon)

    Returns
    -------
    DataFrame  -> ["Date", "Equity", "Buy&Hold"]
    """

    ### 1) Pozisyon sinyali (horizon-aware)
    sig = np.zeros(len(df), dtype=int)
    raw_long = (df["prob"].values > thr).astype(int)

    # aynı sinyali 'hold' gün ileriye taşı
    for i in range(hold, len(sig)):
        sig[i] = raw_long[i - hold]

    ### 2) Günlük getiriler
    ret = df["Close"].pct_change().fillna(0).values

    ### 3) Strateji getirisi + komisyon
    # pozisyon değişimi  (|sig_t - sig_{t-1}|)  -> 0 veya 1
    turn = np.abs(np.diff(np.insert(sig, 0, 0)))
    strat_ret = sig * ret - fee * turn

    ### 4) Kümülatif eğriler
    equity   = np.cumprod(1 + strat_ret)
    buyhold  = np.cumprod(1 + ret)

    out = pd.DataFrame({
        "Date": df["Date"].values,
        "Equity": equity,
        "Buy&Hold": buyhold
    })

    return out
