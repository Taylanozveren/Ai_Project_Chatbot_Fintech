import pandas as pd, numpy as np

def backtest(df: pd.DataFrame, thr: float = 0.02, fee: float = 0.0005) -> pd.DataFrame:
    """
    Basit strateji back-test’i.

    Parametreler
    ------------
    df   : Date, Close, prob sütunlarını içerir
    thr  : Sinyal eşiği  (prob > thr → long %100)
    fee  : İşlem başı tek yön komisyon (ör. 5 bp)

    Döndürür
    --------
    pandas.DataFrame  →  ["Equity", "Buy&Hold"] indeks = Date
    """
    # 1) Pozisyon sinyali
    sig = (df["prob"] > thr).astype(int)

    # 2) Günlük getiri
    ret = df["Close"].pct_change().fillna(0)

    # 3) Strateji getirisi (komisyon dahil)
    strat = (sig.shift() * ret) - fee * sig.diff().abs().clip(lower=0)

    # 4) Kümülatif sermaye eğrileri
    equity = (1 + strat).cumprod()
    bh = (1 + ret).cumprod()

    # ► Date'i kaybetmeyelim; sütun olarak döndür
    out = pd.DataFrame(
        {"Date": df["Date"], "Equity": equity, "Buy&Hold": bh}
    )
    return out.dropna()
