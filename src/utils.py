# utils.py

import pandas as pd

# Default out‑of‑sample start for all back‑tests
TEST_START = "2022-01-01"


def get_bt_slice(df: pd.DataFrame, start_date: str = TEST_START) -> pd.DataFrame:
    """Return the dataframe slice starting from *start_date* (inclusive).

    Ensures that the ``Date`` column is in *datetime* format, keeps
    original ordering, and resets the index for clean downstream use.
    """
    if "Date" not in df.columns:
        raise KeyError("DataFrame must contain a 'Date' column.")

    out = df.copy()
    if out["Date"].dtype == "object":
        out["Date"] = pd.to_datetime(out["Date"])

    mask = out["Date"] >= pd.to_datetime(start_date)
    return out.loc[mask].reset_index(drop=True)
