import shap, joblib, pandas as pd, numpy as np
from pathlib import Path
IN  = Path("dashboard_data"); OUT = IN

for coin in ("btc","eth"):
    bst = joblib.load(IN / f"{coin}_model.pkl")
    df  = pd.read_parquet(IN / f"{coin}_full.parquet")
    explainer = shap.TreeExplainer(bst, feature_perturbation="tree_path_dependent")
    shap_vals = explainer(df[bst.feature_name()])
    np.save(OUT / f"{coin}_shap.npy", shap_vals.values)
print("[✓] SHAP matrisleri kaydedildi.")
