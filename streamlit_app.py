# streamlit_app.py  ───────────────────────────────────────────
"""
Crypto-Momentum Dashboard  (ML layer)
─────────────────────────────────────
▪ Asset selector (BTC / ETH)
▪ Today’s long-probability KPI  + recommended action
▪ Walk-forward AUC heat-map
▪ Simple back-test vs. Buy-&-Hold (+ CSV download & threshold curve)
▪ Confusion-matrix on 2024 slice
▪ Top-10 SHAP contributions  (bar & optional waterfall)
"""

# ==== core & local imports ====
import sys, pathlib, datetime as dt
import streamlit as st
import pandas as pd, numpy as np, joblib, shap
import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))          # helper path
from backtest_helper import backtest

# ==== static config ====
DATA   = ROOT / "dashboard_data"
RES    = ROOT / "results"
COINS  = ["btc", "eth"]
TARGET = "bin_h3_thr2"                      # 3-day / +2 % label
TEST0  = np.datetime64("2024-01-01")

# ==== cached loaders ====
@st.cache_data(show_spinner=False)
def load_artifacts(coin: str):
    df    = pd.read_parquet(DATA / f"{coin}_full.parquet")
    mdl   = joblib.load     (DATA / f"{coin}_model.pkl")
    shapv = np.load         (DATA / f"{coin}_shap.npy")
    feats = joblib.load     (DATA /  "feature_list.pkl")
    return df, mdl, shapv, feats

@st.cache_data(show_spinner=False)
def load_walk():
    fp = RES / "walk_metrics_v6.csv"
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()

# ═════════ UI ═════════
st.sidebar.title("⚙️  Settings")
coin = st.sidebar.selectbox("Asset", COINS)
thr  = st.sidebar.slider("Long-signal threshold",
                         0.00, 0.50, 0.02, 0.01,
                         help="If model-probability > threshold ➜ go LONG next day")

st.title("🔮 Crypto Momentum – Machine-Learning Panel")
st.caption(f"Target = **{TARGET}** → price rises **> 2 %** within **3 days**")

with st.expander("ℹ️  Pipeline summary"):
    st.markdown(
        """
1. **Data prep** – Daily OHLCV + social-sentiment → 28 technical / sentiment features  
2. **Model** – LightGBM binary classifier, walk-forward-validated (3 rolling windows)  
3. **Signal** – if *P(long)* > threshold ➜ 100 % long next day (no leverage, 5 bps fee)  
4. **Explain** – SHAP values show which features push the probability up/down
        """.strip()
    )

# ==== load data ====
df, model, shap_vals, feat_names = load_artifacts(coin)
walk = load_walk()

today = df.iloc[-1]
prob  = float(today["prob"])

# ==== KPI card ====
st.markdown(f"<h3 style='text-align:center'>Long probability today: "
            f"<strong>{prob:.1%}</strong></h3>", unsafe_allow_html=True)
st.progress(min(prob/0.5, 1.0))

is_long = prob > thr
color   = "#16c172" if is_long else "#ffb703"
action  = "📈 <strong>GO LONG</strong>" if is_long else "⏸️ <strong>STAY FLAT</strong>"
st.markdown(f"<h2 style='text-align:center;color:{color}'>{action}</h2>",
            unsafe_allow_html=True)

# ==== walk-forward heat-map ====
if not walk.empty:
    st.subheader("🚦 Walk-forward AUC (3 windows)")
    auc_tab = (walk.pivot_table("auc", ["coin", "target"], "step")
                     .loc[(coin, slice(None))])
    st.dataframe(auc_tab.style.background_gradient(cmap="Blues").format("{:.3f}"))

# ==== back-test section ====
# ── back-test (line_chart) ———————————————
st.subheader("📉 Back-test vs. Buy-&-Hold")

equity = backtest(df[["Date", "Close", "prob"]].copy(), thr)

st.line_chart(             # x eksenini açıkça belirt
    equity, x="Date", y=["Equity", "Buy&Hold"],
    use_container_width=True
)

csv_bytes = equity.to_csv(index=False).encode()
st.download_button("📥 Download equity curve (CSV)",
                   csv_bytes, f"{coin}_equity_curve.csv")

# ▸ OPTIONAL: threshold-sensitivity curve
with st.expander("🔧 Threshold sensitivity"):
    test = df[df["Date"].values >= TEST0]
    x, y = [], []
    for t in np.linspace(0, 0.5, 51):
        eq = backtest(test[["Date","Close","prob"]].copy(), t)["Equity"].iloc[-1]
        x.append(t); y.append(eq)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(x, y); ax.axvline(thr, ls="--", c="r")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Final equity vs. start")
    st.pyplot(fig)

# ==== confusion-matrix (2024 slice) ====
st.subheader("🧐 Confusion matrix • 2024 test subset")
mask   = df["Date"].values >= TEST0
y_true = df.loc[mask, TARGET].astype(int)
y_prob = df.loc[mask, "prob"]
y_pred = (y_prob > thr).astype(int)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues",
            xticklabels=["Flat","Long"], yticklabels=["Flat","Long"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown(f"*AUC = **{roc_auc_score(y_true, y_prob):.3f}**, "
            f"accuracy = **{accuracy_score(y_true, y_pred):.3f}***")

# ==== SHAP explanations ====
st.subheader("⭐  Top-10 feature contributions (today)")
imp = (pd.Series(shap_vals[-1], index=feat_names)
         .abs().nlargest(10).sort_values())
st.bar_chart(imp)

with st.expander("🔍  Full SHAP waterfall (today)"):
    shap.initjs()
    base_val = shap.TreeExplainer(model).expected_value
    base_val = np.atleast_1d(base_val)[-1]              # sadece pozitif sınıf

    expl = shap.Explanation(
        values        = shap_vals[-1],
        base_values   = base_val,
        data          = df.loc[today.name, feat_names],
        feature_names = feat_names
    )
    shap.plots.waterfall(expl, max_display=14, show=False)
    plt.gca().set_ylabel("")            # “^” kaynaklı MathText hatasını önler
    st.pyplot(plt.gcf(), clear_figure=True)
