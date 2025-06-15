"""
Professional Crypto-Momentum Dashboard
Enhanced ML + DL Analytics Platform
"""
# ── TensorFlow + eski (2.x) Keras yapılandırması ─────────────────────────
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # tf.keras 2.x yolunu seçsin

import tensorflow as tf                  # tf.keras bundan sonra hazır
import sys
sys.modules["keras"] = tf.keras          # herhangi bir 'import keras' çağrısı → tf.keras
sys.modules["keras.api._v2.keras"] = tf.keras   # bazı paketler bu yolu kullanıyor
# ─────────────────────────────────────────────────────────────────────────

# ↓ Artık diğer kütüphaneleri gönül rahatlığıyla içe aktarabilirsin
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────────────────────
#    🛈 Backtest & Strategy-Return Explanation (shared by ML & DL panels)
# ──────────────────────────────────────────────────────────────────────────
INFO_TEXT = """
**⚠️ Backtest disclaimer:**  
This dashboard shows *theoretical* performance assuming 100% capital is 
switched long/flat on every signal. Transaction fees, slippage, financing 
costs and position-sizing limits are *not* modeled, so real-world returns 
will be materially lower.

**Why does BTC favor Deep Learning but ETH favor Machine Learning?**  
- **BTC (DL > ML):** Our multi-task LSTM captures 1-day, 3-day and 5-day trends  
  in one model and so rides long bull runs more completely.  
- **ETH (ML > DL):** The LightGBM "3-day, +2%" model issues more aggressive  
  buy signals around ETH's short, sharp rallies—boosting its theoretical return.
"""

# Configuration
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

# Path constants
ML_DATA = ROOT / "dashboard_data"
ML_RES = ROOT / "results"
DL_SEQ_DIR = ROOT / "dl" / "outputs"
DL_MODEL_DIR = DL_SEQ_DIR / "models"
DL_WALK = ROOT / "dl" / "reports" / "dl_walkforward.csv"

ML_COINS = ["btc", "eth"]
DL_TARGETS = {'h1': '1-day', 'h3': '3-day', 'h5': '5-day', 'r3': '3d-return', 'r5': '5d-return'}
ML_TEST_DATE = np.datetime64("2024-01-01")

# Streamlit config
st.set_page_config(page_title="Crypto Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .success-signal {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .warning-signal {
        background: linear-gradient(90deg, #fc4a1a 0%, #f7b733 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Import backtest helper
try:
    from backtest_helper import backtest
except ImportError:
    def backtest(df, threshold):
        equity = df.copy()
        equity["Equity"] = 1.0
        equity["Buy&Hold"] = (df["Close"] / df["Close"].iloc[0])
        return equity

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Settings")
    panel = st.radio("Select Analysis Panel", ["🤖 ML Panel", "🧠 DL Panel"], index=0)

    st.markdown("### 📊 Asset Selection")
    coin = st.selectbox("Choose Cryptocurrency", ML_COINS, format_func=lambda x: x.upper())

    st.markdown("### 🎯 Strategy Parameters")
    threshold = st.slider("Signal Threshold", 0.0, 0.5, 0.02, 0.01, help="Probability threshold for long signals")

    st.markdown("### 📈 Performance Metrics")
    show_metrics = st.checkbox("Show Detailed Metrics", True)
    show_backtest = st.checkbox("Show Backtest", True)

    # Force refresh button to clear cache
    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Show last data update time
    try:
        data_file = ML_DATA / f"{coin}_full.parquet"
        if data_file.exists():
            mod_time = datetime.fromtimestamp(data_file.stat().st_mtime)
            st.caption(f"📅 Data Updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    except:
        pass

# Utility functions
def get_file_hash(file_path):
    """Get file modification time as hash for cache invalidation"""
    try:
        return str(file_path.stat().st_mtime)
    except:
        return str(datetime.now().timestamp())

@st.cache_data(ttl=60)  # Reduced cache time to 1 minute
def load_ml_data(coin_name, _file_hash=None):
    try:
        df = pd.read_parquet(ML_DATA / f"{coin_name}_full.parquet")
        model = joblib.load(ML_DATA / f"{coin_name}_model.pkl")
        shap_vals = np.load(ML_DATA / f"{coin_name}_shap.npy")
        features = joblib.load(ML_DATA / "feature_list.pkl")

        # ✅ Date column handling - ensure proper datetime format
        if 'Date' in df.columns:
            if df['Date'].dtype == 'object':
                df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        return df, model, shap_vals, features
    except Exception as e:
        st.error(f"❌ Error loading ML data: {e}")
        return None, None, None, None

@st.cache_data(ttl=300)
def load_walk_forward():
    try:
        return pd.read_csv(ML_RES / "walk_metrics_v6.csv")
    except:
        return pd.DataFrame()

@st.cache_resource(ttl=60)  # Reduced cache time
def load_dl_model(coin_name, _file_hash=None):
    try:
        seq_data = np.load(DL_SEQ_DIR / f"{coin_name}_seq.npz")
        dl_model = tf.keras.models.load_model(
                 DL_MODEL_DIR / f"{coin}_lstm_mt.h5")
        return seq_data, dl_model
    except Exception as e:
        st.error(f"❌ Error loading DL model: {e}")
        return None, None

def create_signal_card(probability, threshold, signal_type="ML"):
    if probability > threshold:
        st.markdown(f"""
        <div class="success-signal">
            <h3>✅ {signal_type} SIGNAL: GO LONG</h3>
            <p>Probability: {probability:.1%} (Threshold: {threshold:.1%})</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-signal">
            <h3>🔒 {signal_type} SIGNAL: STAY FLAT</h3>
            <p>Probability: {probability:.1%} (Threshold: {threshold:.1%})</p>
        </div>
        """, unsafe_allow_html=True)

def create_performance_chart(equity_df, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df['Date'],
        y=equity_df['Equity'],
        mode='lines',
        name='Strategy',
        line=dict(color='#667eea', width=3),
        hovertemplate='<b>Strategy</b><br>Date: %{x}<br>Return: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=equity_df['Date'],
        y=equity_df['Buy&Hold'],
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#fc4a1a', width=2, dash='dash'),
        hovertemplate='<b>Buy & Hold</b><br>Date: %{x}<br>Return: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )

    return fig

def create_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Flat', 'Predicted Long'],
        y=['Actual Flat', 'Actual Long'],
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hovertemplate='<b>%{y}</b><br>%{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix (2024 Test)", x=0.5),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_shap_chart(shap_values, feature_names, feature_values, top_n=10):
    indices = np.argsort(np.abs(shap_values))[-top_n:]

    fig = go.Figure()

    colors = ['#fc4a1a' if x < 0 else '#11998e' for x in shap_values[indices]]

    fig.add_trace(go.Bar(
        x=shap_values[indices],
        y=[f"{feature_names[i]}<br>({feature_values.iloc[i]:.3f})" for i in indices],
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="Top Feature Contributions (SHAP)", x=0.5),
        xaxis_title="SHAP Value",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )

    return fig

def format_date_safely(date_obj):
    """Safely format date object to string"""
    try:
        if pd.isna(date_obj):
            return "N/A"
        if isinstance(date_obj, pd.Timestamp):
            return date_obj.strftime("%Y-%m-%d")
        elif isinstance(date_obj, str):
            return pd.to_datetime(date_obj).strftime("%Y-%m-%d")
        else:
            return str(date_obj)
    except:
        return "Invalid Date"

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 Crypto-Momentum Dashboard</h1>
    <p>Advanced ML & Deep Learning Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# Load data with file change detection
data_loaded = False
file_hash = get_file_hash(ML_DATA / f"{coin}_full.parquet")

if panel == "🤖 ML Panel":
    ml_data = load_ml_data(coin, _file_hash=file_hash)
    if ml_data[0] is not None:
        df, model, shap_vals, feat_names = ml_data
        data_loaded = True
else:
    dl_data = load_dl_model(coin, _file_hash=file_hash)
    if dl_data[0] is not None:
        seq_data, dl_model = dl_data
        df_raw = pd.read_parquet(ML_DATA / f"{coin}_full.parquet")

        # ✅ Ensure proper date handling for DL panel
        df = df_raw.copy()
        if 'Date' in df.columns:
            if df['Date'].dtype == 'object':
                df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        data_loaded = True

if not data_loaded:
    st.error("❌ Unable to load required data files. Please check your data directory.")
    st.stop()

# Latest data display with safe date formatting
latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)

with col1:
    # ✅ Safe date formatting
    formatted_date = format_date_safely(latest['Date'])
    st.metric("📅 Latest Date", formatted_date)

with col2:
    st.metric("💰 Close Price", f"${latest['Close']:.2f}")

with col3:
    try:
        if len(df) > 1:
            price_change = ((latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']) * 100
            st.metric("📈 24h Change", f"{price_change:+.2f}%")
        else:
            st.metric("📈 24h Change", "N/A")
    except:
        st.metric("📈 24h Change", "N/A")

with col4:
    volume = latest.get('Volume', 0)
    if pd.isna(volume):
        volume = 0
    st.metric("📊 Volume", f"{volume:,.0f}")

st.markdown("---")

# ML Panel
if panel == "🤖 ML Panel":
    st.markdown("## 🔮 Machine Learning Analysis")
    # ➤ contextual info for ML users
    with st.expander("🛈 How to interpret Strategy Returns", expanded=False):
        st.markdown(INFO_TEXT)

    # Signal display
    prob = latest.get("prob", 0.0)
    if pd.isna(prob):
        prob = 0.0
    create_signal_card(prob, threshold, "ML")

    # Metrics and backtest
    if show_metrics or show_backtest:
        col1, col2 = st.columns(2)

        with col1:
            if show_metrics:
                st.markdown("### 📊 Performance Metrics")

                # 2024 test metrics
                test_mask = df["Date"] >= ML_TEST_DATE
                if test_mask.sum() > 0:
                    y_true = df.loc[test_mask, "bin_h3_thr2"].astype(int)
                    y_prob = df.loc[test_mask, "prob"].fillna(0)
                    y_pred = (y_prob > threshold).astype(int)

                    auc_score = roc_auc_score(y_true, y_prob)
                    accuracy = accuracy_score(y_true, y_pred)

                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("🎯 AUC Score", f"{auc_score:.3f}")
                    metric_col2.metric("✅ Accuracy", f"{accuracy:.3f}")

                    # Confusion matrix
                    st.plotly_chart(create_confusion_matrix(y_true, y_pred), use_container_width=True)
                else:
                    st.warning("No test data available for 2024+")

        with col2:
            if show_backtest:
                st.markdown("### 💹 Strategy Performance")
                try:
                    backtest_df = df[["Date", "Close", "prob"]].copy().dropna()
                    if len(backtest_df) > 0:
                        equity = backtest(backtest_df, threshold)

                        final_return = equity["Equity"].iloc[-1]
                        bh_return = equity["Buy&Hold"].iloc[-1]

                        perf_col1, perf_col2 = st.columns(2)
                        perf_col1.metric("🎯 Strategy Return", f"{final_return:.2f}x")
                        perf_col2.metric("📈 Buy & Hold", f"{bh_return:.2f}x")

                        st.plotly_chart(create_performance_chart(equity, f"{coin.upper()} ML Strategy"), use_container_width=True)
                    else:
                        st.warning("No valid data for backtesting")
                except Exception as e:
                    st.error(f"Backtest error: {e}")

    # SHAP analysis
    with st.expander("🔍 Feature Importance Analysis"):
        try:
            fig_shap = create_shap_chart(shap_vals[-1], feat_names, latest[feat_names])
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP analysis error: {e}")

# ─────────────────────────────────────────────────── DL PANEL ───────────────────────────────────────────────────
else:
    st.markdown("## 🧠 Deep Learning Analysis")
    # ➤ contextual info for DL users
    with st.expander("🛈 How to interpret Strategy Returns", expanded=False):
        st.markdown(INFO_TEXT)

    try:
        # 1) Read sequences
        seq = seq_data["X"]
        all_preds = dl_model.predict(seq, verbose=0)

        # 2) Get 3-day (h3) long-prob series
        if isinstance(all_preds, list):
            dl_prob_series = all_preds[1].ravel()    # second output = h3
        else:
            dl_prob_series = all_preds.ravel()

        # 3) Show today's signal
        main_prob = float(dl_prob_series[-1])
        create_signal_card(main_prob, threshold, "DL")

        # 4) Multi-horizon predictions (optional)
        st.markdown("### 🔮 Multi-Horizon Predictions")
        # Add multi-horizon display here if needed

        # 5) Calculate correct start index for backtest
        seq_len = seq.shape[1]

        # 6) Initialize df_dl.prob with NaN
        df_dl = df.copy()
        df_dl["prob"] = np.nan

        # 7) Map DL predictions to correct dates
        start_idx = seq_len
        end_idx = start_idx + len(dl_prob_series)
        if end_idx <= len(df_dl):
            df_dl.iloc[start_idx:end_idx, df_dl.columns.get_loc("prob")] = dl_prob_series

        # 8) Backtest with non-NaN portion
        valid_data = df_dl.dropna(subset=["prob"])
        if len(valid_data) > 0:
            equity_dl = backtest(
                valid_data[["Date","Close","prob"]].copy(),
                threshold
            )

            # 9) Performance metrics
            final_return = equity_dl["Equity"].iloc[-1]
            bh_return = equity_dl["Buy&Hold"].iloc[-1]

            c1, c2 = st.columns(2)
            c1.metric("🎯 DL Strategy Return", f"{final_return:.2f}×")
            c2.metric("📈 Buy & Hold", f"{bh_return:.2f}×")

            # 10) Chart
            st.plotly_chart(
                create_performance_chart(equity_dl, f"{coin.upper()} DL Strategy"),
                use_container_width=True
            )
        else:
            st.warning("No valid DL predictions available for backtesting")

    except Exception as e:
        st.error(f"❌ DL prediction error: {e}")
        st.info("Check DL model / sequence length configuration.")

# Walk-forward analysis
walk_df = load_walk_forward()
if not walk_df.empty:
    with st.expander("📈 Walk-Forward Analysis"):
        try:
            coin_walk = walk_df[walk_df['coin'] == coin]
            if not coin_walk.empty:
                auc_pivot = coin_walk.pivot_table(index='target', columns='step', values='auc')

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=auc_pivot.values,
                    x=auc_pivot.columns,
                    y=auc_pivot.index,
                    colorscale='Blues',
                    showscale=True,
                    hovertemplate='Step: %{x}<br>Target: %{y}<br>AUC: %{z:.3f}<extra></extra>'
                ))

                fig_heatmap.update_layout(
                    title=dict(text="Walk-Forward AUC Scores", x=0.5),
                    xaxis_title="Validation Step",
                    yaxis_title="Target",
                    height=400
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display walk-forward analysis: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 <strong>Crypto-Momentum Dashboard</strong> | Built with Streamlit, LightGBM & TensorFlow</p>
    <p>📊 Advanced ML/DL Analytics for Cryptocurrency Trading</p>
</div>
""", unsafe_allow_html=True)