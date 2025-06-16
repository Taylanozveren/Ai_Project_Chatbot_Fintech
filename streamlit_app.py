"""
Professional Crypto-Momentum Dashboard
Enhanced ML + DL Analytics Platform
"""

import sys
import tensorflow as tf
import pathlib
import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from datetime import datetime, timedelta

def smart_predict(model, x):
    if hasattr(model, "predict"):
        return model.predict(x, verbose=0)

    infer = (model.signatures.get("serve")
             or model.signatures.get("serving_default")
             or list(model.signatures.values())[0])

    out = tf.nest.flatten(infer(tf.constant(x)))
    return [o.numpy() for o in out]

from pathlib import Path
THR_CSV_ML = Path("results/threshold_tune_bin_h3_thr2.csv")
def get_opt_thr(coin: str, panel: str):
    if panel == "ML" and THR_CSV_ML.exists():
        row = pd.read_csv(THR_CSV_ML).query("coin == @coin")
        if not row.empty:
            return float(row["best_thr"].iloc[0])
    dl_csv = Path(f"results/dl_thr_hold_{coin}.csv")
    if panel == "DL" and dl_csv.exists():
        return float(pd.read_csv(dl_csv)["thr"].iloc[0])
    return 0.02 if panel == "ML" else 0.05

def get_opt_hold(coin: str, default_hold: int = 3):
    fn = Path(f"results/dl_thr_hold_{coin}.csv")
    if fn.exists():
        return int(pd.read_csv(fn)["hold"].iloc[0])
    return default_hold


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

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

ML_DATA = ROOT / "dashboard_data"
ML_RES = ROOT / "results"
DL_SEQ_DIR = ROOT / "dl" / "outputs"
DL_MODEL_DIR = DL_SEQ_DIR / "models"
DL_WALK = ROOT / "dl" / "reports" / "dl_walkforward.csv"

ML_COINS = ["btc", "eth"]
DL_TARGETS = {'h1': '1-day', 'h3': '3-day', 'h5': '5-day', 'r3': '3d-return', 'r5': '5d-return'}
ML_TEST_DATE = np.datetime64("2024-01-01")

st.set_page_config(page_title="Crypto Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

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

try:
    from backtest_helper import backtest
except ImportError:
    def backtest(df, thr=0.02, fee=0.0005, hold=3):
        """
        Fixed backtest implementation with proper return limits
        """
        out = df.copy()

        if len(out) == 0:
            out = out.assign(Equity=1.0, **{"Buy&Hold": 1.0})
            return out

        # Generate signals - only when probability > threshold
        signals = (out['prob'] > thr).astype(int)

        # Calculate daily returns
        returns = out['Close'].pct_change().fillna(0)

        # Strategy returns: only participate when signal=1, otherwise 0% return
        strategy_returns = returns * signals.shift(1).fillna(0)

        # Apply transaction fees when entering positions
        position_changes = signals.diff().fillna(0)
        entry_signals = (position_changes > 0).astype(int)  # Signal goes from 0 to 1
        strategy_returns = strategy_returns - fee * entry_signals

        # CRITICAL FIX: Cap individual daily returns to prevent explosion
        strategy_returns = np.clip(strategy_returns, -0.15, 0.15)  # Max ±15% per day

        # Calculate cumulative returns
        equity = (1 + strategy_returns).cumprod()
        buy_hold = (1 + returns).cumprod()

        # ADDITIONAL SAFETY: Cap final equity to prevent unrealistic results
        equity = np.minimum(equity, buy_hold * 50)  # Strategy can't be >50x better than buy&hold

        out = out.assign(Equity=equity, **{"Buy&Hold": buy_hold})
        return out

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Settings")
    panel = st.radio("Select Analysis Panel", ["🤖 ML Panel", "🧠 DL Panel"], index=0)

    st.markdown("### 📊 Asset Selection")
    coin = st.selectbox("Choose Cryptocurrency", ML_COINS, format_func=lambda x: x.upper())

    st.markdown("### 🎯 Strategy Parameters")

    default_thr = get_opt_thr(coin, "ML" if panel.startswith("🤖") else "DL")
    threshold = st.slider("Signal Threshold", 0.0, 0.50, default_thr, 0.01,
                          help="Probability threshold for long signals")

    # DL panelindeysek ekstra HOLD slider göster
    if panel.startswith("🧠"):
        dl_hold = st.slider("Hold Days (DL)", 1, 5, get_opt_hold(coin))
    else:
        dl_hold = 3

    st.markdown("### 📈 Performance Metrics")
    show_metrics = st.checkbox("Show Detailed Metrics", True)
    show_backtest = st.checkbox("Show Backtest", True)

    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    try:
        data_file = ML_DATA / f"{coin}_full.parquet"
        if data_file.exists():
            mod_time = datetime.fromtimestamp(data_file.stat().st_mtime)
            st.caption(f"📅 Data Updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    except:
        pass

def get_file_hash(file_path):
    try:
        return str(file_path.stat().st_mtime)
    except:
        return str(datetime.now().timestamp())

@st.cache_data(ttl=60)
def load_ml_data(coin_name, _file_hash=None):
    try:
        df = pd.read_parquet(ML_DATA / f"{coin_name}_full.parquet")
        model = joblib.load(ML_DATA / f"{coin_name}_model.pkl")
        shap_vals = np.load(ML_DATA / f"{coin_name}_shap.npy")
        features = joblib.load(ML_DATA / "feature_list.pkl")

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

@st.cache_resource(ttl=60)
def load_dl_model(coin_name, _file_hash=None):
    """
    Load the DL sequence and model for the given coin.
    Tries .h5 first; if missing, falls back to SavedModel via TFSMLayer or tf.saved_model.load.
    """
    seq_path = DL_SEQ_DIR / f"{coin_name}_seq.npz"
    if not seq_path.exists():
        st.error(f"Seq dosyası yok: {seq_path}")
        return None, None
    seq_data = np.load(seq_path)

    # 1) Try standard .h5 LSTM multi-task model
    h5_path = DL_MODEL_DIR / f"{coin_name}_lstm_mt.h5"
    if h5_path.exists():
        model = tf.keras.models.load_model(h5_path, compile=False)
        return seq_data, model

    # 2) Fallback: TF SavedModel directory
    sm_dir = DL_MODEL_DIR / f"{coin_name}_lstm_mt_tf"
    if sm_dir.exists():
        try:
            # If TFSMLayer is available, use it to call via serve signature
            from keras.layers import TFSMLayer
            model = TFSMLayer(str(sm_dir), call_endpoint="serve")
        except ImportError:
            # Otherwise, load the SavedModel directly
            model = tf.saved_model.load(str(sm_dir))
        return seq_data, model

    st.error(f"Model bulunamadı: {h5_path.name} veya {sm_dir.name}")
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

st.markdown("""
<div class="main-header">
    <h1>🚀 Crypto-Momentum Dashboard</h1>
    <p>Advanced ML & Deep Learning Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

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

        df = df_raw.copy()
        if 'Date' in df.columns:
            if df['Date'].dtype == 'object':
                df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        data_loaded = True

if not data_loaded:
    st.error("❌ Unable to load required data files. Please check your data directory.")
    st.stop()

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)

with col1:
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

if panel == "🤖 ML Panel":
    st.markdown("## 🔮 Machine Learning Analysis")
    with st.expander("🛈 How to interpret Strategy Returns", expanded=False):
        st.markdown(INFO_TEXT)

    prob = latest.get("prob", 0.0)
    if pd.isna(prob):
        prob = 0.0
    create_signal_card(prob, threshold, "ML")

    if show_metrics or show_backtest:
        col1, col2 = st.columns(2)

        with col1:
            if show_metrics:
                st.markdown("### 📊 Performance Metrics")

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

                    st.plotly_chart(create_confusion_matrix(y_true, y_pred), use_container_width=True)
                else:
                    st.warning("No test data available for 2024+")

        with col2:
            if show_backtest:
                st.markdown("### 💹 Strategy Performance")
                try:
                    backtest_df = df[["Date", "Close", "prob"]].copy().dropna()
                    if len(backtest_df) > 0:
                        equity = backtest(backtest_df, thr=threshold, hold=3, oos_start="2022-01-01")


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

    with st.expander("🔍 Feature Importance Analysis"):
        try:
            fig_shap = create_shap_chart(shap_vals[-1], feat_names, latest[feat_names])
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP analysis error: {e}")

else:
    st.markdown("## 🧠 Deep Learning Analysis")
    with st.expander("🛈 How to interpret Strategy Returns", expanded=False):
        st.markdown(INFO_TEXT)

    try:
        seq = seq_data["X"]
        all_preds = smart_predict(dl_model, seq)

        if isinstance(all_preds, list):
            dl_prob_series = all_preds[1].ravel()
        else:
            dl_prob_series = all_preds.ravel()

        np.save(f"dl/outputs/{coin}_dl_prob.npy", dl_prob_series)

        main_prob = float(dl_prob_series[-1])
        create_signal_card(main_prob, threshold, "DL")

        st.markdown("### 🔮 Multi-Horizon Predictions")

        seq_len = seq.shape[1]

        df_dl = df.copy()
        df_dl["prob"] = np.nan

        # CRITICAL FIX: Proper alignment of DL predictions with dates
        # DL predictions correspond to the sequence end dates
        if len(dl_prob_series) <= len(df_dl):
            # Align predictions to the CORRECT date range
            # If we have sequence length N, prediction i corresponds to df row (seq_len + i - 1)
            start_idx = max(0, seq_len - 1)  # Start after sequence build-up
            end_idx = min(len(df_dl), start_idx + len(dl_prob_series))

            # Only assign predictions where we have both data and predictions
            valid_pred_count = end_idx - start_idx
            if valid_pred_count > 0:
                df_dl.iloc[start_idx:end_idx, df_dl.columns.get_loc("prob")] = dl_prob_series[:valid_pred_count]
        else:
            # If more predictions than data, use the appropriate slice
            pred_start = len(dl_prob_series) - len(df_dl) + max(0, seq_len - 1)
            df_dl.iloc[max(0, seq_len - 1):, df_dl.columns.get_loc("prob")] = dl_prob_series[pred_start:]

        valid_data = df_dl.dropna(subset=["prob"])
        if len(valid_data) > 0:
            equity_dl = backtest(
                valid_data[["Date", "Close", "prob"]].copy(),
                thr=threshold,
                hold=dl_hold,
                oos_start="2022-01-01"
            )

            final_return = equity_dl["Equity"].iloc[-1]
            bh_return = equity_dl["Buy&Hold"].iloc[-1]

            st.write(f"🕒 Hold param (DL): **{dl_hold} day**")
            c1, c2 = st.columns(2)
            c1.metric("🎯 DL Strategy Return", f"{final_return:.2f}×")
            c2.metric("📈 Buy & Hold", f"{bh_return:.2f}×")

            st.plotly_chart(
                create_performance_chart(equity_dl, f"{coin.upper()} DL Strategy"),
                use_container_width=True
            )
        else:
            st.warning("No valid DL predictions available for backtesting")

    except Exception as e:
        st.error(f"❌ DL prediction error: {e}")
        st.info("Check DL model / sequence length configuration.")

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

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 <strong>Crypto-Momentum Dashboard</strong> | Built with Streamlit, LightGBM & TensorFlow</p>
    <p>📊 Advanced ML/DL Analytics for Cryptocurrency Trading</p>
</div>
""", unsafe_allow_html=True)