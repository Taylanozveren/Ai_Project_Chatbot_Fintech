# 🚀 Crypto-Momentum Dashboard & Chatbot (v1.0 Final)

## 📌 Project Overview

The Crypto-Momentum Dashboard & Chatbot project is an integrated financial AI platform designed for cryptocurrency price forecasting, primarily focusing on Bitcoin (BTC) and Ethereum (ETH). It leverages historical price data and market sentiment from news and social media sources to provide real-time, actionable insights through Machine Learning (ML) and Deep Learning (DL) techniques.

### Important Notices

- ✅ **Price Data**: Continuously updated via automated pipelines (yfinance)
- ⚠️ **Sentiment Data**: Static data (last update: May 25, 2025), limited due to external sources
- ⚠️ **NLP & LLM Chatbot Modules**: Deferred to future phases

## 🎯 Objectives & Capabilities

### ✅ Completed Features

**Data Engineering & Preprocessing:**
- Daily OHLCV data (2018–2025) from Yahoo Finance
- Aggregated daily sentiment scores (limited to May 2025)
- Feature engineering pipeline (lag features, technical indicators)

**Exploratory Data Analysis (EDA):**
- Visualization of price-sentiment relationships
- Identification of high-impact news periods

**Machine Learning (ML):**
- LightGBM Classifier: Price direction (binary classification for +2% moves over 1, 3, and 5 days)
- Walk-forward validation (realistic backtests)

**Deep Learning (DL):**
- Single-task & Multi-task LSTM: Capturing short/medium-term trends
- 1D-CNN Architecture: Experimental study (not deployed due to lower performance)

**Explainability:**
- SHAP analysis for feature importance (summary/waterfall plots)

**Strategy & Backtesting:**
- Capital allocation simulations (theoretical returns vs. buy-and-hold)

**Interactive Dashboard:**
- Real-time ML & DL predictions via Streamlit
- Visual analytics (heatmaps, confusion matrices, equity curves)

### 🛠️ Future Enhancements

- Real-time NLP sentiment analysis
- LLM-powered conversational chatbot (FastAPI, Azure OpenAI / Hugging Face)

## 📊 Technical Details & Architecture

| Layer | Technologies & Libraries |
|-------|-------------------------|
| Data Engineering | Python, Pandas, NumPy, yfinance, Parquet |
| ML Modeling | scikit-learn, LightGBM, joblib |
| DL Modeling | TensorFlow, Keras |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Chatbot (Planned) | FastAPI, Azure OpenAI / Hugging Face |
| Storage | Parquet, CSV |
| Version Control | Git, GitHub |

## 📁 Folder Structure

```
Project/
├── data/
│   ├── raw/                  # Raw BTC/ETH/news data files
│   ├── processed/            # Feature-engineered data
│   └── dashboard_data/       # Streamlit-ready data
│
├── src/                      # Data pipeline & ML scripts
│   ├── 01_data_processing.py
│   ├── 02_merge_data.py
│   ├── 03_data_validation.py
│   ├── 04_clean_merged.py
│   ├── 05_feature_engineering.py
│   ├── 06_train_walkforward.py
│   ├── 07_detailed_report.py
│   ├── 08_threshold_tune.py
│   ├── 09_prepare_dashboard_data.py
│   ├── 11_shap_values.py
│   └── backtest_helper.py
│
├── dl/                       # Deep Learning workflows
│   ├── seq_window.py         # Sliding window data prep
│   ├── train.py              # Training LSTM & CNN
│   ├── lstm.py               # Single-task LSTM
│   ├── lstm_multitask.py     # Multi-task LSTM
│   ├── cnn.py                # CNN (experimental)
│   ├── evaluate.py           # Evaluation scripts
│   ├── outputs/              # Prepared datasets (.npz)
│   └── scalers/              # Scaler objects (.pkl)
│
├── models/                   # ML (.pkl) & DL (.h5) models
├── results/                  # Reports & visualizations
├── notebooks/                # Jupyter notebooks (EDA)
├── streamlit_app.py          # Streamlit dashboard
└── requirements.txt          # Dependencies
```

## 🚀 Installation & Usage

### Step 1: Clone Repository

```bash
git clone https://github.com/Taylanozveren/Ai_Project_Chatbot_Fintech.git
cd Ai_Project_Chatbot_Fintech
```

### Step 2: Virtual Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run Data Pipeline & ML Scripts

```bash
python src/01_data_processing.py
python src/02_merge_data.py
python src/03_data_validation.py
python src/04_clean_merged.py
python src/05_feature_engineering.py
python src/06_train_walkforward.py
python src/07_detailed_report.py
python src/08_threshold_tune.py
python src/09_prepare_dashboard_data.py
python src/11_shap_values.py
```

### Step 4: Train & Evaluate DL Models

```bash
python dl/seq_window.py
python dl/train.py
python dl/evaluate.py
```

### Step 5: Launch Streamlit Dashboard

```bash
streamlit run streamlit_app1.py
```

## ⚠️ Current Limitations & Notices

- **NLP Module**: Static sentiment data up to May 25, 2025 (due to external source constraints)
- **Chatbot & LLM Integration**: Planned future enhancements (pending development)

## 📅 Roadmap & Next Steps

| Timeline | Milestone | Status |
|----------|-----------|--------|
| June 2025 | ML & DL Dashboard Finalized ✅ | Completed |
| June 2025 | Deployment of Dashboard ✅ | Completed |
| July 2025+ | Real-time NLP Integration (Currently Just Invalid csv Data 🛠️ | Planned |
| July 2025+ | LLM-Powered Chatbot 🤖 | Planned |

## 🤝 Contribution & Collaboration

**Lead Developer & Architect**: Taylan Özveren  
📧 **Contact**: taylanozveren67@gmail.com

Contributions are highly encouraged:

1. Fork the repository
2. Create your branch:
   ```bash
   git checkout -b feature/MyNewFeature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add MyNewFeature"
   ```
4. Push to branch:
   ```bash
   git push origin feature/MyNewFeature
   ```
5. Open Pull Request on GitHub

## 📃 License

This project is licensed under the MIT License - see LICENSE.md for details.

