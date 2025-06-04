# Fintech Crypto Dashboard & Chatbot Project

## 🌟 Project Overview
This project integrates cryptocurrency price data (Bitcoin and Ethereum) with sentiment analysis of news articles to build an end-to-end solution that includes:
1. **Data Collection & Preprocessing**  
2. **Exploratory Data Analysis (EDA)**  
3. **Machine Learning (ML) & Deep Learning (DL) Modeling**  
4. **Interactive Streamlit Dashboard**  
5. **Large Language Model (LLM) Integration & Chatbot**

The goal is to provide users with accurate price predictions, directional forecasts, sentiment-driven insights, and a conversational interface that answers questions based on both historical data and live analysis.

---

## 📊 Data Sets
1. **Price Data (yfinance)**  
   - Daily OHLCV data for BTC-USD and ETH-USD from January 1, 2018 to December 31, 2025.  
   - Retrieved programmatically using `yfinance` in `src/data_processing.py`.  

2. **News Data (CryptoNewsDataset)**  
   - Contains ~216,000 crypto-related news items spanning September 2017 to May 2025, covering 660+ cryptocurrencies.  
   - Each record includes `newsDatetime`, `title`, `text`, `currency` (e.g., “BTC” or “ETH”), and user-based sentiment labels (positive/negative/important).  
   - Place the raw CSV as `data/raw/crypto_news.csv`.  

3. **Processed Daily Sentiment Summaries**  
   - Filter news to include only BTC and ETH items.  
   - Use VADER sentiment analysis to compute a compound sentiment score for each article.  
   - Aggregate at daily granularity to produce `avg_sentiment` (mean compound score) and `news_count` (number of articles).  

4. **Merged Price & Sentiment Data**  
   - Join daily price data with daily sentiment summaries (left join by date).  
   - Fill missing sentiment values forward (ffill) and assign `news_count = 0` for days without news.  
   - Resulting files: `merged_btc.csv` and `merged_eth.csv`.  

5. **Feature-Engineered ML/DL Datasets**  
   - Compute technical indicators:  
     - MA7, MA30 (7-day and 30-day moving averages)  
     - RSI14 (14-day Relative Strength Index)  
     - MACD_diff (difference between MACD and signal line)  
     - return_1d (daily percentage return)  
   - For ML models: create `target_price` (next day’s closing price) and `target_dir` (binary direction: price_up = 1 if next day close > today).  
   - Save as `*_ml_dataset.csv`.  
   - For DL (LSTM/GRU) models: use a sliding window of length 14 to form X sequences and y targets, scale all features to [0,1] with MinMaxScaler, and save `*_X.npy`, `*_y.npy`, and `*_scaler.pkl`.  

---

## 🎯 Project Objectives
- **Data Integration & Preprocessing**  
  - Create a clean, joined time series of cryptocurrency prices and daily sentiment metrics.  
  - Ensure no missing dates in crypto prices; fill or carry forward missing sentiment when necessary.  

- **Exploratory Data Analysis (EDA)**  
  - Visualize correlations between price and sentiment over time.  
  - Identify periods with unusually high news volume and noticeable price movements.  

- **Machine Learning Models**  
  - Train **Random Forest Regressors** to predict next-day closing price.  
  - Train **Random Forest Classifiers** to predict next-day price direction (up/down).  
  - Evaluate using RMSE, MAE for regression and Accuracy, Precision/Recall/F1 for classification.  

- **Deep Learning Models**  
  - Build and train **LSTM** and **GRU** architectures for time series regression.  
  - Include two stacked recurrent layers and dropout to mitigate overfitting.  
  - Compare test RMSE with classical ML models to identify the best-performing approach.  

- **Interactive Dashboard**  
  - Display daily closing price, average sentiment, and news count charts for both BTC and ETH.  
  - Provide a “Next-Day Prediction” panel that shows model output (price or direction) depending on user-selected model.  
  - Summarize test performance metrics in a table for easy comparison.  

- **LLM-Enhanced Chatbot**  
  - Embed news articles into a vector database (FAISS or ChromaDB).  
  - On user queries, retrieve the top-k (e.g., k=5) most relevant news embeddings.  
  - Generate context-aware answers using either Azure OpenAI (GPT-3.5 / GPT-4) or an open-source model (e.g., Llama 2, Falcon).  
  - If the user asks “What will happen tomorrow?”, include the ML/DL model’s prediction in the prompt to the LLM, enabling a coherent, data-driven response.  

---

## ⚙️ Installation & Setup
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Taylanozveren/Ai_Project_Chatbot_Fintech.git
   cd Ai_Project_Chatbot_Fintech
