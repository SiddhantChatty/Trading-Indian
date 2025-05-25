import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv

load_dotenv()

VALID_EMAIL = os.getenv("VALID_EMAIL")
VALID_PASSCODE = os.getenv("VALID_PASSCODE")

def fetch_nse_data(symbol: str):
    df = yf.download(f"{symbol}.NS", period="10y")
    df.reset_index(inplace=True)
    df.rename(columns={"Adj Close": "Close"}, inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def generate_features(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def train_model(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    features = ['SMA_20', 'RSI', 'MACD', 'Return']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df['Target'])
    return model

def generate_signal(df, model):
    latest = df.iloc[-1:]
    features = ['SMA_20', 'RSI', 'MACD', 'Return']
    prediction = model.predict(latest[features])[0]
    return 'BUY' if prediction == 1 else 'SELL'

def forecast_target_and_stoploss(df):
    df = df.copy()
    df['Days'] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[['Days']], df['Close'])
    future_days = 60
    future_price = model.predict([[len(df) + future_days]])[0]
    current_price = df.iloc[-1]['Close']
    stop_loss = current_price * 0.95
    target = future_price
    return round(stop_loss, 2), round(target, 2)

def scan_top_stocks(stock_list):
    suggestions = []
    for sym in stock_list:
        try:
            df = fetch_nse_data(sym)
            df = generate_features(df)
            model = train_model(df)
            signal = generate_signal(df, model)
            if signal == 'BUY':
                stop_loss, target = forecast_target_and_stoploss(df)
                suggestions.append({
                    'Stock': sym,
                    'Current Price': round(df.iloc[-1]['Close'], 2),
                    'Stop Loss': stop_loss,
                    'Target (1-3M)': target
                })
        except:
            continue
    return pd.DataFrame(suggestions)

def display_custom_section(title, stock_list):
    st.subheader(title)
    result_df = scan_top_stocks(stock_list)
    if not result_df.empty:
        st.dataframe(result_df)
    else:
        st.info(f"No strong BUY signals in the {title} list.")

def main_app():
    st.title("📈 Indian AI Trading Bot – NSE 10-Year Analyzer")
    symbol = st.sidebar.text_input("📌 Enter NSE Symbol (e.g., RELIANCE):", "RELIANCE")
    capital = st.sidebar.number_input("💰 Enter Your Capital (₹):", min_value=1000, value=10000)
    run_btn = st.sidebar.button("▶️ Run Analysis")
    suggest_btn = st.sidebar.button("🔥 Suggest Top Stocks")
    low_perf_btn = st.sidebar.button("📉 Scan Low-Performing Picks")
    ipo_btn = st.sidebar.button("🆕 New IPO Watchlist")

    if run_btn:
        try:
            df = fetch_nse_data(symbol)
            df = generate_features(df)
            model = train_model(df)
            signal = generate_signal(df, model)
            stop_loss, target = forecast_target_and_stoploss(df)
            current_price = df.iloc[-1]['Close']
            quantity = int(capital // current_price)

            st.header(f"🔍 Analysis for: {symbol}")
            st.markdown(f"**Signal:** `{signal}`")
            st.markdown(f"**Current Price:** ₹{round(current_price, 2)}")
            st.markdown(f"**Target Price (3 months):** ₹{target}")
            st.markdown(f"**Stop-Loss:** ₹{stop_loss}")

            if signal == 'BUY':
                st.success(f"✅ Suggestion: Buy **{quantity} shares** with ₹{capital}")
            else:
                st.warning("⚠️ Suggestion: Hold or Sell")

            st.subheader("📊 Historical Close Price (10 Years)")
            st.line_chart(df.set_index("Date")["Close"])

            st.subheader("📈 Indicators Over Time")
            st.line_chart(df.set_index("Date")[['SMA_20', 'RSI', 'MACD']].dropna())

            st.subheader("🗂️ Last 10 Days of Data")
            st.dataframe(df.tail(10))

        except Exception as e:
            st.error(f"Error: {str(e)}")

    if suggest_btn:
        top_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN", "LT", "ITC", "HINDUNILVR"]
        display_custom_section("🔥 Weekly Top Stock Suggestions", top_stocks)

    if low_perf_btn:
        underdogs = ["WAAREE", "IDEA", "YESBANK", "JPPOWER", "SUZLON", "IRCTC", "ZOMATO"]
        display_custom_section("📉 Low-Performing But Potential Gainers", underdogs)

    if ipo_btn:
        ipo_list = ["TATAELXSI", "MOBIKWIK", "NYKAA", "PAYTM", "DELHIVERY"]
        display_custom_section("🆕 Trending IPOs", ipo_list)

def login():
    st.title("🔐 Login to Access AI Bot")
    email = st.text_input("📧 Gmail ID:")
    password = st.text_input("🔑 Passcode:", type="password")
    if st.button("Login"):
        if email == VALID_EMAIL and password == VALID_PASSCODE:
            st.success("Login successful ✅")
            st.session_state['logged_in'] = True
        else:
            st.error("❌ Invalid login credentials")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_app()
else:
    login()

