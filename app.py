import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import os

from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file

# Load login credentials from environment variables (secrets)
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
    signal = 'BUY' if prediction == 1 else 'SELL'
    return signal

def forecast_target_and_stoploss(df):
    df = df.copy()
    df['Days'] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[['Days']], df['Close'])
    future_days = 60  # ~3 months
    future_price = model.predict([[len(df) + future_days]])[0]
    current_price = df.iloc[-1]['Close']
    stop_loss = current_price * 0.95  # 5% below current
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

def main_app():
    st.title("📈 Indian Stock Market AI Trading Bot - 10 Year Analysis")

    symbol = st.sidebar.text_input("Enter NSE Stock Symbol (e.g., RELIANCE):", "RELIANCE")
    capital = st.sidebar.number_input("Enter Your Capital (₹):", min_value=1000, value=10000)
    run_bot = st.sidebar.button("Run Analysis")
    suggest_btn = st.sidebar.button("Suggest Top Stocks")

    if run_bot:
        try:
            df = fetch_nse_data(symbol)
            df = generate_features(df)

            model = train_model(df)
            signal = generate_signal(df, model)
            stop_loss, target = forecast_target_and_stoploss(df)

            current_price = df.iloc[-1]['Close']
            quantity = int(capital // current_price)

            st.header(f"Signal for {symbol}: {signal}")
            st.markdown(f"**Suggested Stop-Loss:** ₹{stop_loss}")
            st.markdown(f"**Target in 1-3 Months:** ₹{target}")
            st.markdown(f"**Current Price:** ₹{round(current_price, 2)}")
            if signal == 'BUY':
                st.success(f"You can buy **{quantity} shares** with ₹{capital}")
            else:
                st.warning("Consider selling if you already hold this stock.")

            st.subheader("📊 Last 10 Days of Data")
            st.dataframe(df.tail(10))

            st.subheader("📉 10-Year Historical Close Prices")
            st.line_chart(df.set_index("Date")["Close"])

            st.subheader("🧠 Model Features Over Time")
            st.line_chart(df.set_index("Date")[['SMA_20', 'RSI', 'MACD']].dropna())

        except Exception as e:
            st.error(f"Error: {str(e)}")

    if suggest_btn:
        top_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN", "LT", "ITC", "HINDUNILVR"]
        st.subheader("📌 Weekly BUY Suggestions")
        suggestions_df = scan_top_stocks(top_stocks)
        if not suggestions_df.empty:
            st.dataframe(suggestions_df)
        else:
            st.info("No strong BUY signals detected in the top stocks right now.")

def login():
    st.title("🔐 Login to Indian Stock Market AI Bot")
    email = st.text_input("Enter your Gmail ID:")
    password = st.text_input("Enter your Passcode:", type="password")
    login_button = st.button("Login")

    if login_button:
        if email == VALID_EMAIL and password == VALID_PASSCODE:
            st.success("Login successful!")
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid email or passcode.")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_app()
else:
    login()
