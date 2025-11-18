import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import math

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="StockSphere",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
h1 { color: #1a237e; font-weight: bold; }
h2 { color: #d84315; font-weight: bold; }
h3 { color: #00695c; font-weight: bold; }
.stButton>button { background-color: #ff6f61; color: yellow; font-weight: bold; }
.stMetricLabel { color: #283593; font-weight: bold; }
.css-1d391kg { background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ====== HERO SECTION ======
st.markdown("""
<div style="text-align:center; padding:20px; background-color:#F4C2C2; border-radius:10px;">
    <h1>🌐 Welcome to StockSphere</h1>
    <h3>AI-powered Stock Price Predictor</h3>
    <p>Get insights, predictions, forecasts, and simulate stock prices easily.</p>
</div>
""", unsafe_allow_html=True)

# ====== SIDEBAR ======
with st.sidebar:
    st.header("Settings")
    USE_DEMO_MODE = st.checkbox("Use Demo Data (if Yahoo Finance blocked)", value=False)
    stock = st.text_input("Enter Stock Symbol", "GOOG")
    st.markdown("---")

# ====== CURRENCY RATES ======
INR_TO_USD = 1/83   # convert INR to USD
USD_TO_INR = 83     # convert USD back to INR

# ====== DATA DOWNLOAD ======
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)
data_file = f"{stock}_data.csv"

if USE_DEMO_MODE:
    st.warning("⚠️ Using Demo Data Mode")
    dates = pd.date_range(start='2010-01-01', end=datetime.now(), freq='B')
    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.02
    prices = 100 * np.exp(np.cumsum(returns) + 0.0003 * np.arange(len(dates)))
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(len(dates)) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02),
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, len(dates)),
    }, index=dates)
    st.success(f"✓ Demo data loaded for {stock}!")
else:
    try:
        st.info("Downloading stock data...")
        data = yf.download(stock, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError("No data downloaded")
        data.to_csv(data_file)
        st.success("✓ Data downloaded and saved successfully!")
    except Exception as e:
        if os.path.exists(data_file):
            st.warning(f"⚠️ Download failed. Using saved data from: {data_file}")
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            st.error(f"❌ Error: {e}\n💡 Enable 'Use Demo Data' in sidebar.")
            st.stop()

# ====== DATA CLEANUP ======
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])

if len(data) < 200:
    st.error("Not enough data for predictions (≥200 required).")
    st.stop()

# ====== LOAD MODEL ======
@st.cache_resource
def get_model(path="Latest_stock_price_model.keras"):
    if os.path.exists(path):
        return load_model(path)
    return None

model = get_model()

# ====== MOVING AVERAGES ======
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

# ====== TABS LAYOUT ======
tabs = st.tabs(["Overview", "Analysis", "Predictions", "Future Forecast", "Simulator"])

# -------------------- Overview --------------------
with tabs[0]:
    st.header("🌟 Overview")
    current_price_inr = data['Close'].iloc[-1] * USD_TO_INR
    high_52week_inr = data['Close'].max() * USD_TO_INR
    low_52week_inr = data['Close'].min() * USD_TO_INR

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{current_price_inr:.2f}")
    col2.metric("52-Week High", f"₹{high_52week_inr:.2f}")
    col3.metric("52-Week Low", f"₹{low_52week_inr:.2f}")

    st.subheader("Raw Data Sample (₹)")
    st.dataframe((data * USD_TO_INR).tail(10), use_container_width=True)

# -------------------- Analysis --------------------
# -------------------- Analysis --------------------
with tabs[1]:
    st.header("📊 Analysis")

    # Plot Close price and moving averages
    fig, ax = plt.subplots(figsize=(12,5))
    plt.plot(data['Close']*USD_TO_INR, color="#1a237e", label="Close Price")
    plt.plot(data['MA100']*USD_TO_INR, color="#d84315", label="MA100")
    plt.plot(data['MA200']*USD_TO_INR, color="#00695c", label="MA200")
    plt.title(f"{stock} Price & Moving Averages (₹)", fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend()
    st.pyplot(fig)

    # ----- Heatmap of Daily % Change -----
    st.subheader("📈 Daily Percentage Change Heatmap")

    data['Daily Change %'] = data['Close'].pct_change() * 100
    # Create a pivot table: months vs years for heatmap
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    pivot = data.pivot_table(values='Daily Change %', index='Month', columns='Year', aggfunc='mean')

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
    plt.title(f"{stock} Average Daily % Change Heatmap")
    plt.ylabel("Month")
    plt.xlabel("Year")
    st.pyplot(plt)


# -------------------- Predictions --------------------
with tabs[2]:
    st.header("🤖 Model Predictions")
    if model:
        scaler = MinMaxScaler((0,1))
        scaled_close = scaler.fit_transform(data[['Close']])
        split = int(len(scaled_close) * 0.7)
        train, test = scaled_close[:split], scaled_close[split:]
        x_test, y_test = [], []
        for i in range(100, len(test)):
            x_test.append(test[i-100:i])
            y_test.append(test[i])
        x_test, y_test = np.array(x_test), np.array(y_test)
        predictions = model.predict(x_test, verbose=0)
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test.reshape(-1,1))

        # Convert to INR
        inv_predictions_inr = inv_predictions * USD_TO_INR
        inv_y_test_inr = inv_y_test * USD_TO_INR

        plot_df = pd.DataFrame({"Original": inv_y_test_inr.flatten(),
                                "Predicted": inv_predictions_inr.flatten()},
                               index=data.index[split+100:])
        st.line_chart(plot_df)

        from sklearn.metrics import mean_absolute_percentage_error
        mape = mean_absolute_percentage_error(inv_y_test_inr, inv_predictions_inr)
        accuracy = 100 - (mape*100)
        st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
    else:
        st.info("Model not found. Predictions unavailable.")

# -------------------- Future Forecast --------------------
with tabs[3]:
    st.header("🔮 Future Forecast (30 Days)")
    if model:
        last_100_scaled = scaled_close[-100:]
        future_input = last_100_scaled.reshape(1,100,1)
        future_preds = []
        for _ in range(30):
            next_pred = model.predict(future_input, verbose=0)[0][0]
            future_preds.append(next_pred)
            last_100_scaled = np.append(last_100_scaled[1:], [[next_pred]], axis=0)
            future_input = last_100_scaled.reshape(1,100,1)
        future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
        future_prices_inr = future_prices * USD_TO_INR
        future_dates = pd.date_range(start=pd.Timestamp(data.index[-1]) + pd.Timedelta(days=1), periods=30)
        future_df = pd.DataFrame({'Predicted Close': future_prices_inr}, index=future_dates)
        st.line_chart(future_df)
        st.metric("Price in 30 Days", f"₹{future_prices_inr[-1]:.2f}")
        change = ((future_prices_inr[-1]-data['Close'].iloc[-1]*USD_TO_INR)/(data['Close'].iloc[-1]*USD_TO_INR))*100
        st.metric("Expected Change", f"{change:.2f}%")
    else:
        st.info("Future forecast unavailable. Model not found.")

# -------------------- Stock Price Simulator --------------------
with tabs[4]:
    st.header("🎮 Stock Price Simulator")
    simulator_price_inr = st.number_input("Enter Starting Price (₹)", value=float(data['Close'].iloc[-1]*USD_TO_INR), step=1.0)
    simulator_price_usd = simulator_price_inr * INR_TO_USD
    simulator_days = st.number_input("Number of Days to Simulate", min_value=1, max_value=365, value=30, step=1)
    simulate_btn = st.button("Simulate Price")
    
    if simulate_btn:
        np.random.seed(42)
        simulated_prices_usd = [simulator_price_usd]
        for _ in range(simulator_days-1):
            change_pct = np.random.normal(0, 1)
            new_price = simulated_prices_usd[-1] * (1 + change_pct/100)
            simulated_prices_usd.append(new_price)

        # Convert simulated prices back to INR
        simulated_prices_inr = [price*USD_TO_INR for price in simulated_prices_usd]
        sim_dates = [datetime.today() + timedelta(days=i) for i in range(simulator_days)]
        sim_df = pd.DataFrame({"Simulated Price": simulated_prices_inr}, index=sim_dates)
        
        st.line_chart(sim_df)
        st.dataframe(sim_df.tail(10))

        # ====== RISK ASSESSMENT USING MAPE ======
        simulated_array = np.array(simulated_prices_inr)
        reference_array = np.full_like(simulated_array, simulator_price_inr)
        mape = np.mean(np.abs((simulated_array - reference_array) / reference_array)) * 100

        st.subheader("⚠️ Risk Assessment")
        st.metric("Simulated Price Volatility (MAPE)", f"{mape:.2f}%")

        # Risk category
        if mape < 2:
            st.success("Low Risk ✅")
        elif mape < 5:
            st.warning("Moderate Risk ⚠️")
        else:
            st.error("High Risk ❌")

