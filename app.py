import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Trend Predictor")

# 1. Title
st.title("📉 Stock Price Trend Prediction using LSTM")
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.button("Predict"):

    # 2. Fetch data
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df[['Close']]
    st.write("Raw Closing Price Data", data.tail())

    # 3. Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    time_step = 60
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 4. LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM model..."):
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # 5. Prediction
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    # 6. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual, label='Actual Price')
    ax.plot(predictions, label='Predicted Price')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.legend()
    st.pyplot(fig)

    # 7. Save model and prediction
    model.save("lstm_stock_model.keras")
    pd.DataFrame(predictions, columns=["Predicted_Price"]).to_csv("predictions.csv", index=False)
    st.success("✅ Model saved and predictions exported!")

