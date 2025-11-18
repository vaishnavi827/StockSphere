import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Download data
stock = 'GOOG'
data = yf.download(stock, start="2010-01-01", end="2021-01-01")
data = data[['Close']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split BEFORE sequence creation
split = int(len(scaled_data) * 0.7)
train_data = scaled_data[:split]
test_data = scaled_data[split:]

# Create training sequences
x_train, y_train = [], []
for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Create testing sequences
x_test, y_test = [], []
for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

# Compile & train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save model
model.save("Latest_stock_price_model.keras")
print("✅ Model trained and saved as 'Latest_stock_price_model.keras'")

# Predict on test data only
predictions = model.predict(x_test)

# Reverse scaling
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAPE
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# Convert to pseudo accuracy
accuracy = 100 - mape

print(f"MAPE on Test Data: {mape:.2f}%")
print(f"Pseudo Accuracy: {accuracy:.2f}%")
