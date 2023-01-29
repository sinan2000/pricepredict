import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from info import api_key

# Retrieve data
symbol = input('Stock Ticker: ')
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
response = requests.get(url)
data = response.json()
stock_data = data['Time Series (Daily)']

#Collect data
df = pd.DataFrame(stock_data)
df = df.transpose()
df = df[['5. adjusted close', '6. volume']]
df.rename(columns={'5. adjusted close':'closing_price', '6. volume':'volume'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.iloc[::-1]

# Add moving average of 30 days to DataFrame
df['mva_30'] = df['closing_price'].rolling(window=30).mean()

# Create training and test sets
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['closing_price', 'volume', 'mva_30']])
time_steps = 30
num_features = 3
X_train = []
y_train = []
for i in range(time_steps, len(scaled_data)):
    X_train.append(scaled_data[i-time_steps:i, :])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

# Create the LSTM model
model = Sequential()
model.add(LSTM(150, input_shape=(X_train.shape[1], num_features), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=1, batch_size=8)

# Predict the future closing price in the next 30 days
future_prices = []
current_price = df.iloc[-time_steps:][['closing_price', 'volume', 'mva_30']]
current_price = scaler.transform(current_price)
current_price = np.reshape(current_price, (1, time_steps, num_features))
for i in range(30):
    next_price = model.predict(current_price)
    next_price = scaler.inverse_transform(next_price)
    future_prices.append(next_price.reshape(-1, 1)[0][0])
    current_price = np.delete(current_price, 0, axis=1)
    current_price = np.append(current_price, [[next_price[0][0], df.iloc[-time_steps+i+1]['volume'], 
    df.iloc[-time_steps+i+1]['mva_30']]], axis=1)
    current_price = np.reshape(current_price, (1, time_steps, num_features))

future_prices = np.array(future_prices).reshape(-1,1)

# Plot
last_60_days = df.iloc[-60:]['closing_price'] # get the last 60 days closing prices
future_prices = scaler.inverse_transform(future_prices) # inverse transform the future prices to get the original scale
plt.plot(last_60_days, label='Last 60 Days', color='blue')
plt.plot(range(60, 90), future_prices, label='Future Prices', color='red')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()