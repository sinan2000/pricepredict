import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from info import api_key

def predict(symbol):
    # Collect data
    #symbol = input('Stock Ticker: ')
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    stock_data = data['Time Series (Daily)']
    df = pd.DataFrame(stock_data)
    df = df.transpose()
    df = df[['5. adjusted close']]
    df.rename(columns={'5. adjusted close':'closing_price'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[::-1]

    # Create training and test sets
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['closing_price']])
    time_steps = 30
    num_features = 1
    X_train = []
    y_train = []
    for i in range(time_steps, len(scaled_data)):
        X_train.append(scaled_data[i-time_steps:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], num_features))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(150, input_shape=(X_train.shape[1], num_features), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=30, batch_size=8)

    # Predict the future closing price in the next 30 days
    future_prices = []
    current_price = df.iloc[-time_steps:]['closing_price']
    current_price = scaler.transform(current_price.values.reshape(-1, 1))
    current_price = np.reshape(current_price, (1, time_steps, num_features))
    for i in range(30):
        next_price = model.predict(current_price)
        future_prices.append(scaler.inverse_transform(next_price)[0][0])
        current_price = np.append(current_price[:,1:,:], [[[next_price[0][0]]]], axis=1)

    print(future_prices)

    # Plot the prices
    plt.plot(range(len(df)), df['closing_price'], color='blue', label='Historical Prices')
    future_prices_index = range(len(df),len(df)+30)
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1,1))
    plt.plot(future_prices_index, future_prices, color='red', label='Predicted Prices')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.title('Historical and Predicted Prices')
    plt.legend()
    plt.show()
