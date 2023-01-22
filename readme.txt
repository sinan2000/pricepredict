##Stock Price Predict


###Description
This is a Python program that takes as input the ticker of a stock and tries to calculate the price for the particular stock for the future 30 days.
It works by collecting the closing prices of the stock by calling an API, creating a training model called LSTM (Long Short-Term Memory) with a single layer, 150 neurons, 8 training batches
that will train and try to predict the future price. The final result will be shown in a plot.

###Usage
You have to get a free API key from the website mentioned in the References section. Create a 'info.py' file inside the directory and add a single line like this:
api_key = <your_API_key>

You need to have all the used libraries installed on your machine. In the following updates, I will add an auto venv creator so that it can work automatically.

###References
API used for data collection: https://www.alphavantage.co/

Research: https://www.sciencedirect.com/science/article/pii/S2666827022000378