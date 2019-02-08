import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
# from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo
from pandas_datareader import data,wb


# Input symbol file
symbol_file = 'symbol_map.json'

# Choose a time period
start_date = datetime.datetime(2004, 4, 5)
end_date = datetime.datetime(2007, 6, 2)

# Load the symbol map
with open(symbol_file, 'r') as f:
    symbol_dict = json.loads(f.read())

symbols, names = np.array(list(symbol_dict.items())).T

# quotes = [data.DataReader(symbol, 'yahoo',start_date, end_date)
#                 for symbol in symbols]
# 获取数据
quotes = []
names2 = []
for symbol in symbols:
# data = pdr.get_data_yahoo("TWX", start="2017-01-01", end="2017-04-30")
# data = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
    try:
        print(symbol)
        quotes.append(data.get_data_yahoo(symbol, start=start_date, end=end_date))
        names2.append(names[symbols==symbol])
    except :
        print(symbol+'fail')
        pass

# Extract opening and closing quotes
opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(np.float)
closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(np.float)

# The daily fluctuations of the quotes 
delta_quotes = closing_quotes - opening_quotes

# Build a graph model from the correlations
edge_model = covariance.GraphLassoCV()

# Standardize the data 
X = delta_quotes.copy().T
X /= X.std(axis=0)

# Train the model
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Build clustering model using affinity propagation
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()
names3_1 = np.array(names2)
names3_2 = np.array([name[0] for name in names2])

# # Print the results of clustering
for i in range(num_labels + 1):
    print("Cluster", i+1, "-->", ', '.join(names3_2[labels == i]))

