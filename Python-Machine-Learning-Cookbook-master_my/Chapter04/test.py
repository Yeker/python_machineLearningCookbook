# from pandas_datareader import data, wb
# import datetime
# import matplotlib.pyplot as plt
# #
# # import fix_yahoo_finance as yf
# # yf.pdr_override()
# # end = datetime.now()
# # start = datetime(end.year - 1, end.month, end.day)
# # alibaba = data.get_data_yahoo('YHOO', start, end)
# #
# #
# #
# # alibaba['Adj Close'].plot(legend=True, figsize=(10, 4))
# # plt.show()
#
# from pandas_datareader import data as pdr
#
# import fix_yahoo_finance as yf
# import json
# import numpy as np
#
# yf.pdr_override()  # 需要调用这个函数
#
#
# symbol_file = 'symbol_map.json'
# with open(symbol_file, 'r') as f:
#     symbol_dict = json.loads(f.read())
# symbols, names = np.array(list(symbol_dict.items())).T
# # data = pdr.get_data_yahoo("TWX", start="2017-01-01", end="2017-04-30")
# # 获取数据
# # for symbol in symbols:
# # # data = pdr.get_data_yahoo("TWX", start="2017-01-01", end="2017-04-30")
# # # data = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
# #     try:
# #         print(symbol)
# #         data = pdr.get_data_yahoo(symbol, start="2017-01-01", end="2017-04-30")
# #         data['Adj Close'].plot(legend=True, figsize=(10, 4))
# #         plt.show()
# #     except ValueError:
# #         pass
#
# # Choose a time period
# start_date = datetime.datetime(2004, 4, 5)
# end_date = datetime.datetime(2007, 6, 2)
# quotes = []
# for symbol in symbols:
# # data = pdr.get_data_yahoo("TWX", start="2017-01-01", end="2017-04-30")
# # data = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
#     try:
#         print(symbol)
#         quotes.append(data.get_data_yahoo(symbol, start=start_date, end=end_date))
#     except ValueError:
#         pass
# # print(quotes)
#
# opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(np.float)
# closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(np.float)
# print(opening_quotes)
# print(closing_quotes)
#


#
print(max(80, 100, 1000)-min(80, 100, 1000))
