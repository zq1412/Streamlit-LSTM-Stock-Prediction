import yfinance as yf
import pandas as pd
aapl= yf.Ticker("aapl")
df = aapl.history(start="2022-04-01", end="2023-04-01", interval="1d")

# data.reset_index(inplace=True)
# data = data[['Date', 'Close']]

idx = pd.date_range(start='2022-04-01', end='2023-04-01')
df.index = pd.DatetimeIndex(df.index)
df.index = df.index.tz_localize(None) 
# # #df['Date'] = pd.to_datetime(df['Date'])
# # #print(df)
# # #df.set_index='Date'
# # #print(df)
# # df.index = pd.DatetimeIndex(df.index)
# # print(df)
df = df.reindex(idx, method = 'pad')
print(df)