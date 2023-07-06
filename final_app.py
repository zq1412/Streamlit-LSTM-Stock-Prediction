import streamlit as st

from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from plotly import graph_objs as go

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import forecasting
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.utils.likelihood_models import GaussianLikelihood

import yfinance as yf


#Getting date from one year ago
one_year_ago = datetime.now() - relativedelta(years=1) 
one_year_ago = one_year_ago.strftime("%Y-%m-%d")

#Getting date today
today = datetime.now()
today = today.strftime("%Y-%m-%d")

#Adding one day to the date one year ago 
one_year_ago_plus_one = datetime.now() - relativedelta(years=1) + timedelta(days=1)
one_year_ago_plus_one = one_year_ago_plus_one.strftime("%Y-%m-%d")


st.title('Stock Forecast App')

stocks = ('AAPL', 'ACN', 'ADBE', 'ADI', 'ADSK', 'AKAM', 'AMAT', 'AMD', 'ANET', 'ANSS', 'AVGO', 'CDNS', 'CDW', 'CRM', 'CSCO', 'CTSH', 'DXC', 'ENPH', 'EPAM', 'FFIV', 'FICO', 'FSLR', 'FTNT', 'GLW', 'INTU', 'IT', 'JNPR', 'KEYS', 'KLAC', 'LRCX', 'MCHP', 'MPWR', 'MSI', 'MU', 'NOW', 'NTAP', 'NVDA', 'NXPI', 'ON', 'PTC', 'QCOM', 'QRVO', 'ROP', 'ROP', 'SEDG', 'SNPS', 'SWKS', 'TEL', 'TER', 'TRMB', 'TXN', 'TYL', 'VRSN', 'ZBRA')
selected_stock = st.selectbox('Select the stock you wish to view/predict', stocks)

buying_or_selling = st.radio(
    "Are you looking to buy or sell?",
    ('Buy', 'Sell'))

if buying_or_selling == 'Buy':
    expected_ROI = st.number_input("How much ROI (in percentage) are you expecting?")

if buying_or_selling == 'Sell':
    expected_ROI = st.number_input("How much ROI (in percentage) are you expecting?")
    original_price = st.number_input("How much did you buy the stock for?")

def get_year_data(tick):
    #Accessing YFinance
    ticker= yf.Ticker(tick)

    #Getting pandas dataframe of stock data from one year ago
    df = ticker.history(start=one_year_ago, end=today, interval="1d")

    idx = pd.date_range(start=one_year_ago_plus_one, end=today)
    #Making index into Datetime index
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize(None) 

    #Reindexing dataframe to fill in missing dates due to stock market closing on weekends
    df = df.reindex(idx, method = 'pad')

    return df

def load_data_and_process(tick):
    #Accessing YFinance
    ticker= yf.Ticker(tick)

    #Getting pandas dataframe of stock data from one year ago
    df = ticker.history(start=one_year_ago, end=today, interval="1d")

    idx = pd.date_range(start=one_year_ago_plus_one, end=today)
    #Making index into Datetime index
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize(None) 

    #Reindexing dataframe to fill in missing dates due to stock market closing on weekends
    df = df.reindex(idx, method = 'pad')

    #Put dataframe into a Darts "Timeseries" object so that data can be fed into a Darts forecasting model. For more information on Timeseries objects, please look here: https://unit8co.github.io/darts/generated_api/darts.timeseries.html 
    series = TimeSeries.from_dataframe(df)

    #Drop all columns besides "Close" Column
    adj_series = series.drop_columns(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])

    #Transform training set, validation set, and entire series to values between 0 and 1.
    transformer = Scaler()

    #train_transformed = transformer.fit_transform(training)
    #val_transformed = transformer.transform(validation)

    series_transformed = transformer.fit_transform(adj_series)
    
    my_model = RNNModel(
    model="LSTM",
    n_rnn_layers = 4, # Number of LSTM layers
    hidden_dim=10,
    dropout=0.06,
    batch_size=31,
    n_epochs=15,
    optimizer_kwargs={"lr": 1e-3}, #learning rate
    pl_trainer_kwargs={
      "accelerator": "cpu",
      "devices": 1
    },
    model_name="Stock_Forecast",
    log_tensorboard=True,
    training_length=20,
    input_chunk_length=14,
    force_reset=True,
    save_checkpoints=True,
    )
    #Fit model on training data
    my_model.fit(series_transformed)

    pred = my_model.predict(n=60)
    #Perform inverse scaling on prediction
    predicted_values = transformer.inverse_transform(pred)
    #original_series = transformer.inverse_transform(series_transformed)
    #fig = plt.plot(predicted_values)
    #fig1 = plt.plot(adj_series)
    #st.pyplot(fig1, fig)

    last_value_from_pred = TimeSeries.pd_dataframe(predicted_values)["Close"][-1]
    print(last_value_from_pred)
    return predicted_values, adj_series, last_value_from_pred
  
data = get_year_data(selected_stock)
st.subheader('Data From Previous Year')
st.write(data)

predicted_values, adj_series, last_value_from_pred = load_data_and_process(selected_stock)
#ROI = ((predicted_values[-1]-adj_series[-1])/adj_series[-1]) * 100 
last_value_from_original = data["Close"][-1]

st.write(f'The forecasted price in two months is: {last_value_from_pred}')

if buying_or_selling == 'Buy':
    ROI = ((last_value_from_pred-last_value_from_original)/last_value_from_original) * 100 
    if ROI >= expected_ROI:
        st.write(f'ROI if bought today: {ROI}')
        st.write(f'Based on your expected ROI ({expected_ROI}), we recommend that this stock is worth looking into to purchase.')
    else:
        st.write(f'ROI if bought today: {ROI}')
        st.write(f'Based on your expected ROI ({expected_ROI}), we recommend that this stock is not worth looking into.')
if buying_or_selling == 'Sell':
    ROI_sell = ((last_value_from_pred-original_price)/original_price) * 100
    if ROI_sell >= expected_ROI: 
        st.write(f'ROI if sold today: {ROI_sell}')
        st.write(f'Based on your expected ROI ({expected_ROI}), we recommend that this stock is worth selling right now.')
    else:
        st.write(f'ROI if sold today: {ROI_sell}')
        st.write(f'Based on your expected ROI ({expected_ROI}), we recommend that this stock is not worth selling right now.')

st.subheader(f'{selected_stock}')
fig = plt.figure()
adj_series.plot(label='Data')
predicted_values.plot(label='forecast')
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
st.pyplot(fig)

