#Optuna Hyperparameter Optimization
#import darts optuna yfinance

import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts import TimeSeries
from darts.metrics import smape, mape
from darts.models import RNNModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import forecasting
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller

from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import forecasting
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller


import yfinance as yf



## load data and preprocess

#Getting date from one year ago
one_year_ago = datetime.now() - relativedelta(years=1) 
one_year_ago = one_year_ago.strftime("%Y-%m-%d")

#Getting date today
today = datetime.now()
today = today.strftime("%Y-%m-%d")

#Adding one day to the date one year ago 
one_year_ago_plus_one = datetime.now() - relativedelta(years=1) + timedelta(days=1)
one_year_ago_plus_one = one_year_ago_plus_one.strftime("%Y-%m-%d")

#Accessing YFinance
ticker= yf.Ticker("ADBE")

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

# split in train / validation (note: in practice we would also need a test set)
VAL_LEN = 36
TEST_LEN = 37
train, val, test = adj_series[0:292], adj_series[292:-TEST_LEN], adj_series[-TEST_LEN:]

# scale
scaler = Scaler(MaxAbsScaler())
train = scaler.fit_transform(train)
val = scaler.transform(val)

# define objective function
def objective(trial):
    # select input and output chunk lengths
    input_chunk_length = trial.suggest_int("input_chunk_length", 7, 28)
    
  
    # Other hyperparameters
    batch_size = trial.suggest_int("batch_size", 16, 64)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 2, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 1, 10)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    n_epochs = trial.suggest_float("n_epochs", 10, 100 )


    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner, early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0


    # reproducibility
    torch.manual_seed(42)

    # build the LSTM model, set log_tensorboard = False, random_state was = 42
    my_model = RNNModel(
    model="LSTM",
    n_rnn_layers = n_rnn_layers, # Number of LSTM layers
    hidden_dim=hidden_dim,
    dropout=dropout,
    batch_size=batch_size,
    n_epochs=50,
    optimizer_kwargs={"lr": 1e-3}, #learning rate
    model_name="Stock_Forecast",
    training_length=30,
    input_chunk_length=input_chunk_length,
    force_reset=True,
    save_checkpoints=True,
    )

    # train the model
    my_model.fit(
        series=train,
        val_series=val,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    my_model = RNNModel.load_from_checkpoint("Stock_Forecast")

    # Evaluate how good it is on the validation set, using sMAPE
    preds = my_model.predict(series=train, n=len(test))
    smapes = smape(test, preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)

    return smape_val if smape_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, callbacks=[print_callback])