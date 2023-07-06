# Streamlit-Based LSTM S&P 500 Technology Stock Price Predictor

This streamlit based web app makes a live call to gather stock data from Yahoo Finance through the yfinance library and uses it to predict the stock price for the next two months using a LSTM model optimized using Bayesian Optimization. Using this prediction, the app makes a recommendation about whether the user should buy or sell the stock. The model is constructed using Darts (https://unit8co.github.io/darts/index.html) a timeseries forecasting library in Python. The predictions are based on the daily closing price of a stock over the course of the past year. 

## Information on the Different Files
### 1. app.py
This file contains the source code for the application. This is the file to run to use the application. 

### 2. ADBE_Hyperparameter_Optimization.ipynb
This is the code used to train the hyperparameters of the model. Optuna, a Baysian Hyperparameter Optimization library, is used to find the hyperparamters found in final_app.py. The hyperparameters for the model found in final_app.py are found using one stock's data, namely Adobe stock. The code for this file is adapted from the code found at this URL: https://unit8co.github.io/darts/userguide/hyperparameter_optimization.html

### 3. Final_model.ipynb
This is the experimental code for the LSTM model based on one stock's data. 

## Running the Application
1. Clone repository
2. Open app.py
3. Be sure to install the requirements in the requirements.txt file
4. Run code by running this command in the terminal: streamlit run app.py

## Screenshots 
![Alt text](https://user-images.githubusercontent.com/121399538/251434730-e2beccc1-04fa-42d7-882e-f4c848ced768.png)
![Alt text](https://user-images.githubusercontent.com/121399538/251434788-fcb1e700-1453-457c-98f6-9a1d92165721.png)
![Alt text](https://user-images.githubusercontent.com/121399538/251434816-c9de4523-a52b-4a24-aa13-ce86608b79e0.png)
## References:

### Hyperparameter Optimization:

Lectures on Bayesian Optimization
https://www.youtube.com/watch?v=i0cKa0di_lo&ab_channel=ArtificialIntelligence-AllinOne
https://www.youtube.com/watch?v=bcy6A57jAwI&t=440s&ab_channel=AIxplained

Optuna Lecture:
https://www.youtube.com/watch?v=P6NwZVl8ttc&t=75s&ab_channel=PyTorch

Optuna Code I adapted for my model:
https://unit8co.github.io/darts/userguide/hyperparameter_optimization.html

### App:

Streamlit Stock Price Prediction App: https://github.com/patrickloeber/python-fun/blob/master/stockprediction/main.py - I found out how to do the stock dropdown and how to show table data in the app through this code.  

Streamlit Documentation: https://docs.streamlit.io/

### Machine Lerning:

Some LSTM tutorials: 
https://www.youtube.com/watch?v=CbTU92pbDKw&ab_channel=GregHogg
https://www.youtube.com/watch?v=c0k-YLQGKjY&t=1509s&ab_channel=GregHogg

Darts Documentation: https://unit8co.github.io/darts/

Tutorial on Darts: 
https://www.youtube.com/watch?v=Kf6b5falv0M&t=563s&ab_channel=PyData

Timeseries forecasting example Code using Darts
https://colab.research.google.com/drive/10Z5fsjKPNqyaI9qMo-mgHb6i9l--Roye?usp=sharing

