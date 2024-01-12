# import data 
# spit data
# Normalization data
# create mode
# train model
# use model for predict
# plot predict and actuall
# Check accuracy
# forecast for future

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
from dateutil.relativedelta import relativedelta 

@st.cache_data
def lstm(data, num_year):
    train_data, test_data = data[0:int(len(data)*0.8)],data[int(len(data)*0.8):]
    # Doc du lieu 
    dataset_train = train_data
    training_set = dataset_train.iloc[:, 1:2].values

    # Thuc hien scale du lieu gia ve khoang 0,1
   
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Tao du lieu train, X = 60 time steps, Y =  1 time step
    X_train = []
    y_train = []
    no_of_sample = len(training_set)

    for i in range(600, no_of_sample):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    # Xay dung model LSTM
    regressor = Sequential()
    regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 64, return_sequences = False))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(32))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    regressor.fit(X_train, y_train, epochs = 2, batch_size = 32)

    # --------------------------------------------
        
    # Load du lieu tu 1/1/2019 - 2/10/2019
    dataset_test = test_data
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Tien hanh du doan
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    no_of_sample = len(inputs)

    for i in range(60, no_of_sample):
        X_test.append(inputs[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Du doan tiep gia cac ngay tiep theo 

    dataset_test = dataset_test['Close'][len(dataset_test)-60:len(dataset_test)].to_numpy()
    dataset_test = np.array(dataset_test)

    inputs = dataset_test
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    # lấy lịch giao dịch
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE')

    start_date = test_data['Date'].iloc[-1] + pd.Timedelta(days=1)
    end_date = start_date + relativedelta(years=num_year) 
    early = nyse.schedule(start_date=start_date, end_date=end_date)

    predict_list =[]
    i = 0
    while i<len(early):
        X_test = []
        no_of_sample = len(dataset_test)

        # Lay du lieu cuoi cung
        X_test.append(inputs[no_of_sample - 60:no_of_sample, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Du doan gia
        predicted_stock_price = regressor.predict(X_test)

        # chuyen gia tu khoang (0,1) thanh gia that
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        # Them ngay hien tai vao
        dataset_test = np.append(dataset_test, predicted_stock_price[0], axis=0)
        inputs = dataset_test
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)

        # print('Stock price ', predicted_stock_price[0][0])
        predict_list.append(predicted_stock_price[0][0])
        i = i +1
   
    return pd.DataFrame({ 'Date': early.index, 'Predict': predict_list})