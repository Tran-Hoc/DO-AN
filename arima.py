# import data
# split data, train, test
# Scale data
# find p d q
# fit model
# predict
# plot predict and actual
# forecast n days

import pmdarima as pm
import streamlit as st

from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy
from dateutil.relativedelta import relativedelta 
import pandas as pd

@st.cache_data
def arima (data, num_year):
    data_set = data['Close']
# Seasonal - fit stepwise auto-ARIMA
    model_arima = pm.auto_arima(data_set, start_p=0, start_q=0,
                            test='adf',
                            max_p=3, max_q=3, m=1,
                            start_P=0, seasonal=False,
                            d=1, D=0, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return numpy.array(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # load dataset
    series = data_set

    # seasonal difference
    X = series.values
    days_in_year = 365
    differenced = difference(X, days_in_year)
    # fit model
    model = ARIMA(differenced, order=model_arima.order)
    model_fit = model.fit()


     # lấy lịch giao dịch
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar('NYSE')

    start_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
    end_date = start_date + relativedelta(years=num_year) 
    early = nyse.schedule(start_date=start_date, end_date=end_date)

    step = len(early)
    # multi-step out-of-sample forecast
    forecast = model_fit.forecast(steps=step)
    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    pred = list()
    pred.append(1)

    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        # print('Day %d: %f' % (day, inverted))
        pred.append(inverted)
        history.append(inverted)
        day += 1

    prediction = pred[1:]
    return  pd.DataFrame({ 'Date': early.index, 'Predict': prediction})