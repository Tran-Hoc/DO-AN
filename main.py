from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from services import load_data, plot_data, plot_multiple_data, plot_volume, plot_data_predict
from lstm import lstm
from arima import arima
# Set page layout to wide
st.set_page_config(layout="wide", page_title="Forcastify", page_icon="📈")

# Sidebar

st.sidebar.title("Options")
start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2015, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())

# Header
st.markdown("<h1 style='text-align: center;'>Stock Forecast App 📈</h1>", unsafe_allow_html=True)

selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Forecasting", "Comparison"],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up", "card-checklist" ],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)

# Stock selection
stocks = ("AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "NFLX", "NVDA", "AMD")

# Stocks abreviations
selected_stock = st.sidebar.selectbox("Select stock for prediction", stocks)
selected_stocks = st.sidebar.multiselect("Select stocks for comparison", stocks)

years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

# Display a loading spinner while loading data
with st.spinner("Loading data..."):
    data = load_data(selected_stock, start_date, end_date)
    sleep(1)

# Display the success message
success_message = st.success("Data loaded successfully!")

# Introduce a delay before clearing the success message
sleep(1)

# Clear the success message
success_message.empty()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Convert end_date to datetime
end_date_datetime = pd.to_datetime(end_date)

# Filter forecast based on end_date
forecast = forecast[forecast['ds'] >= end_date_datetime]

# Dataframes Tab
if selected_tab == "Dataframes":
    # Display historical data
    st.markdown("<h2><span style='color: orange;'>{}</span> Historical Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays historical stock price data for {} from {} to {}.".format(selected_stock, start_date, end_date))
    
    # Copy data
    new_data = data.copy()

    # Drop Adj Close and Volume columns
    new_data = data.drop(columns=['Adj Close'])
    reversed_data = new_data[::-1]

    st.dataframe(reversed_data, use_container_width=True)
    # Raw data plot
    plot_data(data)

    # Data Volume plot
    plot_volume(data)
  
# Plots Tab
if selected_tab == "Plots":
    print("PLot")

# Statistics Tab
if selected_tab == "Statistics":
    st.markdown("<h2><span style='color: orange;'>Descriptive </span>Statistics</h2>", unsafe_allow_html=True)
    st.write("This section provides descriptive statistics for the selected stock.")

    # Descriptive Statistics Table
    # drop the date column
    data = data.drop(columns=['Date', 'Adj Close', 'Volume'])
    st.table(data.describe())

# Forecasting Tab    
if selected_tab == "Forecasting":

    # model selection

    model_list = ("ARIMA", "LSTM")
    selected_tab_model = st.selectbox("Select model for prediction", model_list)
    if selected_tab_model == "ARIMA":
        st.markdown("ARIMA model")
        predict_list = arima(data, years_to_predict)
        st.dataframe(predict_list)
        plot_data_predict(predict_list)

    if selected_tab_model =="LSTM":
        st.markdown("LSTM model")
        predict_list = lstm(data, years_to_predict)
        st.dataframe(predict_list)
        plot_data_predict(predict_list)


# Comparison Tab
if selected_tab == "Comparison":
    if selected_stocks:
        # Forecast multiple stocks
        stocks_data = []
        forcasted_data = []
        for stock in selected_stocks:
            stocks_data.append(load_data(stock, start_date, end_date))

        st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Comparison Plot</h2>".format(', '.join(selected_stocks)), unsafe_allow_html=True)
        st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(', '.join(selected_stocks), end_date, end_date + pd.Timedelta(days=period)))

        for i, data in enumerate(stocks_data):
            if data is not None:
                df_train = data[["Date", "Close"]]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                model = Prophet()
                model.fit(df_train)
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                forecast = forecast[forecast['ds'] >= end_date_datetime]
                st.markdown("<h3><span style='color: orange;'>{}</span> Forecast DataFrame</h3>".format(selected_stocks[i]), unsafe_allow_html=True)

                # Copy forecast dataframe
                new_forecast = forecast.copy()

                # Drop unwanted columns
                new_forecast = new_forecast.drop(columns=[
                    'additive_terms', 
                    'additive_terms_lower', 
                    'additive_terms_upper', 
                    'weekly', 
                    'weekly_lower', 
                    'weekly_upper', 
                    'yearly', 
                    'yearly_lower', 
                    'yearly_upper', 
                    'multiplicative_terms', 
                    'multiplicative_terms_lower', 
                    'multiplicative_terms_upper'
                ])

                # Rename columns
                new_forecast = new_forecast.rename(columns={
                    "ds": "Date", 
                    "yhat": "Close", 
                    "yhat_lower": "Close Lower",
                    "yhat_upper": "Close Upper",
                    "trend": "Trend", 
                    "trend_lower": "Trend Lower", 
                    "trend_upper": "Trend Upper"
                })

                st.dataframe(new_forecast, use_container_width=True)

                forcasted_data.append(forecast)

        plot_multiple_data(forcasted_data, selected_stocks)
    else:
        st.warning("Please select at least one stock if you want to compare them.")

# Detail tab
if selected_tab == "Detail":
    st.markdown("<h2>Detail tab ...</h2>", unsafe_allow_html=True)
    st.write("This section displays detail stock price data")
    

# Display balloons at the end
# st.balloons()