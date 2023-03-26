import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64
from datetime import datetime, date
import yfinance as yf

def load_data(symbol, periods_input):
    try:
        # Load time series data
        data = yf.download(symbol, start="2018-07-02", end="2023-03-02")

        # Preprocess data
        appdata = pd.DataFrame(data)
        appdata = appdata.reset_index()
        appdata = appdata.rename(columns={"Date": "ds", "Close": "y"})
        appdata = appdata[['ds', 'y']]
        
        # Fit Prophet model
        model = Prophet()
        model.fit(appdata)
       
        # Make forecast
        future = model.make_future_dataframe(periods=periods_input)
        forecast = model.predict(future)
        forecast_filtered = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].loc[forecast['ds'] > appdata['ds'].max()]
       
        # Generate visualizations
        fig_forecast = model.plot(forecast)
        fig_components = model.plot_components(forecast)
       
        # Display results
        st.write(forecast_filtered)
        st.write(fig_forecast)
        st.write(fig_components)
    except:
        st.write("An error occurred while processing the data. Please check that the stock symbol is valid and try again.")

# Set up Streamlit app
st.title('Time Series Forecasting Using Streamlit')

# Get user input
symbol = st.selectbox('Select a stock', ['RELIANCE.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS'])
periods_input = st.number_input('How many days forecast do you want', min_value=1, max_value=365)

# Load and display data
if symbol != '':
    load_data(symbol, periods_input)