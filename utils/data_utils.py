import pandas as pd 
import numpy as np 
import yfinance
import streamlit as st
import yfinance as yf

# Removes invalid tickers and only returns those with complete data for the specified date range
def load_market_data(possible_markets, ticker_dictionary, start_date, end_date)-> dict[str, list[yf.Ticker]]:
    market_ticker_classes_dict = {}
    end_date_inc = pd.to_datetime(end_date)+pd.Timedelta(days=1) # To ensure we get data for the end date as well
    data_loading_progress = st.progress(0, text="Loading Market Data...")
    for market in possible_markets:
        ticker_classes = []

        for tick in ticker_dictionary[market]:
            ticker_class = yf.Ticker(tick)
            hist_data = ticker_class.history(start=start_date, end=end_date_inc)
            if (not hist_data.empty):
                if (pd.to_datetime(hist_data.index[0]).date() == pd.to_datetime(start_date).date()) and (pd.to_datetime(hist_data.index[-1]).date() == pd.to_datetime(end_date).date()):
                    ticker_classes.append(ticker_class)

        data_loading_progress.progress(1/len(possible_markets))
        market_ticker_classes_dict[market] = ticker_classes

    data_loading_progress.progress(1.0, text="Market Data Loaded!")
    st.success("Market Data Loaded Successfully!")
    return market_ticker_classes_dict