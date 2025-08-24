import pandas as pd 
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

def get_fundamentals(ticker_class:yf.Ticker)-> dict:
    fundamental_columns = ['P/E (trailing)', 'P/E (forward)', 'EPS (trailing)', 'EPS (forward)', 'Price to Book', 'ROE', 'ROA', 'Current Ratio', 'Quick Ratio', 'Debt to Equity']
    pe_trailing = ticker_class.info.get('trailingPE', None)
    forward_pe = ticker_class.info.get('forwardPE', None)
    eps_trailing = ticker_class.info.get('trailingEps', None)
    forward_eps = ticker_class.info.get('forwardEps', None)
    price_to_book = ticker_class.info.get('priceToBook', None)
    ROE = ticker_class.info.get('returnOnEquity', None)
    ROA = ticker_class.info.get('returnOnAssets', None)
    current_ratio = ticker_class.info.get('currentRatio', None)
    debt_to_equity = ticker_class.info.get('debtToEquity', None) / 100 if ticker_class.info.get('debtToEquity', None) is not None else None
    quick_ratio = ticker_class.info.get('quickRatio', None)
    
    return dict(zip(fundamental_columns, [pe_trailing, forward_pe, eps_trailing, forward_eps, price_to_book, ROE, ROA, current_ratio, quick_ratio, debt_to_equity]))
