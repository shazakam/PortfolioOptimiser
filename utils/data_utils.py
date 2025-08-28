import pandas as pd 
import streamlit as st
import yfinance as yf
from .volatility_calculator import VolatilityCalculator
import numpy as np

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


def get_fundamental_loops(possible_markets:list, yf_market_tag_index : dict, start_date : str, end_date : str)-> dict:
    return_and_risk_metrics = {}
    fundamental_metrics = {}
    total_num_equitites = sum(len(st.session_state.market_data[market]) for market in possible_markets) if st.session_state.market_data is not None else 1

    data_loading_progress = st.progress(0, text="Processing Market Data...")
    total_processed = 1
    # Wrap this loop in a function and cache it
    for market in possible_markets:
        market_index_ticker = yf.Ticker(yf_market_tag_index[market][1])
        market_index_data = market_index_ticker.history(start=start_date, end=end_date, auto_adjust = False).asfreq('B').ffill()
        for eq_ticker_class in st.session_state.market_data[market]:
            # Calculate return and risk metrics from price data
            hist_data = eq_ticker_class.history(start=start_date, end=end_date, auto_adjust = False).asfreq('B').ffill()
            
            volatility_calculator = VolatilityCalculator()
            total_return = (hist_data['Adj Close'].iloc[-1] - hist_data['Adj Close'].iloc[0]) / hist_data['Adj Close'].iloc[0]
            beta_monthly = volatility_calculator.calculate_beta(hist_data['Adj Close'].values, market_index_data['Adj Close'].values, 21)
            monthly_returns = hist_data['Adj Close'].resample('ME').last().pct_change().dropna()
            annual_returns = hist_data['Adj Close'].resample('YE').last().pct_change().dropna()

            geo_monthly = np.exp(np.log(monthly_returns+1).mean()) - 1
            geo_annual = np.exp(np.log(annual_returns+1).mean()) - 1

            return_and_risk_metrics[eq_ticker_class.ticker] = {
                'Total Return over period': total_return,
                'Monthly Mean Return': geo_monthly,
                'Annual Mean Return': geo_annual,
                'Beta (monthly)': beta_monthly
            }

            # Get Fundamental Data
            fundamental_metrics[eq_ticker_class.ticker] = get_fundamentals(eq_ticker_class)

            data_loading_progress.progress(total_processed/total_num_equitites, text=f"Processing Market Data... ({total_processed}/{total_num_equitites})")
            total_processed += 1

    data_loading_progress.progress(1.0, text="Market Data Processed!")
    st.success("Market Data Processed Successfully!")
    print(total_num_equitites)
    return return_and_risk_metrics, fundamental_metrics


