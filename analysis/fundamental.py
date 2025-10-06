import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from utils.data_utils import *
from utils.plotting_utils import *
import plotly.express as px

st.session_state.fundamental_ticker = st.text_input("Insert Ticker for Fundamental Analysis")
st.session_state.competitor_tickers = st.text_input("Insert Competitor Tickers (comma separated) for Industry Average Comparison")
st.session_state.fund_start_date = st.date_input("Start Date", value=pd.to_datetime("2021-07-01"))
st.session_state.fund_end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-28"))

if st.button("Run the fantastic fundamental analysis"):

        ticker_data = yf.Ticker(st.session_state.fundamental_ticker)
        hist_data = ticker_data.history(start = st.session_state.fund_start_date, end = st.session_state.fund_end_date).asfreq('B').ffill()
        hist_data.index = hist_data.index.tz_localize(None)
        summarise_fundamentals(hist_data, ticker_data)

        competitor_list = [comp.strip() for comp in st.session_state.competitor_tickers.split(',')] if st.session_state.competitor_tickers else []

        # Chart of yearly PE ratio with industry average - maybe let user list competitors to get average? 

        pe_ratio = calculate_pe_over_time(ticker_data, hist_data, competitor_list)
        
        pe_fig = plot_pe_over_time(pe_ratio)
        st.plotly_chart(pe_fig, use_container_width=True)

        # st.markdown("#### 📊 Performance Metrics")
        # col1, col2, col3 = st.columns(3)
        # col1.metric("Portfolio Value", "$1.2M", "+3.4%")
        # col2.metric("Sharpe Ratio", "1.45", "+0.08")
        # col3.metric("Volatility", "12.5%", "-0.6%")

        # Chart of yearly PB - industry average

        # Chart of yearly EPS - industry average

        # Free cash flow yield

        # Profit margin

        # Current Ratio

        # Revenue Growth

        # DCF Valuation - maybe in another page?

        # LLM Future cash flow pediction!

        # Summary of quick checks

    # except:
    #     st.warning("Please enter a valid ticker symbol.")
