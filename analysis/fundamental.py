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
        st.markdown("### Trailing PE over time")
        pe_ratio = calculate_pe_over_time(ticker_data, hist_data, competitor_list)
        pe_fig = plot_ratio_over_time(pe_ratio, 'P/E', None, 'Current Industry Average P/E')
        st.plotly_chart(pe_fig, use_container_width=True)


        st.markdown("### Price To Book Value")
        pb = round(ticker_data.info['priceToBook'],2)
        pb_industry_average = round(calculate_industry_average_metric(competitor_list, 'priceToBook'),2)
        col1, col2 = st.columns(2,border=True)
        col1.metric('P/B Ratio', pb)
        col2.metric('Industry Average P/B Ratio', pb_industry_average)

        # Chart of yearly EPS - industry average
        st.markdown("### EPS over time")
        eps = pd.DataFrame(ticker_data.income_stmt.loc['Diluted EPS'].dropna())
        eps = eps.rename({'Diluted EPS': f'Diluted EPS ({ticker_data.info.get("currency", "N/A")} / Share)'}, axis = 1)
        eps['Date'] = eps.index.tz_localize(None)
        fig = px.line(eps.reset_index(), 
                      x='Date', 
                      y=f'Diluted EPS ({ticker_data.info.get("currency", "N/A")} / Share)',
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Current Free Cash Flow")
        # Free cash flow yield
        fcf_yield = round(ticker_data.cash_flow.loc['Free Cash Flow'].iloc[0] / ticker_data.info.get("marketCap"),2)
        fcf_average_yield = round(calculate_industry_average_cash_flow_yield(competitor_list),2)
        col1, col2 = st.columns(2,border=True)
        col1.metric('FCF Yield', fcf_yield)
        col2.metric('Industry Average FCF Yield', fcf_average_yield)

        # Current Ratio

        # Revenue Growth

        # DCF Valuation - maybe in another page?

        # LLM Future cash flow pediction!

        # Summary of quick checks

    # except:
    #     st.warning("Please enter a valid ticker symbol.")
