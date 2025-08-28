import streamlit as st 
import pandas as pd
import yfinance as yf
from utils.data_utils import load_market_data, get_fundamentals, get_fundamental_loops
import jax.numpy as jnp
from utils.volatility_calculator import VolatilityCalculator
import numpy as np 
import plotly.express as px

@st.cache_data
def get_equity_list() -> pd.DataFrame:
    df = pd.read_csv(
            "data/Euronext_Equities_2025-08-19.csv",
            sep=";",          # Euronext uses semicolons
            skiprows=0,       # adjust if file has metadata rows
            on_bad_lines="skip"  # skip problematic rows if any
        )
    return df


st.title("Portfolio Optimiser")
try:
    market_df = get_equity_list()

    yf_market_tag_index = {
        'Oslo Børs': ['OL', 'OSEBX.OL'],
        'Euronext Paris': ['PA', 'Yabadadoo.PA']} # Add more markets, their yfinance ticker suffices and market index tickers here (for beta calculations)

    # TODO: Add more markets
    possible_markets = st.multiselect('Select Markets', ['Oslo Børs', 'Euronext Paris'], default=['Oslo Børs'])
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-07-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-28"))

    if not possible_markets:
        st.warning("Please select at least one market.")
    else:
        # Filter out equities based on selected markets AND if they have the relevant data available
        filtered_df = market_df[market_df['Market'].isin(possible_markets)]
        market_grouped_equities = filtered_df.groupby('Market')
        
        ticker_dictionary = {}
        for market in possible_markets:
            tickers = market_grouped_equities.get_group(market)['Symbol'].tolist()
            tickers = [f"{ticker}.{yf_market_tag_index[market][0]}" for ticker in tickers]
            ticker_dictionary[market] = tickers  

        if "market_data" not in st.session_state:
            st.session_state.market_data = None      

        # Load in yf Ticker for each equity
        if st.button("Load Market Data"):
            st.session_state.market_data = load_market_data(possible_markets, ticker_dictionary, start_date, end_date)
            
            
        # Currency Adjustment (if needed)
    
        # Calculate volatility metrics for each company and show results alongside fundamental data + some summary stats for the equities 
    
        if st.button("Process Market Data") and st.session_state.market_data is not None:

            return_and_risk_metrics, fundamental_metrics = get_fundamental_loops(possible_markets, yf_market_tag_index, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            return_and_risk_df = pd.DataFrame(return_and_risk_metrics).T 
            fundamental_metrics_df = pd.DataFrame(fundamental_metrics).T
            
            fundamental_metrics_df['P/E (forward)'] = pd.to_numeric(fundamental_metrics_df['P/E (forward)'].replace(['Infinity', 'inf', '-inf'], None), errors='coerce')
            fundamental_metrics_df['P/E (trailing)'] = pd.to_numeric(fundamental_metrics_df['P/E (trailing)'].replace(['Infinity', 'inf', '-inf'], None), errors='coerce')

            # Risk Metric Tabs
            risk_tabs = st.tabs(["Return and Risk Metrics", "Total Return Distribution", "Monthly Return Distribution", "Annual Return Distribution", "Beta (monthly) Distribution"])
            with risk_tabs[0]:
                st.title("Return and Risk Metrics")
                st.dataframe(return_and_risk_df)

            for i, col in enumerate(return_and_risk_df.columns):
                with risk_tabs[i+1]:
                    st.title(f"{col} Distribution")
                    fig = px.histogram(return_and_risk_df[col], x=col, nbins=50, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

            # Valuation and Profitability Metrics Tabs
            valuation_columns = ['P/E (trailing)', 'P/E (forward)', 'Price to Book', 'ROE', 'ROA']
            valuation_tabs = st.tabs(['Valuation and Profitability Metrics'] + valuation_columns)

            with valuation_tabs[0]:
                st.title("Valuation and Profit Metrics")
                st.dataframe(fundamental_metrics_df[valuation_columns])

            for i, col in enumerate(valuation_columns):
                with valuation_tabs[i+1]:
                    st.title(f"{col} Distribution")
                    fig = px.histogram(fundamental_metrics_df[col], x=col, nbins=50, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

            # Liquidity and Solvency Metrics Tabs 
            liquidity_columns = ['Current Ratio', 'Quick Ratio', 'Debt to Equity']
            liquidity_tabs = st.tabs(['Liquidity and Solvency Metrics'] + liquidity_columns)

            with liquidity_tabs[0]:
                st.title("Liquidity and Solvency Metrics")
                st.dataframe(fundamental_metrics_df[liquidity_columns])

            for i, col in enumerate(liquidity_columns):
                with liquidity_tabs[i+1]:
                    st.title(f"{col} Distribution")
                    fig = px.histogram(fundamental_metrics_df[col].dropna(), x=col, nbins=50, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

            st.session_state.final_df = pd.concat([return_and_risk_df, fundamental_metrics_df], axis=1)
        
except Exception as e:
    st.error(f"Error loading data: {e}")