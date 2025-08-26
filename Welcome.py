import streamlit as st 
import pandas as pd
import yfinance as yf
from utils.data_utils import load_market_data, get_fundamentals
import jax.numpy as jnp
from utils.volatility_calculator import VolatilityCalculator
import numpy as np 
import plotly.express as px

st.title("Portfolio Optimiser")
st.write("Welcome to the Portfolio Optimiser App! Use the sidebar to navigate between pages.")
st.write("1. Data Loading: Load market data and calculate key metrics.")
st.write("2. Metric Filtering: Filter equities based on your preferences and select a portfolio model.")
st.write("3. Select the final collection of Stocks you wish to have in your Portfolio and view the results and visualizations of your portfolio.")

# @st.cache_data
# def get_equity_list() -> pd.DataFrame:
#     df = pd.read_csv(
#             "data/Euronext_Equities_2025-08-19.csv",
#             sep=";",          # Euronext uses semicolons
#             skiprows=0,       # adjust if file has metadata rows
#             on_bad_lines="skip"  # skip problematic rows if any
#         )
#     return df


# st.title("Portfolio Optimiser")
# try:
#     market_df = get_equity_list()

#     yf_market_tag_index = {
#         'Oslo Børs': ['OL', 'OSEBX.OL'],
#         'Euronext Paris': ['PA', 'Yabadadoo.PA']} # Add more markets, their yfinance ticker suffices and market index tickers here (for beta calculations)

#     # TODO: Add more markets
#     possible_markets = st.multiselect('Select Markets', ['Oslo Børs', 'Euronext Paris'], default=['Oslo Børs'])
#     start_date = st.date_input("Start Date", value=pd.to_datetime("2021-07-01"))
#     end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-28"))

#     if not possible_markets:
#         st.warning("Please select at least one market.")
#     else:
#         # Filter out equities based on selected markets AND if they have the relevant data available
#         filtered_df = market_df[market_df['Market'].isin(possible_markets)]
#         market_grouped_equities = filtered_df.groupby('Market')
        
#         ticker_dictionary = {}
#         for market in possible_markets:
#             tickers = market_grouped_equities.get_group(market)['Symbol'].tolist()
#             tickers = [f"{ticker}.{yf_market_tag_index[market][0]}" for ticker in tickers]
#             ticker_dictionary[market] = tickers  

#         if "market_data" not in st.session_state:
#             st.session_state.market_data = None      

#         # Load in yf Ticker for each equity
#         if st.button("Load Market Data"):
#             st.session_state.market_data = load_market_data(possible_markets, ticker_dictionary, start_date, end_date)
            
#         # Currency Adjustment (if needed)
    
#         # Calculate volatility metrics for each company and show results alongside fundamental data + some summary stats for the equities 
#         return_and_risk_metrics = {}

#         fundamental_metrics = {}

#         if st.session_state.market_data is not None:
#             for market in possible_markets:
#                 market_index_ticker = yf.Ticker(yf_market_tag_index[market][1])
#                 market_index_data = market_index_ticker.history(start=start_date, end=end_date, auto_adjust = False).asfreq('B').ffill()
#                 for eq_ticker_class in st.session_state.market_data[market]:

#                     # Calculate return and risk metrics from price data
#                     hist_data = eq_ticker_class.history(start=start_date, end=end_date, auto_adjust = False).asfreq('B').ffill()
                    
#                     volatility_calculator = VolatilityCalculator()
#                     total_return = (hist_data['Adj Close'].iloc[-1] - hist_data['Adj Close'].iloc[0]) / hist_data['Adj Close'].iloc[0]
#                     beta_monthly = volatility_calculator.calculate_beta(hist_data['Adj Close'].values, market_index_data['Adj Close'].values, 21)
#                     monthly_returns = hist_data['Adj Close'].resample('ME').last().pct_change().dropna()
#                     annual_returns = hist_data['Adj Close'].resample('YE').last().pct_change().dropna()

#                     geo_monthly = np.exp(np.log(monthly_returns+1).mean()) - 1
#                     geo_annual = np.exp(np.log(annual_returns+1).mean()) - 1

#                     return_and_risk_metrics[eq_ticker_class.ticker] = {
#                         'Total Return over period': total_return,
#                         'Monthly Mean Return': geo_monthly,
#                         'Annual Mean Return': geo_annual,
#                         'Beta (monthly)': beta_monthly
#                     }

#                     # Get Fundamental Data
#                     fundamental_metrics[eq_ticker_class.ticker] = get_fundamentals(eq_ticker_class)

#             return_and_risk_df = pd.DataFrame(return_and_risk_metrics).T 
#             fundamental_metrics_df = pd.DataFrame(fundamental_metrics).T
#             # Python
#             fundamental_metrics_df['P/E (forward)'] = pd.to_numeric(fundamental_metrics_df['P/E (forward)'].replace(['Infinity', 'inf', '-inf'], None), errors='coerce')
#             fundamental_metrics_df['P/E (trailing)'] = pd.to_numeric(fundamental_metrics_df['P/E (trailing)'].replace(['Infinity', 'inf', '-inf'], None), errors='coerce')

#             # Risk Metric Tabs
#             risk_tabs = st.tabs(["Return and Risk Metrics", "Total Return Distribution", "Monthly Return Distribution", "Annual Return Distribution", "Beta (monthly) Distribution"])
#             with risk_tabs[0]:
#                 st.title("Return and Risk Metrics")
#                 st.dataframe(return_and_risk_df)

#             for i, col in enumerate(return_and_risk_df.columns):
#                 with risk_tabs[i+1]:
#                     st.title(f"{col} Distribution")
#                     fig = px.histogram(return_and_risk_df[col], x=col, nbins=50, title=f"Distribution of {col}")
#                     st.plotly_chart(fig, use_container_width=True)

#             # Valuation and Profitability Metrics Tabs
#             valuation_columns = ['P/E (trailing)', 'P/E (forward)', 'Price to Book', 'ROE', 'ROA']
#             valuation_tabs = st.tabs(['Valuation and Profitability Metrics'] + valuation_columns)

#             with valuation_tabs[0]:
#                 st.title("Valuation and Profit Metrics")
#                 st.dataframe(fundamental_metrics_df[valuation_columns])

#             for i, col in enumerate(valuation_columns):
#                 with valuation_tabs[i+1]:
#                     st.title(f"{col} Distribution")
#                     fig = px.histogram(fundamental_metrics_df[col], x=col, nbins=50, title=f"Distribution of {col}")
#                     st.plotly_chart(fig, use_container_width=True)

#             # Liquidity and Solvency Metrics Tabs 
#             liquidity_columns = ['Current Ratio', 'Quick Ratio', 'Debt to Equity']
#             liquidity_tabs = st.tabs(['Liquidity and Solvency Metrics'] + liquidity_columns)

#             with liquidity_tabs[0]:
#                 st.title("Liquidity and Solvency Metrics")
#                 st.dataframe(fundamental_metrics_df[liquidity_columns])

#             for i, col in enumerate(liquidity_columns):
#                 with liquidity_tabs[i+1]:
#                     st.title(f"{col} Distribution")
#                     fig = px.histogram(fundamental_metrics_df[col].dropna(), x=col, nbins=50, title=f"Distribution of {col}")
#                     st.plotly_chart(fig, use_container_width=True)

#             st.session_state.final_df = pd.concat([return_and_risk_df, fundamental_metrics_df], axis=1)
        
# except Exception as e:
#     st.error(f"Error loading data: {e}")
