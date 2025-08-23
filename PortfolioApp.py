import streamlit as st 
import pandas as pd
import yfinance as yf
from utils.data_utils import load_market_data
import jax.numpy as jnp
from utils.volatility_calculator import VolatilityCalculator
import numpy as np 

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
        return_and_risk_metrics = {}

        if st.session_state.market_data is not None:
            for market in possible_markets:
                market_index_ticker = yf.Ticker(yf_market_tag_index[market][1])
                market_index_data = market_index_ticker.history(start=start_date, end=end_date, auto_adjust = False).asfreq('B').ffill()
                for eq_ticker_class in st.session_state.market_data[market]:
                    hist_data = eq_ticker_class.history(start=start_date, end=end_date, auto_adjust = False).asfreq('B').ffill()
                    
                    volatility_calculator = VolatilityCalculator()
                    beta_weekly = volatility_calculator.calculate_beta(hist_data['Adj Close'].values, market_index_data['Adj Close'].values, 5)
                    beta_monthly = volatility_calculator.calculate_beta(hist_data['Adj Close'].values, market_index_data['Adj Close'].values, 21)
                    daily_returns = hist_data['Adj Close'].pct_change(1).dropna()
                    weekly_returns = hist_data['Adj Close'].resample('W-FRI').last().pct_change(5).dropna()
                    monthly_returns = hist_data['Adj Close'].resample('ME').last().pct_change(21).dropna()

                    geo_daily = np.exp(np.log(daily_returns+1).mean()) - 1
                    geo_weekly = np.exp(np.log(weekly_returns+1).mean()) - 1
                    geo_monthly = np.exp(np.log(monthly_returns+1).mean()) - 1

                    return_and_risk_metrics[eq_ticker_class.ticker] = {
                        'Daily Mean Return': geo_daily,
                        'Weekly Mean Return': geo_weekly,
                        'Monthly Mean Return': geo_monthly,
                        'Beta (weekly)': beta_weekly,
                        'Beta (monthly)': beta_monthly
                    }
            return_and_risk_df = pd.DataFrame(return_and_risk_metrics).T
            st.dataframe(return_and_risk_df)

        # Filter according to user preferences for volatility and fundamental metrics

        # Select final equities to run portfolio model on

        # Display Time series Data for Selected Equities

        # Select Portfolio Model to run and select relevant parameters

        # Display final results and visualisations
        

except Exception as e:
    st.error(f"Error loading data: {e}")
