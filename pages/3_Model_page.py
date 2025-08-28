import streamlit as st
import pandas as pd
import jax.numpy as jnp

# Select Stocks for model

if 'market_data' in st.session_state:
    market_data = st.session_state.market_data
    equities = []
    for market_key in market_data.keys():
        for ticker_data in market_data[market_key]:
            ticker_str = ticker_data.ticker
            equities.append(ticker_str)

    possible_markets = st.multiselect('Select Equities for Portfolio', equities)

else:
    st.warning("Please load market data first on the 'Data Loading' page.")

# Display Time series Data for Selected Equities

# Display basic info for them i.e. Returns, Earnings Growth yapa yapa yapa

# Select Portfolio Model to run and select relevant parameters

# Display final model results and visualisations