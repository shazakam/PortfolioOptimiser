import streamlit as st 
import pandas as pd
import yfinance as yf
from utils.data_utils import load_market_data, get_fundamentals
import jax.numpy as jnp
from utils.volatility_calculator import VolatilityCalculator
import numpy as np 
import plotly.express as px

welcome_page = st.Page("welcome/welcome_page.py", title="Welcome", default=True)

data_loading = st.Page(
    "portfolio/1_Data_Loading_page.py", title="Data Loading")

metric_filtering = st.Page("portfolio/2_Metric_Filtering_page.py", title="Metric Filtering")

portfolio_modelling = st.Page(
    "portfolio/3_Model_page.py", title="Portfolio Models")

equity_analysis = st.Page("analysis/fundamental.py", title="Fundamental Equity Analysis")

pg = st.navigation(
    {
        "Welcome": [welcome_page],
        "Portfolio Optimisation": [data_loading, metric_filtering, portfolio_modelling],
        "Fundamental Equity Analysis": [equity_analysis]
    }
)

pg.run()