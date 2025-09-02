import streamlit as st
import pandas as pd
import jax.numpy as jnp
from itertools import chain
import plotly.express as px
from utils.optimiser import PortofolioWeightCalculator

# Select Stocks for model

market_data = st.session_state.market_data
equity_dict = dict(chain.from_iterable(d.items() for d in market_data.values()))

possible_equities = st.multiselect('Select Equities for Portfolio', equity_dict.keys())


if 'market_data' in st.session_state and len(possible_equities) > 0:
    # Display Time series Data for Selected Equities
    portfolio_timeseries = []

    for equity_ticker in possible_equities:
        equity_ticker_class = equity_dict[equity_ticker]
        equity_timeseries = equity_ticker_class.history(auto_adjust = False, start = st.session_state.start_date, end = st.session_state.end_date).asfreq('B').ffill()['Adj Close']

        portfolio_timeseries.append(equity_timeseries)

    portfolio_timeseries = pd.concat(portfolio_timeseries, axis = 1)
    portfolio_timeseries.columns = possible_equities

    equities_to_plot = st.multiselect('Select Equities to plot from selected equities', possible_equities)
    

    if equities_to_plot:

        # Select relevant columns
        plot_data = portfolio_timeseries[equities_to_plot].copy()

        # Reset index to make the dates a column
        plot_data = plot_data.reset_index().rename(columns={"index": "Date"})

        # Melt the DataFrame for Plotly long format
        plot_data_melted = plot_data.melt(
            id_vars="Date", 
            var_name="Equity", 
            value_name="Price"
        )

        # Create Plotly line chart
        fig = px.line(
            plot_data_melted, 
            x="Date", 
            y="Price", 
            color="Equity", 
            title="Equity Prices Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

        method = st.selectbox('Select Model Solution method', ['Direct Matrix Solution', 'Gradient Based solution'])

        if method == 'Direct Matrix Solution':
            st.text('This method solves a simple Lagrangian Optimisation problem with equality constraints. As a consequence, it does not enforce weights to be positive and that they only sum to one. Negative weights imply shorting and large weights (both in negative terms and positive terms) imply introducing leverage to achieve targeted return.')
            
            targeted_return = st.number_input(label='Input targeted return for portfolio', step = 0.01)
            period = st.number_input(label='Period over which to calculate returns (1 = 1 day)', step = 1)

            if st.button('Calculate weights'):
                opt = PortofolioWeightCalculator()
                weights = opt.efficient_frontier_method(portfolio_timeseries, period, targeted_return)

                weight_df = pd.DataFrame(weights[:-2].T, columns=portfolio_timeseries.columns, index = ['Portfolio Weights']).T
                st.dataframe(weight_df)
    # Select Portfolio Model to run and select relevant parameters

    # Display final model results and visualisations
else:
    st.warning("Please load market data first on the 'Data Loading' page.")
