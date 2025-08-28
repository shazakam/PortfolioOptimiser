import jax.numpy as jnp
import pandas as pd
import numpy as np 
import streamlit as st

class PortofolioWeightCalculator():

    def __init__(self, equity_data : pd.DataFrame, expected_return_method:str, risk_tolerance:float):
        
        self.equity_data = equity_data # Convert to JNP array M x M
        self.expected_return_method = expected_return_method 
        self.risk_tolerance = risk_tolerance
        self.optimal_weights = None

    def optimise_weights(self, method : str) -> jnp.Array:

        if method == 'efficient_frontier':
            # Run efficient frontier method

            return
        
        elif method == 'descent':
            # Run gradient descent

            return
        else:
            st.warning("Not a valid method, you idiot")
            print('Not a valid method')

            return 
    
    def efficient_frontier_method(self, expected_returns : jnp.Array) -> jnp.Array:
        
        return
    
    def descent_based_method(self, expected_returns : jnp.Array) -> jnp.Array:
        
        return
    
    def calculate_covariance(self, expected_returns : jnp.Array) -> jnp.Array:
        
        return
    
    def visualise_frontier(self):
        
        return

    def historical_expected_returns(self, period:int) -> jnp.Array:

        # Calculate geometric mean returns over a given period

        return