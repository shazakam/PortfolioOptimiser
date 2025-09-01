import jax.numpy as jnp
import pandas as pd
import numpy as np 
import streamlit as st

class PortofolioWeightCalculator():

    def __init__(self, risk_tolerance:float):
        
        self.risk_tolerance = risk_tolerance
        self.optimal_weights = None
    
    # Should add a check to see if the final matrix is even invertible :/ 
    def efficient_frontier_method(self, equity_data, period : int, target_return_mu : float) -> jnp.Array:

        R = self.geometric_expected_returns(equity_data, period=period)
        cov = self.calculate_covariance(R)
        ones_vector = jnp.ones(R.shape)
        zero_vector = jnp.zeros(R.shape)

        row_0 = jnp.concatenate([2.0*cov, R, -1.0*ones_vector], axis = 1)
        row_1 = jnp.concatenate([R.T, zero_vector.T, zero_vector.T], axis = 1)
        row_2 = jnp.concatenate([ones_vector.T, zero_vector.T, zero_vector.T], axis = 1)

        A = jnp.concatenate([row_0, row_1, row_2], axis = 0)
        b = jnp.array([0.0, target_return_mu, 1])
        solution = jnp.linalg.solve(A, b)

        return solution
    
    def descent_based_method(self, expected_returns : jnp.Array) -> jnp.Array:
        
        return
    
    def calculate_covariance(self, expected_returns_timeseries : jnp.Array) -> jnp.Array:

        covariance = jnp.cov(expected_returns_timeseries, rowvar=False)
        
        return covariance
    
    def visualise_frontier(self):
        
        return

    def geometric_expected_returns(self, equity_data, period:int) -> jnp.Array:

        # Calculate geometric mean returns over a given period
        start_val = equity_data[0, :]
        end_val = equity_data[-1, :]

        ratio = end_val / start_val

        compound_growth = ratio**(1.0/period)
        
        return compound_growth