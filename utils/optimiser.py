import jax.numpy as jnp
import pandas as pd
import numpy as np 
import streamlit as st

class PortofolioWeightCalculator():

    def __init__(self, risk_tolerance:float):
        
        self.risk_tolerance = risk_tolerance
        self.optimal_weights = None
    
    # Should add a check to see if the final matrix is even invertible :/ 
    def efficient_frontier_method(self, equity_data : pd.DataFrame, period : int, target_return_mu : float) -> jnp.array:
        epsilon = 1e-6
        R = self.geometric_expected_returns(equity_data, period=period)
        R = R.reshape(-1,1)
        timeseries_returns = self.calculate_timeseries_returns(equity_data, period)

        cov = self.calculate_covariance(timeseries_returns)
        cov = cov + jnp.eye(cov.shape[0])*epsilon
        ones_vector = jnp.ones(R.shape)
        zero_vector = jnp.zeros(R.shape)

        row_0 = jnp.concatenate([2.0*cov, -R, -1.0*ones_vector], axis = 1)
        row_1 = jnp.concatenate([R.T, jnp.array([0.0]).reshape(-1,1), jnp.array([0.0]).reshape(-1,1)], axis = 1)
        row_2 = jnp.concatenate([ones_vector.T, jnp.array([0.0]).reshape(-1,1), jnp.array([0.0]).reshape(-1,1)], axis = 1)

        A = jnp.concatenate([row_0, row_1, row_2], axis = 0)

        print(A.shape)
        b = jnp.array([target_return_mu, 1])
        b = b.reshape(-1,1)
        b = jnp.concatenate([zero_vector, b], axis = 0)
        solution = jnp.linalg.solve(A, b)

        w = solution[:-2]         # portfolio weights
        lambda_1, lambda_2 = solution[-2:, 0]   # scalars

        print("Weights:", w)
        print("Sum of weights:", w.sum())
        print("Target return:", (R.T @ w)[0,0])
        print("Lambda1:", lambda_1, "Lambda2:", lambda_2)

        return solution
    
    def descent_based_method(self, expected_returns : jnp.array) -> jnp.array:
        
        return
    
    def calculate_covariance(self, expected_returns_timeseries : jnp.array) -> jnp.array:

        covariance = jnp.cov(expected_returns_timeseries, rowvar=False)
        
        return covariance
    
    def visualise_frontier(self):
        
        return

    def geometric_expected_returns(self, equity_data, period : int) -> jnp.array:

        # Calculate geometric mean returns over a given period
        data = jnp.array(equity_data.values)
        start_val = data[0, :]
        end_val = data[-1, :]

        ratio = end_val / start_val

        compound_growth = ratio**(1.0/period) - 1
        print(f'Compound Growth: {compound_growth}')
        return compound_growth
    
    def calculate_timeseries_returns(self, equity_data, period : int) -> jnp.array:
        timeseries_returns = equity_data.pct_change(period).dropna()
        return_array = jnp.array(timeseries_returns.values)
        return return_array