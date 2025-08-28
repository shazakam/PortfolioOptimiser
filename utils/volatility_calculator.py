import jax
import numpy as np
from sklearn.linear_model import LinearRegression

class VolatilityCalculator():

    def __init__(self):
        pass

    def calculate_beta(self, historical_data : np.array, market_data : np.array, return_period : int) -> float:
        # Compute simple returns over the given period
        returns = historical_data[return_period:] / historical_data[:-return_period] - 1
        market_returns = market_data[return_period:] / market_data[:-return_period] - 1

        # Reshape for sklearn regression
        X = market_returns.reshape(-1, 1)
        y = returns

        # Run linear regression (intercept = alpha, slope = beta)
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]

        return beta
    
    # Apply std Log Returns across DataFrame Columns
    # def calculate_std_log_returns_for_df(self, df, return_period, daily_week_monthly):
    #     volatility_dict = {}
    #     for column in df.columns:
    #         if column != 'Date':
    #             volatility = self.annualised_std_log_returns(jax.numpy.array(df[column].values), return_period, daily_week_monthly)
    #             volatility_dict[column] = volatility
    #     return volatility_dict
    

    # Apply Beta Calculation across DataFrame Columns
    def calculate_beta_for_df(self, df, market_data, return_period):
        beta_dict = {}
        for column in df.columns:
            if column != 'Date':
                beta = self.calculate_beta(df[column].values, market_data.values, return_period)
                beta_dict[column] = beta
        return beta_dict