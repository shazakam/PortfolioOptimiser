import jax

class VolatilityCalculator():

    def __init__(self):
        pass

    def annualised_std_log_returns(self, historical_data : jax.Array, return_period : int, daily_week_monthly : str) -> float:
        # Calculate Log Returns
        log_returns = jax.numpy.log(historical_data[return_period:] / historical_data[:-return_period])

        # Calculate Standard Deviation of Log Returns
        if daily_week_monthly == 'daily':
            return jax.numpy.std(log_returns)*250
        elif daily_week_monthly == 'weekly':
            return jax.numpy.std(log_returns)*36
        elif daily_week_monthly == 'monthly':
            return jax.numpy.std(log_returns)*9

    def calculate_beta(self, historical_data : jax.Array, market_data : jax.Array, return_period : int) -> float:
        # Calculate Log Returns
        returns = historical_data[return_period:] / historical_data[:-return_period]
        market_returns = market_data[return_period:] / market_data[:-return_period]

        print(f"Log Returns: {returns}, Market Log Returns: {market_returns}")

        # Calculate Covariance and Variance
        covariance = jax.numpy.cov(returns, market_returns)[0][1]
        variance = jax.numpy.var(market_returns)

        # Calculate Beta
        beta = covariance / variance
        return beta
    
    # Apply std Log Returns across DataFrame Columns
    def calculate_std_log_returns_for_df(self, df, return_period, daily_week_monthly):
        volatility_dict = {}
        for column in df.columns:
            if column != 'Date':
                volatility = self.annualised_std_log_returns(jax.numpy.array(df[column].values), return_period, daily_week_monthly)
                volatility_dict[column] = volatility
        return volatility_dict
    

    # Apply Beta Calculation across DataFrame Columns
    def calculate_beta_for_df(self, df, market_data, return_period):
        beta_dict = {}
        for column in df.columns:
            if column != 'Date':
                beta = self.calculate_beta(jax.numpy.array(df[column].values), jax.numpy.array(market_data.values), return_period)
                beta_dict[column] = beta
        return beta_dict