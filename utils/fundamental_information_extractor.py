import yfinance as yf 
import pandas as pd 
import numpy as np 

class FundamentalInformationExtractor():
    
    def __init__(self):
        pass

    def get_ticker_data(self, ticker: str, start : str, end : str) -> pd.DataFrame:
        """
        Fetches historical data for a given ticker.
        """
        data = yf.download(ticker, start, end)
        return data

    def calculate_pe_ratio(self, ticker: str) -> float:
       return
    
    def calculate_pb_ratio(self, ticker: str) -> float:
        return
    
    def calculate_return_on_equity(self, ticker: str) -> float:
        return
    
    def calculate_current_ratio(self, ticker: str) -> float:
        return
    
    def calculate_debt_to_equity_ratio(self, ticker: str) -> float:
        return
    
    def calculate_quick_ratio(self, ticker: str) -> float:
        return
    
    def calculate_operating_cash_flow(self, ticker: str) -> float:
        return
    
    def free_cash_flow(self, ticker: str) -> float:
        return
    
    
    

