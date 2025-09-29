import streamlit as st
import pandas as pd

st.title("Portfolio Optimiser - Model Selection, Parameters and Equities")

if 'final_df' in st.session_state: 
    df = st.session_state.final_df

    # Slider values for PE, PB, Beta, ROE, Current Ration, Debt to Equity
    pe_min, pe_max = st.slider("Select P/E (trailing) range:", 0.0, 100.0, (0.0, 50.0), step=0.5)
    pb_min, pb_max = st.slider("Select Price to Book range:", 0.0, 20.0, (0.0, 5.0), step=0.1)
    beta_min, beta_max = st.slider("Select Beta (monthly) range:", -3.0, 3.0, (0.0, 1.5), step=0.05)
    roe_min, roe_max = st.slider("Select ROE range:", -1.0, 1.0, (0.0, 0.5), step=0.01)
    current_ratio_min, current_ratio_max = st.slider("Select Current Ratio range:", 0.0, 10.0, (0.0, 3.0), step=0.1)
    debt_to_equity_min, debt_to_equity_max = st.slider("Select Debt to Equity range:", 0.0, 5.0, (0.0, 1.0), step=0.05) 

    filtered_df = df[
        (df['P/E (trailing)'].between(pe_min, pe_max)) &
        (df['Price to Book'].between(pb_min, pb_max)) &
        (df['Beta (monthly)'].between(beta_min, beta_max)) &
        (df['ROE'].between(roe_min, roe_max)) &
        (df['Current Ratio'].between(current_ratio_min, current_ratio_max)) &
        (df['Debt to Equity'].between(debt_to_equity_min, debt_to_equity_max))
    ]
    st.dataframe(filtered_df)
else:
    st.warning("Please load market data first on the 'Data Loading' page.")
