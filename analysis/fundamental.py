import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

import plotly.express as px

st.session_state.fundamental_ticker = st.text_input("Insert Ticker for Fundamental Analysis")
st.session_state.fund_start_date = st.date_input("Start Date", value=pd.to_datetime("2021-07-01"))
st.session_state.fund_end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-28"))

if st.button("Run the fantastic fundamental analysis"):

        ticker_data = yf.Ticker(st.session_state.fundamental_ticker)
        hist_data = ticker_data.history(start = st.session_state.fund_start_date, end = st.session_state.fund_end_date).asfreq('B').ffill()
        hist_data.index = hist_data.index.tz_localize(None)
        st.write(f"Showing fundamental analysis for {st.session_state.fundamental_ticker}")
        st.line_chart(hist_data['Close'], use_container_width=True)
        info_dict = ticker_data.info
        st.write("### Key Information")
        st.write(f"**Sector:** {info_dict.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info_dict.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** {info_dict.get('marketCap', 'N/A')}")
        st.write(f'**Currency**: {info_dict.get("currency", "N/A")}')
        st.write(f"**Website:** {info_dict.get('website', 'N/A')}")
        st.write(f"**Description:** {info_dict.get('longBusinessSummary', 'N/A')}")

        # Chart of yearly PE ratio with industry average - maybe let user list competitors to get average? 

        eps = ticker_data.incomestmt.loc['Diluted EPS'].dropna()
        eps.index = eps.index.tz_localize(None)

        year_av = []

        for end_date in list(eps.index):
            start_date = end_date - pd.DateOffset(years=1)
            average = hist_data.loc[start_date:end_date]['Close'].mean()
            year_av.append(float(average))

        year_av.reverse()

        pe_ratio = pd.DataFrame(np.array(year_av) / np.array(eps), index=eps.index, columns=['Diluted P/E Ratio'])
        pe_ratio['Date'] = pe_ratio.index

        # Create Plotly line chart
        pe_fig = px.line(
            pe_ratio, 
            x="Date", 
            y="Diluted P/E Ratio", 
            title="Diluted P/E Ratio Over Time",
            markers=True
        )

        st.plotly_chart(pe_fig, use_container_width=True)

        # Chart of yearly PB - industry average

        # Chart of yearly EPS - industry average

        # Free cash flow yield

        # Profit margin

        # Current Ratio

        # Revenue Growth

        # DCF Valuation - maybe in another page?

        # LLM Future cash flow pediction!


        # Summary of quick checks

    # except:
    #     st.warning("Please enter a valid ticker symbol.")
