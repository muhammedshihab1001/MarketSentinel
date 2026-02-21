import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/v1/portfolio"

st.set_page_config(page_title="MarketSentinel Dashboard", layout="wide")

st.title("📈 MarketSentinel Terminal")

tickers_input = st.text_input(
    "Enter tickers (comma separated)",
    "AAPL,MSFT,NVDA,JPM,SPY"
)

if st.button("Run Signal"):

    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    with st.spinner("Running inference..."):

        response = requests.post(
            API_URL,
            json={"tickers": tickers}
        )

    if response.status_code != 200:
        st.error(response.text)
    else:
        data = response.json()

        st.subheader("Model Metadata")
        st.json(data["meta"])

        portfolio = pd.DataFrame(data["portfolio"])

        st.subheader("Portfolio Signals")
        st.dataframe(portfolio)

        st.subheader("Long vs Short")

        longs = portfolio[portfolio["weight"] > 0]
        shorts = portfolio[portfolio["weight"] < 0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Long Exposure", round(longs["weight"].sum(), 3))
            st.dataframe(longs)

        with col2:
            st.metric("Short Exposure", round(shorts["weight"].sum(), 3))
            st.dataframe(shorts)