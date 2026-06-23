import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Market Research Bot", layout="wide")

st.title("📈 Stock Market Research Bot")
st.write("Enter a ticker symbol to get comprehensive market research and investor data")

# Sidebar for user input
with st.sidebar:
    st.header("Search")
    ticker = st.text_input("Enter Stock/ETF Ticker (e.g., AAPL, SPY):", placeholder="AAPL").upper()
    
    if st.button("Research", key="search_button"):
        st.session_state.search_ticker = ticker

# Main content
if "search_ticker" in st.session_state and st.session_state.search_ticker:
    ticker = st.session_state.search_ticker
    
    try:
        # Fetch ticker data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A'):.2f}" if isinstance(info.get('currentPrice'), (int, float)) else "N/A")
        
        with col2:
            market_cap = info.get('marketCap', 'N/A')
            if isinstance(market_cap, (int, float)):
                st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
            else:
                st.metric("Market Cap", "N/A")
        
        with col3:
            pe_ratio = info.get('trailingPE', 'N/A')
            st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A")
        
        with col4:
            div_yield = info.get('dividendYield', 'N/A')
            st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if isinstance(div_yield, (int, float)) else "N/A")
        
        st.divider()
        
        # Company description
        st.subheader("Company Overview")
        description = info.get('longBusinessSummary', 'No description available')
        st.write(description)
        
        st.divider()
        
        # Key metrics
        st.subheader("Key Investor Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.write(f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekHigh'), (int, float)) else "**52 Week High:** N/A")
            st.write(f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else "**52 Week Low:** N/A")
            st.write(f"**Average Volume:** {info.get('averageVolume', 'N/A'):,}" if isinstance(info.get('averageVolume'), (int, float)) else "**Average Volume:** N/A")
        
        with metrics_col2:
            st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if isinstance(info.get('fullTimeEmployees'), (int, float)) else "**Employees:** N/A")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        
        st.divider()
        
        # Price chart
        st.subheader("1-Year Price History")
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], 
                                    mode='lines', name='Close Price',
                                    fill='tozeroy'))
            fig.update_layout(
                title=f"{ticker} - 1 Year Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Financial highlights
        st.subheader("Financial Highlights")
        financial_col1, financial_col2, financial_col3 = st.columns(3)
        
        with financial_col1:
            pb_ratio = info.get('priceToBook', 'N/A')
            st.write(f"**Price/Book:** {pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else "**Price/Book:** N/A")
        
        with financial_col2:
            debt_to_equity = info.get('debtToEquity', 'N/A')
            st.write(f"**Debt/Equity:** {debt_to_equity:.2f}" if isinstance(debt_to_equity, (int, float)) else "**Debt/Equity:** N/A")
        
        with financial_col3:
            roe = info.get('returnOnEquity', 'N/A')
            st.write(f"**ROE:** {roe*100:.2f}%" if isinstance(roe, (int, float)) else "**ROE:** N/A")
        
        st.success(f"✅ Research complete for {ticker}")
        
    except Exception as e:
        st.error(f"❌ Error: Could not find ticker '{ticker}'. Please check the symbol and try again.")
        st.write("Example valid tickers: AAPL, MSFT, GOOGL, SPY, QQQ")

else:
    st.info("👈 Enter a ticker symbol in the sidebar to begin your market research")
