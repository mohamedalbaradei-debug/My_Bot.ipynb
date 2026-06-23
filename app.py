import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.graph_objects as go

# Import pandas_ta with error handling
try:
    import pandas_ta as ta
except ImportError:
    ta = None

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Investment Analyst Bot", layout="wide")
st.title("🤖 AI Investment Analyst Bot")
st.write("Enter a ticker symbol to get comprehensive market research and investment analysis")

# --- Configuration ---
def get_macro_and_market_cycle():
    """Fetch basic macro context: CPI, GDP, NFP proxies, yield curve, VIX, Fed rate probability."""
    # VIX
    try:
        vix_df = yf.download("^VIX", period="2d", progress=False)
        if not vix_df.empty:
            vix = vix_df['Close'].iloc[-1].item()
        else:
            vix = "N/A"
    except Exception as e:
        vix = f"Error: {e}"

    # 10-year Treasury Yield
    try:
        tnx_df = yf.download("^TNX", period="2d", progress=False)
        if not tnx_df.empty:
            us10y_yield = tnx_df['Close'].iloc[-1].item()
        else:
            us10y_yield = "N/A"
    except Exception as e:
        us10y_yield = f"Error: {e}"

    macro_note = (
        """📝 Macro data (general commentary based on recent trends):
  • CPI (~3.4%): Consumer Price Index, measures inflation. Current level is above the Fed's 2% target, suggesting persistent inflation (bearish for bonds, mixed for stocks depending on earnings).
  • GDP (~3.2%): Gross Product, measures economic output. Current growth is solid, indicating a healthy economy (bullish).
  • NFP (~+272k): Non-Farm Payrolls, measures job creation. Strong job growth indicates a robust labor market but could fuel inflation concerns (mixed).
  • Fed Funds Rate: Fed likely on hold (83% prob), meaning interest rates are expected to remain stable in the short term."""
    )
    cycle_note = (
        "Market Cycle: Yield curve slightly inverted, late‑cycle dynamics. "
        "General commentary on sector rotation."
    )
    return vix, us10y_yield, macro_note, cycle_note

def get_asset_data(ticker):
    """Fetch price, technicals, fundamentals for any asset type."""
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        df = asset.history(period="1y")
        if df.empty:
            return None, None, None, None

        # Technical indicators using pandas_ta if available
        if ta is not None and len(df) >= 200:
            try:
                df.ta.rsi(length=14, append=True)
                df.ta.macd(append=True)
                df.ta.sma(length=50, append=True)
                df.ta.sma(length=200, append=True)
            except:
                # Fallback: calculate manually if pandas_ta fails
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
        else:
            # Manual calculation fallback
            if len(df) >= 50:
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
            if len(df) >= 200:
                df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Support / Resistance
        recent = df.tail(60)
        support = recent['Low'].min()
        resistance = recent['High'].max()

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) >= 2 else None

        ma_cross_status = "N/A"
        if previous is not None and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']) and \
               pd.notna(previous['SMA_50']) and pd.notna(previous['SMA_200']):
                if previous['SMA_50'] <= previous['SMA_200'] and latest['SMA_50'] > latest['SMA_200']:
                    ma_cross_status = "Golden Cross (Bullish)"
                elif previous['SMA_50'] >= previous['SMA_200'] and latest['SMA_50'] < latest['SMA_200']:
                    ma_cross_status = "Death Cross (Bearish)"
                elif latest['SMA_50'] > latest['SMA_200']:
                    ma_cross_status = "SMA50 > SMA200 (Bullish Trend)"
                elif latest['SMA_50'] < latest['SMA_200']:
                    ma_cross_status = "SMA50 < SMA200 (Bearish Trend)"

        # RSI calculation (fallback manual)
        rsi_value = latest.get('RSI_14', 'N/A')
        if rsi_value == 'N/A' and len(df) >= 14:
            try:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_value = 100 - (100 / (1 + rs.iloc[-1]))
            except:
                rsi_value = 'N/A'

        # MACD calculation (simplified fallback)
        macd_value = latest.get('MACD_12_26_9', 'N/A')
        macd_signal_value = latest.get('MACDs_12_26_9', 'N/A')

        analysis = {
            "current_price": latest['Close'],
            "rsi": rsi_value,
            "macd": macd_value,
            "macd_signal": macd_signal_value,
            "sma50": latest.get('SMA_50', 'N/A'),
            "sma200": latest.get('SMA_200', 'N/A'),
            "support": support,
            "resistance": resistance,
            "volume": latest['Volume'],
            "change_1d": (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(df) >= 2 else 'N/A',
            "ma_cross_status": ma_cross_status
        }

        quote_type = info.get('quoteType', '').lower()
        return df, analysis, info, quote_type
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None, None, None, None

def get_rsi_sentiment(rsi):
    if not isinstance(rsi, (int, float)): return "N/A"
    if rsi < 30: return "Bullish (Oversold - potential rebound)"
    if rsi > 70: return "Bearish (Overbought - potential reversal)"
    if rsi >= 50: return "Neutral/Bullish (Stronger momentum)"
    return "Neutral/Bearish (Weaker momentum)"

def get_macd_sentiment(macd, macd_signal):
    if not (isinstance(macd, (int, float)) and isinstance(macd_signal, (int, float))): return "N/A"
    if macd > macd_signal: return "Bullish (Momentum trending up)"
    if macd < macd_signal: return "Bearish (Momentum trending down)"
    return "Neutral (Crossover pending)"

def get_sma_sentiment(price, sma):
    if not (isinstance(price, (int, float)) and isinstance(sma, (int, float))): return "N/A"
    if price > sma: return "Bullish (Price above average)"
    if price < sma: return "Bearish (Price below average)"
    return "Neutral (Price at average)"

def get_vix_sentiment(vix):
    if not isinstance(vix, (int, float)): return "N/A"
    if vix < 20: return "(Bullish - Low Volatility)"
    if vix > 30: return "(Bearish - High Volatility)"
    return "(Neutral - Moderate Volatility)"

def get_us10y_yield_sentiment(us10y_yield):
    if not isinstance(us10y_yield, (int, float)): return "N/A"
    if us10y_yield < 3.0: return "(Bullish - Lower Cost of Capital)"
    if us10y_yield > 4.0: return "(Bearish - Higher Cost of Capital)"
    return "(Neutral)"

def get_news_summary_placeholder():
    return "• News unavailable due to website scraping restrictions (403 Forbidden). Consider using a dedicated news API for reliable data."

def get_fear_greed_index_alternative():
    """Fetch Fear & Greed Index from alternative.me API (more reliable)."""
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['data']:
            index_value = data['data'][0]['value']
            index_classification = data['data'][0]['value_classification']
            return f"{index_classification} ({index_value}/100)"
        else:
            return "N/A (No data available)"
    except Exception as e:
        return f"N/A (Error: {str(e)})"

def get_fear_greed_index_placeholder():
    url = "https://feargreedmeter.com/fear-and-greed-index"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        sentiment_tag = soup.find('div', class_='fng-gauge_title')
        sentiment = sentiment_tag.text.strip() if sentiment_tag else 'N/A'
        index_value_tag = soup.find('div', class_='fng-gauge_value-text')
        index_value = index_value_tag.text.strip() if index_value_tag else 'N/A'
        if sentiment == 'N/A' and index_value == 'N/A':
            return "N/A (Could not find index on feargreedmeter.com)"
        return f"{sentiment} ({index_value})"
    except:
        return "N/A (Error fetching Fear & Greed Index)"

def get_top_performing_stocks(stock_list, num_top_stocks=5):
    performance = []
    for ticker in stock_list:
        try:
            df_perf = yf.download(ticker, period="1y", progress=False)
            if not df_perf.empty and len(df_perf) > 1:
                current_price = df_perf['Close'].iloc[-1].item()
                one_year_ago_price = df_perf['Close'].iloc[0].item()
                if pd.isna(one_year_ago_price) or one_year_ago_price == 0:
                    continue
                growth_percent = ((current_price - one_year_ago_price) / one_year_ago_price) * 100
                performance.append({'ticker': ticker, 'growth_percent': growth_percent})
        except:
            continue
    top_performers = sorted(performance, key=lambda x: x['growth_percent'], reverse=True)
    return top_performers[:num_top_stocks]

def get_recommendations(price, rsi, macd, macd_signal, sma50, sma200, info, ma_cross_status, support, resistance):
    trader_rec = {"action": "", "rationale": "", "stop_loss": "None", "target": "None"}
    investor_rec = {"action": "", "rationale": ""}

    bullish_signals = 0
    bearish_signals = 0

    if isinstance(rsi, (int, float)):
        if rsi < 40: bullish_signals += 1
        if rsi > 60: bearish_signals += 1

    if isinstance(macd, (int, float)) and isinstance(macd_signal, (int, float)):
        if macd > macd_signal: bullish_signals += 1
        if macd < macd_signal: bearish_signals += 1

    if isinstance(price, (int, float)):
        if isinstance(sma50, (int, float)) and price > sma50: bullish_signals += 1
        if isinstance(sma50, (int, float)) and price < sma50: bearish_signals += 1
        if isinstance(sma200, (int, float)) and price > sma200: bullish_signals += 1
        if isinstance(sma200, (int, float)) and price < sma200: bearish_signals += 1

    if "Golden Cross" in ma_cross_status: bullish_signals += 2
    if "Death Cross" in ma_cross_status: bearish_signals += 2

    if bullish_signals > bearish_signals:
        trader_rec["action"] = "Consider Buy/Long"
        trader_rec["rationale"] = "Technical indicators show short-term bullish momentum."
        trader_rec["stop_loss"] = f"${sma50:.2f}" if isinstance(sma50, (int, float)) else "None"
        trader_rec["target"] = f"${resistance:.2f}" if isinstance(resistance, (int, float)) else "None"
    elif bearish_signals > bullish_signals:
        trader_rec["action"] = "Consider Sell/Short"
        trader_rec["rationale"] = "Technical indicators show short-term bearish momentum."
        trader_rec["stop_loss"] = f"${sma50:.2f}" if isinstance(sma50, (int, float)) else "None"
        trader_rec["target"] = f"${support:.2f}" if isinstance(support, (int, float)) else "None"
    else:
        trader_rec["action"] = "Hold/Neutral"
        trader_rec["rationale"] = "Mixed signals, market direction unclear."

    if isinstance(rsi, (int, float)):
        if rsi < 30:
            trader_rec["action"] = "Consider Buy (Oversold)"
            trader_rec["rationale"] = "RSI indicates oversold, potential short-term bounce."
        elif rsi > 70:
            trader_rec["action"] = "Consider Sell/Short (Overbought)"
            trader_rec["rationale"] = "RSI indicates overbought, potential short-term pullback."

    if isinstance(price, (int, float)) and isinstance(sma200, (int, float)):
        if price > sma200:
            investor_rec["action"] = "Consider Accumulate/Hold"
            investor_rec["rationale"] = "Price above 200-day MA, suggesting uptrend."
        elif price < sma200:
            investor_rec["action"] = "Consider Avoid/Reduce"
            investor_rec["rationale"] = "Price below 200-day MA, suggesting downtrend."
        else:
            investor_rec["action"] = "Hold/Neutral"
            investor_rec["rationale"] = "Long-term trend unclear based on 200-day MA."

    return trader_rec, investor_rec

# Sidebar for ticker input
with st.sidebar:
    st.header("📊 Investment Analysis")
    ticker_input = st.text_input("Enter Ticker (AAPL, QQQ, SPY, etc.):", placeholder="QQQ").upper()
    search_button = st.button("Analyze", key="analyze_button")

# Main analysis
if search_button and ticker_input:
    with st.spinner(f"Analyzing {ticker_input}..."):
        df, analysis, info, quote_type = get_asset_data(ticker_input)
        
        if analysis is None:
            st.error("❌ Could not retrieve data. Check ticker symbol.")
        else:
            vix, us10y_yield, macro_note, cycle_note = get_macro_and_market_cycle()
            fear_greed_index_value = get_fear_greed_index_alternative()
            news_summary = get_news_summary_placeholder()

            # Fetch Bitcoin, Oil, Gold
            try:
                btc_data = yf.download("BTC-USD", period="1d", interval="1h", progress=False)
                bitcoin_price = btc_data['Close'].iloc[-1].item() if not btc_data.empty else "N/A"
            except:
                bitcoin_price = "N/A"

            try:
                oil_data = yf.download("CL=F", period="1d", interval="1h", progress=False)
                crude_oil_price = oil_data['Close'].iloc[-1].item() if not oil_data.empty else "N/A"
            except:
                crude_oil_price = "N/A"

            try:
                gold_data = yf.download("GC=F", period="1d", interval="1h", progress=False)
                gold_price = gold_data['Close'].iloc[-1].item() if not gold_data.empty else "N/A"
            except:
                gold_price = "N/A"

            # Stock scanner
            scan_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'PG', 'XOM', 'CVX', 'DIA', 'QQQ', 'SMH', 'UNH']
            top_performers = get_top_performing_stocks(scan_list)

            # Display Analysis
            today_str = datetime.today().strftime('%Y-%m-%d %H:%M')
            name = info.get('shortName', ticker_input)
            price = analysis['current_price']
            rsi = analysis['rsi']
            macd = analysis['macd']
            macd_signal = analysis['macd_signal']
            sma50 = analysis['sma50']
            sma200 = analysis['sma200']
            support = analysis['support']
            resistance = analysis['resistance']
            change = analysis['change_1d']
            ma_cross_status = analysis['ma_cross_status']

            st.success(f"✅ Report for {ticker_input} — {today_str}")
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${price:.2f}")
            with col2:
                st.metric("1D Change", f"{change:+.2f}%")
            with col3:
                st.metric("RSI(14)", f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "N/A")
            with col4:
                st.metric("Volume", f"{analysis['volume']:,.0f}" if isinstance(analysis['volume'], (int, float)) else "N/A")

            st.divider()

            # Technical Analysis
            st.subheader("📈 Technical Analysis")
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.write(f"**RSI Sentiment:** {get_rsi_sentiment(rsi)}")
                st.write(f"**MACD Sentiment:** {get_macd_sentiment(macd, macd_signal)}")
                st.write(f"**SMA50 Sentiment:** {get_sma_sentiment(price, sma50)}")
                st.write(f"**SMA200 Sentiment:** {get_sma_sentiment(price, sma200)}")
            
            with tech_col2:
                st.write(f"**MA Cross Status:** {ma_cross_status}")
                st.write(f"**Support:** ${support:.2f}")
                st.write(f"**Resistance:** ${resistance:.2f}")
                st.write(f"**SMA50:** ${sma50:.2f}" if isinstance(sma50, (int, float)) else "**SMA50:** N/A")

            st.divider()

            # Price Chart
            st.subheader("📊 1-Year Price Chart")
            if not df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                        mode='lines', name='Close Price',
                                        fill='tozeroy'))
                fig.update_layout(
                    title=f"{ticker_input} - 1 Year Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Macro Context
            st.subheader("🌍 Macro & Market Context")
            st.write(f"**VIX:** {f'{vix:.2f}' if isinstance(vix, (int, float)) else vix} {get_vix_sentiment(vix)}")
            st.write(f"**US 10Y Yield:** {f'{us10y_yield:.2f}%' if isinstance(us10y_yield, (int, float)) else us10y_yield} {get_us10y_yield_sentiment(us10y_yield)}")
            st.write(f"**Bitcoin:** ${f'{bitcoin_price:,.2f}' if isinstance(bitcoin_price, (int, float)) else bitcoin_price}")
            st.write(f"**Crude Oil:** ${f'{crude_oil_price:,.2f}' if isinstance(crude_oil_price, (int, float)) else crude_oil_price}")
            st.write(f"**Gold:** ${f'{gold_price:,.2f}' if isinstance(gold_price, (int, float)) else gold_price}")
            st.write(f"**Fear & Greed Index (alternative.me):** {fear_greed_index_value}")

            st.divider()

            # Recommendations
            trader_rec, investor_rec = get_recommendations(price, rsi, macd, macd_signal, sma50, sma200, info, ma_cross_status, support, resistance)
            
            st.subheader("🔮 AI Recommendations")
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.write("**For Traders (Short-term):**")
                st.write(f"Action: {trader_rec['action']}")
                st.write(f"Rationale: {trader_rec['rationale']}")
                st.write(f"Stop-Loss: {trader_rec['stop_loss']}")
                st.write(f"Target: {trader_rec['target']}")
            
            with rec_col2:
                st.write("**For Investors (Long-term):**")
                st.write(f"Action: {investor_rec['action']}")
                st.write(f"Rationale: {investor_rec['rationale']}")

            st.divider()

            # Top Performers
            st.subheader("📈 Top Performing Stocks (1-Year Growth)")
            if top_performers:
                for item in top_performers[:5]:
                    st.write(f"• **{item['ticker']}**: {item['growth_percent']:+.2f}%")
            else:
                st.write("No data available")

            st.divider()
            st.info("⚠️ Disclaimer: This is not financial advice. For educational purposes only. Past performance does not guarantee future results.")

else:
    st.info("👈 Enter a ticker in the sidebar to begin analysis")
