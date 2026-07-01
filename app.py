import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.graph_objects as go
from urllib.parse import quote
import json

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Professional Investment Analyst Bot", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .bullish { color: #00D084; font-weight: bold; }
    .bearish { color: #FF6B6B; font-weight: bold; }
    .neutral { color: #FFB627; font-weight: bold; }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============ TECHNICAL INDICATORS CALCULATION ============

def calculate_rsi(data, period=14):
    """Calculate RSI without pandas_ta"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD without pandas_ta"""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def calculate_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k_line = k_percent.rolling(window=smooth_k).mean()
    d_line = k_line.rolling(window=smooth_d).mean()
    return k_line, d_line

def get_market_breadth():
    """Get simplified market breadth - TRIN proxy using S&P 500"""
    try:
        spy = yf.download("SPY", period="1d", progress=False)
        if not spy.empty:
            volume = spy['Volume'].iloc[-1]
            close = spy['Close'].iloc[-1]
            open_price = spy['Open'].iloc[-1]
            
            if close > open_price:
                return "Bullish Breadth", volume
            else:
                return "Bearish Breadth", volume
        return "N/A", "N/A"
    except:
        return "N/A", "N/A"

def get_excess_liquidity_index():
    """Calculate simplified excess liquidity indicator"""
    try:
        btc = yf.download("BTC-USD", period="30d", progress=False)
        if len(btc) > 1:
            recent_vol = btc['Volume'].iloc[-1]
            avg_vol = btc['Volume'].rolling(window=20).mean().iloc[-1]
            liquidity_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            if liquidity_ratio > 1.3:
                return f"High Liquidity ({liquidity_ratio:.2f}x) - 🟢 Expansionary", "bullish"
            elif liquidity_ratio < 0.7:
                return f"Low Liquidity ({liquidity_ratio:.2f}x) - 🔴 Contractionary", "bearish"
            else:
                return f"Normal Liquidity ({liquidity_ratio:.2f}x) - 🟡 Neutral", "neutral"
        return "N/A", "N/A"
    except:
        return "N/A", "N/A"

# ============ NEWS SCRAPING WITHOUT API ============

def get_news_from_finviz(ticker):
    """Fetch news from Finviz without API"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        # Try to find news table
        news_table = soup.find('table', {'class': 'news-table'})
        if news_table:
            rows = news_table.findAll('tr')
            for row in rows[:5]:  # Get top 5 news
                try:
                    cols = row.findAll('td')
                    if len(cols) >= 2:
                        title = cols[1].get_text(strip=True)
                        time = cols[0].get_text(strip=True)
                        news_items.append(f"• **{time}** - {title}")
                except:
                    continue
        
        if news_items:
            return "\n".join(news_items)
        else:
            return "• No recent news found. Check financial news websites for updates."
    except:
        return "• Unable to fetch news. Try checking Yahoo Finance, Bloomberg, or CNBC directly."

def get_news_from_yahoo(ticker):
    """Fetch news from Yahoo Finance"""
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        if news:
            news_items = []
            for item in news[:5]:
                try:
                    title = item.get('title', 'No title')
                    source = item.get('source', 'Unknown')
                    news_items.append(f"• **{source}**: {title}")
                except:
                    pass
            if news_items:
                return "\n".join(news_items)
        return "• No recent news available from Yahoo Finance."
    except:
        return "• Unable to fetch news from Yahoo Finance."

# ============ MACRO AND MARKET DATA ============

def get_macro_and_market_cycle():
    """Fetch macro data"""
    try:
        vix_df = yf.download("^VIX", period="2d", progress=False)
        vix = vix_df['Close'].iloc[-1].item() if not vix_df.empty else "N/A"
    except:
        vix = "N/A"

    try:
        tnx_df = yf.download("^TNX", period="2d", progress=False)
        us10y_yield = tnx_df['Close'].iloc[-1].item() if not tnx_df.empty else "N/A"
    except:
        us10y_yield = "N/A"

    try:
        dxy_df = yf.download("DXY=F", period="2d", progress=False)
        dxy = dxy_df['Close'].iloc[-1].item() if not dxy_df.empty else "N/A"
    except:
        dxy = "N/A"

    macro_note = "📊 Market Context: Monitor inflation, GDP growth, unemployment, and Fed policy."
    cycle_note = "📈 Market Cycle: Late-cycle dynamics with fluctuating volatility."
    
    return vix, us10y_yield, dxy, macro_note, cycle_note

# ============ ASSET DATA WITH INDICATORS ============

def get_asset_data(ticker):
    """Fetch 2-year data with comprehensive technical indicators"""
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        df = asset.history(period="2y")
        if df.empty:
            return None, None, None, None

        # Calculate technical indicators
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        macd, macd_signal, macd_hist = calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        
        k_line, d_line = calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = k_line
        df['Stoch_D'] = d_line

        # Support & Resistance
        recent = df.tail(60)
        support = recent['Low'].min()
        resistance = recent['High'].max()

        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) >= 2 else None

        ma_cross_status = "N/A"
        if previous is not None and pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
            if pd.notna(previous['SMA_50']) and pd.notna(previous['SMA_200']):
                if previous['SMA_50'] <= previous['SMA_200'] and latest['SMA_50'] > latest['SMA_200']:
                    ma_cross_status = "🟢 Golden Cross (Bullish)"
                elif previous['SMA_50'] >= previous['SMA_200'] and latest['SMA_50'] < latest['SMA_200']:
                    ma_cross_status = "🔴 Death Cross (Bearish)"
                elif latest['SMA_50'] > latest['SMA_200']:
                    ma_cross_status = "🟢 Bullish Trend"
                elif latest['SMA_50'] < latest['SMA_200']:
                    ma_cross_status = "🔴 Bearish Trend"

        analysis = {
            "current_price": latest['Close'],
            "rsi": latest.get('RSI_14', np.nan),
            "macd": latest.get('MACD', np.nan),
            "macd_signal": latest.get('MACD_Signal', np.nan),
            "macd_hist": latest.get('MACD_Hist', np.nan),
            "sma50": latest.get('SMA_50', np.nan),
            "sma200": latest.get('SMA_200', np.nan),
            "atr": latest.get('ATR_14', np.nan),
            "stoch_k": latest.get('Stoch_K', np.nan),
            "stoch_d": latest.get('Stoch_D', np.nan),
            "bb_upper": latest.get('BB_Upper', np.nan),
            "bb_lower": latest.get('BB_Lower', np.nan),
            "bb_middle": latest.get('BB_Middle', np.nan),
            "support": support,
            "resistance": resistance,
            "volume": latest['Volume'],
            "change_1d": (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(df) >= 2 else 0,
            "ma_cross_status": ma_cross_status
        }

        quote_type = info.get('quoteType', '').lower()
        return df, analysis, info, quote_type
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None, None, None, None

# ============ SENTIMENT FUNCTIONS ============

def get_rsi_sentiment(rsi):
    if pd.isna(rsi): return "⚪ N/A", "neutral"
    if rsi < 30: return "🟢 Oversold (BUY Signal)", "bullish"
    if rsi > 70: return "🔴 Overbought (SELL Signal)", "bearish"
    if rsi >= 50: return "🟢 Bullish Momentum", "bullish"
    return "🟡 Weak Momentum", "neutral"

def get_macd_sentiment(macd, signal):
    if pd.isna(macd) or pd.isna(signal): return "⚪ N/A", "neutral"
    if macd > signal: return "🟢 Bullish (Momentum Up)", "bullish"
    if macd < signal: return "🔴 Bearish (Momentum Down)", "bearish"
    return "🟡 Neutral", "neutral"

def get_stoch_sentiment(k, d):
    if pd.isna(k) or pd.isna(d): return "⚪ N/A", "neutral"
    if k < 20 and d < 20: return "🟢 Oversold (BUY)", "bullish"
    if k > 80 and d > 80: return "🔴 Overbought (SELL)", "bearish"
    if k > d: return "🟢 Bullish Crossover", "bullish"
    if k < d: return "🔴 Bearish Crossover", "bearish"
    return "🟡 Neutral", "neutral"

def get_sma_sentiment(price, sma):
    if pd.isna(price) or pd.isna(sma): return "⚪ N/A", "neutral"
    if price > sma: return "🟢 Bullish (Above MA)", "bullish"
    if price < sma: return "🔴 Bearish (Below MA)", "bearish"
    return "🟡 At MA", "neutral"

def get_vix_sentiment(vix):
    if pd.isna(vix) or not isinstance(vix, (int, float)): return "⚪ N/A", "neutral"
    if vix < 15: return "🟢 Very Low - Complacent Market", "bullish"
    if vix < 20: return "🟢 Low Volatility", "bullish"
    if vix > 30: return "🔴 High Fear Index", "bearish"
    return "🟡 Moderate Volatility", "neutral"

def get_us10y_sentiment(yield_val):
    if pd.isna(yield_val) or not isinstance(yield_val, (int, float)): return "⚪ N/A", "neutral"
    if yield_val < 3.0: return "🟢 Lower Rates - Bullish", "bullish"
    if yield_val > 4.5: return "🔴 Higher Rates - Bearish", "bearish"
    return "🟡 Moderate", "neutral"

def get_atr_sentiment(atr, price):
    if pd.isna(atr) or pd.isna(price): return "⚪ N/A", "neutral"
    atr_percent = (atr / price) * 100
    if atr_percent > 3: return f"🔴 High Volatility ({atr_percent:.1f}%)", "bearish"
    if atr_percent < 1: return f"🟢 Low Volatility ({atr_percent:.1f}%)", "bullish"
    return f"🟡 Moderate Volatility ({atr_percent:.1f}%)", "neutral"

def get_bb_sentiment(price, bb_upper, bb_lower):
    if pd.isna(price) or pd.isna(bb_upper) or pd.isna(bb_lower): return "⚪ N/A", "neutral"
    if price > bb_upper: return "🔴 Above Upper Band (Overbought)", "bearish"
    if price < bb_lower: return "🟢 Below Lower Band (Oversold)", "bullish"
    return "🟡 Within Bands (Normal)", "neutral"

# ============ COMPARISON CHART ============

def create_comparison_chart(ticker, df):
    """Create 2-year performance comparison with S&P 500"""
    try:
        spy = yf.download("SPY", start=df.index[0], end=df.index[-1], progress=False)
        
        # Normalize prices to start at 100
        ticker_normalized = (df['Close'] / df['Close'].iloc[0]) * 100
        spy_normalized = (spy['Close'] / spy['Close'].iloc[0]) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=ticker_normalized,
            name=f'{ticker}', line=dict(color='#667eea', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=spy.index, y=spy_normalized,
            name='S&P 500 (SPY)', line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{ticker} vs S&P 500 (2-Year Performance)',
            xaxis_title='Date',
            yaxis_title='Performance (Base = 100)',
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        return fig
    except:
        return None

# ============ MAIN UI ============

st.title("🚀 Professional Investment Analyst Bot")
st.markdown("### Enterprise-Grade Technical Analysis & Market Intelligence")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    ticker_input = st.text_input("Enter Ticker Symbol", placeholder="AAPL", value="").upper()
    analyze_button = st.button("🔍 Analyze", use_container_width=True)

if analyze_button and ticker_input:
    with st.spinner(f"⏳ Analyzing {ticker_input} - Running comprehensive technical analysis..."):
        df, analysis, info, quote_type = get_asset_data(ticker_input)
        
        if analysis is None:
            st.error("❌ Could not retrieve data. Please verify the ticker symbol.")
        else:
            vix, us10y_yield, dxy, macro_note, cycle_note = get_macro_and_market_cycle()
            breadth_sentiment, breadth_volume = get_market_breadth()
            liquidity_status, liquidity_type = get_excess_liquidity_index()
            
            # Get news
            news = get_news_from_yahoo(ticker_input)
            if "No recent" in news or "Unable" in news:
                news = get_news_from_finviz(ticker_input)
            
            # ========== HEADER SECTION ==========
            st.markdown('<div class="section-header">📊 KEY METRICS DASHBOARD</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Price", f"${analysis['current_price']:.2f}")
            with col2:
                change_color = "🟢" if analysis['change_1d'] >= 0 else "🔴"
                st.metric("1D Change", f"{analysis['change_1d']:+.2f}%", delta=f"{change_color}")
            with col3:
                st.metric("Volume", f"{analysis['volume']/1e6:.1f}M")
            with col4:
                if isinstance(vix, (int, float)):
                    st.metric("VIX Index", f"{vix:.2f}")
            with col5:
                if isinstance(us10y_yield, (int, float)):
                    st.metric("10Y Yield", f"{us10y_yield:.2f}%")
            
            st.markdown("---")
            
            # ========== TECHNICAL INDICATORS ==========
            st.markdown('<div class="section-header">📈 TECHNICAL INDICATORS</div>', unsafe_allow_html=True)
            
            ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
            
            with ind_col1:
                st.markdown("#### RSI(14) Momentum")
                rsi_val = analysis['rsi']
                if not pd.isna(rsi_val):
                    rsi_sentiment, rsi_type = get_rsi_sentiment(rsi_val)
                    color = "bullish" if rsi_type == "bullish" else "bearish" if rsi_type == "bearish" else "neutral"
                    st.markdown(f"<span class='{color}'>{rsi_val:.1f}</span>", unsafe_allow_html=True)
                    st.caption(rsi_sentiment)
                else:
                    st.info("Calculating...")
            
            with ind_col2:
                st.markdown("#### MACD Trend")
                macd_val = analysis['macd']
                macd_sig = analysis['macd_signal']
                if not pd.isna(macd_val) and not pd.isna(macd_sig):
                    macd_sentiment, macd_type = get_macd_sentiment(macd_val, macd_sig)
                    color = "bullish" if macd_type == "bullish" else "bearish" if macd_type == "bearish" else "neutral"
                    st.markdown(f"<span class='{color}'>{macd_val:.4f}</span>", unsafe_allow_html=True)
                    st.caption(macd_sentiment)
                else:
                    st.info("Calculating...")
            
            with ind_col3:
                st.markdown("#### Stochastic %K")
                stoch_k = analysis['stoch_k']
                stoch_d = analysis['stoch_d']
                if not pd.isna(stoch_k):
                    stoch_sentiment, stoch_type = get_stoch_sentiment(stoch_k, stoch_d)
                    color = "bullish" if stoch_type == "bullish" else "bearish" if stoch_type == "bearish" else "neutral"
                    st.markdown(f"<span class='{color}'>{stoch_k:.1f}</span>", unsafe_allow_html=True)
                    st.caption(stoch_sentiment)
                else:
                    st.info("Calculating...")
            
            with ind_col4:
                st.markdown("#### ATR Volatility")
                atr_val = analysis['atr']
                if not pd.isna(atr_val):
                    atr_sentiment, atr_type = get_atr_sentiment(atr_val, analysis['current_price'])
                    color = "bullish" if atr_type == "bullish" else "bearish" if atr_type == "bearish" else "neutral"
                    st.markdown(f"<span class='{color}'>${atr_val:.2f}</span>", unsafe_allow_html=True)
                    st.caption(atr_sentiment)
                else:
                    st.info("Calculating...")
            
            st.markdown("---")
            
            # ========== MOVING AVERAGES & TREND ==========
            st.markdown('<div class="section-header">🎯 TREND ANALYSIS</div>', unsafe_allow_html=True)
            
            trend_col1, trend_col2, trend_col3 = st.columns(3)
            
            with trend_col1:
                st.markdown("#### 50-Day SMA")
                sma50_val = analysis['sma50']
                if not pd.isna(sma50_val):
                    sma50_sentiment, sma50_type = get_sma_sentiment(analysis['current_price'], sma50_val)
                    color = "bullish" if sma50_type == "bullish" else "bearish"
                    st.markdown(f"<span class='{color}'>${sma50_val:.2f}</span>", unsafe_allow_html=True)
                    st.caption(sma50_sentiment)
            
            with trend_col2:
                st.markdown("#### 200-Day SMA")
                sma200_val = analysis['sma200']
                if not pd.isna(sma200_val):
                    sma200_sentiment, sma200_type = get_sma_sentiment(analysis['current_price'], sma200_val)
                    color = "bullish" if sma200_type == "bullish" else "bearish"
                    st.markdown(f"<span class='{color}'>${sma200_val:.2f}</span>", unsafe_allow_html=True)
                    st.caption(sma200_sentiment)
            
            with trend_col3:
                st.markdown("#### Trend Status")
                st.info(analysis['ma_cross_status'])
            
            st.markdown("---")
            
            # ========== SUPPORT & RESISTANCE ==========
            st.markdown('<div class="section-header">💪 SUPPORT & RESISTANCE LEVELS</div>', unsafe_allow_html=True)
            
            sr_col1, sr_col2, sr_col3 = st.columns(3)
            
            with sr_col1:
                st.markdown(f"**🟢 Support**")
                st.markdown(f"### ${analysis['support']:.2f}")
                st.caption("Key buying zone")
            
            with sr_col2:
                st.markdown(f"**Current**")
                st.markdown(f"### ${analysis['current_price']:.2f}")
                distance_to_support = ((analysis['current_price'] - analysis['support']) / analysis['current_price']) * 100
                distance_to_resistance = ((analysis['resistance'] - analysis['current_price']) / analysis['current_price']) * 100
                st.caption(f"±{min(distance_to_support, distance_to_resistance):.1f}% to nearest level")
            
            with sr_col2:
                st.markdown(f"**🔴 Resistance**")
                st.markdown(f"### ${analysis['resistance']:.2f}")
                st.caption("Key selling zone")
            
            st.markdown("---")
            
            # ========== MARKET BREADTH & LIQUIDITY ==========
            st.markdown('<div class="section-header">🌊 MARKET BREADTH & LIQUIDITY</div>', unsafe_allow_html=True)
            
            breadth_col1, breadth_col2 = st.columns(2)
            
            with breadth_col1:
                st.markdown("#### Market Breadth (TRIN Proxy)")
                color = "bullish" if "Bullish" in breadth_sentiment else "bearish"
                st.markdown(f"<span class='{color}'>{breadth_sentiment}</span>", unsafe_allow_html=True)
                st.caption("S&P 500 volume analysis")
            
            with breadth_col2:
                st.markdown("#### Liquidity Index")
                color = "bullish" if liquidity_type == "bullish" else "bearish" if liquidity_type == "bearish" else "neutral"
                st.markdown(f"<span class='{color}'>{liquidity_status}</span>", unsafe_allow_html=True)
                st.caption("Market liquidity conditions")
            
            st.markdown("---")
            
            # ========== MACRO CONTEXT ==========
            st.markdown('<div class="section-header">🌍 MACRO & MARKET CONTEXT</div>', unsafe_allow_html=True)
            
            macro_col1, macro_col2, macro_col3 = st.columns(3)
            
            with macro_col1:
                st.markdown("#### VIX (Fear Index)")
                if isinstance(vix, (int, float)):
                    vix_sentiment, vix_type = get_vix_sentiment(vix)
                    color = "bullish" if vix_type == "bullish" else "bearish"
                    st.markdown(f"<span class='{color}'>{vix:.2f}</span>", unsafe_allow_html=True)
                    st.caption(vix_sentiment)
            
            with macro_col2:
                st.markdown("#### 10Y Treasury Yield")
                if isinstance(us10y_yield, (int, float)):
                    bond_sentiment, bond_type = get_us10y_sentiment(us10y_yield)
                    color = "bullish" if bond_type == "bullish" else "bearish"
                    st.markdown(f"<span class='{color}'>{us10y_yield:.2f}%</span>", unsafe_allow_html=True)
                    st.caption(bond_sentiment)
            
            with macro_col3:
                st.markdown("#### Dollar Index")
                if isinstance(dxy, (int, float)):
                    st.markdown(f"**{dxy:.2f}**")
                    st.caption("USD strength indicator")
            
            st.markdown("---")
            
            # ========== 2-YEAR PERFORMANCE CHART ==========
            st.markdown('<div class="section-header">📊 2-YEAR PERFORMANCE vs S&P 500</div>', unsafe_allow_html=True)
            
            comparison_fig = create_comparison_chart(ticker_input, df)
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
            else:
                st.warning("Could not generate comparison chart")
            
            st.markdown("---")
            
            # ========== PRICE CHART ==========
            st.markdown('<div class="section-header">📈 PRICE CHART</div>', unsafe_allow_html=True)
            
            try:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
                
                # Add SMA lines
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA50', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA200', line=dict(color='red')))
                
                fig.update_layout(
                    title=f'{ticker_input} - 2 Year Price History with Moving Averages',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=500,
                    hovermode='x unified',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate price chart: {e}")
            
            st.markdown("---")
            
            # ========== NEWS SECTION ==========
            st.markdown('<div class="section-header">📰 LATEST NEWS</div>', unsafe_allow_html=True)
            
            st.markdown(news)
            
            st.markdown("---")
            
            # ========== TRADING SIGNALS ==========
            st.markdown('<div class="section-header">🎯 AI TRADING SIGNALS</div>', unsafe_allow_html=True)
            
            signal_col1, signal_col2 = st.columns(2)
            
            with signal_col1:
                st.subheader("📊 Short-term (Traders)")
                rsi_sig = "BUY" if analysis['rsi'] < 30 else "SELL" if analysis['rsi'] > 70 else "HOLD"
                macd_sig = "BUY" if (analysis['macd'] > analysis['macd_signal']) else "SELL"
                
                signals = [rsi_sig, macd_sig]
                buy_count = signals.count("BUY")
                sell_count = signals.count("SELL")
                
                if buy_count > sell_count:
                    st.success("🟢 **BULLISH** - Consider Long Position")
                elif sell_count > buy_count:
                    st.error("🔴 **BEARISH** - Consider Short Position")
                else:
                    st.warning("🟡 **NEUTRAL** - Wait for Confirmation")
                
                st.caption(f"RSI: {rsi_sig} | MACD: {macd_sig}")
            
            with signal_col2:
                st.subheader("🎯 Long-term (Investors)")
                sma_sig = "BUY" if analysis['current_price'] > analysis['sma200'] else "SELL"
                trend_sig = "BUY" if "Bullish" in analysis['ma_cross_status'] else "SELL"
                
                signals_lt = [sma_sig, trend_sig]
                buy_count_lt = signals_lt.count("BUY")
                sell_count_lt = signals_lt.count("SELL")
                
                if buy_count_lt > sell_count_lt:
                    st.success("🟢 **ACCUMULATE** - Long-term Uptrend")
                elif sell_count_lt > buy_count_lt:
                    st.error("🔴 **REDUCE** - Long-term Downtrend")
                else:
                    st.warning("🟡 **HOLD** - Consolidation Phase")
                
                st.caption(f"SMA200: {sma_sig} | Trend: {trend_sig}")
            
            st.markdown("---")
            
            # ========== DISCLAIMER ==========
            st.info("⚠️ **DISCLAIMER**: This analysis is for educational purposes only. Not financial advice. Always conduct your own research and consult a financial advisor before trading.")
            
            st.success("✅ Analysis Complete - All systems operational")

else:
    st.info("👈 Enter a ticker symbol and click Analyze to begin comprehensive technical analysis")
