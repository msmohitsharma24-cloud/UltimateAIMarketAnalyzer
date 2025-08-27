# ============================================================
# Ultimate AI Market Analyzer — Streamlit/Replit-ready
# ============================================================

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import requests
import feedparser
import streamlit as st

# Optional TA-Lib fallback
try:
    import talib
    TA_LIB = True
except Exception:
    import pandas_ta as pta
    TA_LIB = False

# -------------------- Helper Functions --------------------
def get_live_stock(symbol: str) -> float:
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10).json()
        return float(resp['priceInfo']['lastPrice'])
    except Exception:
        return np.nan

def get_historical_stock(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        import yfinance as yf
        yf_symbol = symbol.upper() + ".NS" if not symbol.upper().endswith(".NS") else symbol.upper()
        df = yf.Ticker(yf_symbol).history(period=period)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception:
        return pd.DataFrame()

# -------------------- Technical Indicators --------------------
def compute_indicators(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']

    ind = {}

    try:
        if TA_LIB:
            try:
                ind["EMA50"] = talib.EMA(close.values, 50)
                ind["EMA200"] = talib.EMA(close.values, 200)
                ind["RSI"] = talib.RSI(close.values, 14)

                try:
                    macd, macds, _ = talib.MACD(close.values)
                    ind["MACD"] = macd
                    ind["MACD_signal"] = macds
                except Exception:
                    ind["MACD"] = pd.Series(np.zeros(len(df)))
                    ind["MACD_signal"] = pd.Series(np.zeros(len(df)))

            except Exception:
                try:
                    ind["EMA50"] = pta.ema(close, length=50)
                    ind["EMA200"] = pta.ema(close, length=200)
                    ind["RSI"] = pta.rsi(close, length=14)

                    try:
                        macd_df = pta.macd(close)
                        ind["MACD"] = macd_df["MACD_12_26_9"]
                        ind["MACD_signal"] = macd_df["MACDs_12_26_9"]
                    except Exception:
                        ind["MACD"] = pd.Series(np.zeros(len(df)))
                        ind["MACD_signal"] = pd.Series(np.zeros(len(df)))
                except Exception:
                    ind["EMA50"] = pd.Series(np.zeros(len(df)))
                    ind["EMA200"] = pd.Series(np.zeros(len(df)))
                    ind["RSI"] = pd.Series(np.zeros(len(df)))
                    ind["MACD"] = pd.Series(np.zeros(len(df)))
                    ind["MACD_signal"] = pd.Series(np.zeros(len(df)))

            # Candlestick patterns
            try:
                ind["Hammer"] = talib.CDLHAMMER(open_.values, high.values, low.values, close.values)
            except Exception:
                ind["Hammer"] = pd.Series(np.zeros(len(df)))

            try:
                ind["Doji"] = talib.CDLDOJI(open_.values, high.values, low.values, close.values)
            except Exception:
                ind["Doji"] = pd.Series(np.zeros(len(df)))

            try:
                ind["Engulf"] = talib.CDLENGULFING(open_.values, high.values, low.values, close.values)
            except Exception:
                ind["Engulf"] = pd.Series(np.zeros(len(df)))

        else:
            # pandas_ta fallback
            try:
                ind["EMA50"] = pta.ema(close, length=50)
                ind["EMA200"] = pta.ema(close, length=200)
                ind["RSI"] = pta.rsi(close, length=14)
            except Exception:
                ind["EMA50"] = pd.Series(np.zeros(len(df)))
                ind["EMA200"] = pd.Series(np.zeros(len(df)))
                ind["RSI"] = pd.Series(np.zeros(len(df)))

            try:
                macd_df = pta.macd(close)
                ind["MACD"] = macd_df["MACD_12_26_9"]
                ind["MACD_signal"] = macd_df["MACDs_12_26_9"]
            except Exception:
                ind["MACD"] = pd.Series(np.zeros(len(df)))
                ind["MACD_signal"] = pd.Series(np.zeros(len(df)))

            # Candlestick fallback
            ind["Hammer"] = pd.Series(np.zeros(len(df)))
            ind["Doji"] = pd.Series(np.zeros(len(df)))
            ind["Engulf"] = pd.Series(np.zeros(len(df)))

    except Exception:
        ind = {}

    return ind

# -------------------- Composite Score --------------------
def composite_score(ind: dict) -> int:
    score = 50
    try:
        rsi = ind.get("RSI")
        if rsi is not None and len(rsi):
            last_rsi = float(pd.Series(rsi).dropna().iloc[-1])
            if last_rsi < 30:
                score += 15
            elif last_rsi > 70:
                score -= 15

        ema50, ema200 = ind.get("EMA50"), ind.get("EMA200")
        if ema50 is not None and ema200 is not None:
            if ema50[-1] > ema200[-1]:
                score += 20
            else:
                score -= 5

        macd, macds = ind.get("MACD"), ind.get("MACD_signal")
        if macd is not None and macds is not None:
            if macd[-1] > macds[-1]:
                score += 15
            else:
                score -= 5

        for pat in ["Hammer", "Doji", "Engulf"]:
            val = ind.get(pat)
            if val is not None and val[-1] != 0:
                score += 5 if pat != "Doji" else 2
    except Exception:
        pass

    return max(0, min(100, int(score)))

def recommendation(score: int) -> str:
    return "Buy" if score >= 70 else ("Hold" if score >= 40 else "Sell")

# -------------------- Candlestick Plotting --------------------
def plot_candlestick(df: pd.DataFrame, ind: dict = None) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"))

    if ind:
        if "EMA50" in ind:
            fig.add_trace(go.Scatter(x=df.index, y=ind["EMA50"], name="EMA50"))
        if "EMA200" in ind:
            fig.add_trace(go.Scatter(x=df.index, y=ind["EMA200"], name="EMA200"))

    fig.update_layout(height=450, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig

# -------------------- SIP Calculator --------------------
def sip_future_value(monthly: float, rate_annual: float, months: int) -> float:
    r = (rate_annual / 12) / 100.0
    if r == 0:
        return monthly * months
    return monthly * ((pow(1 + r, months) - 1) / r) * (1 + r)

# -------------------- News Aggregator --------------------
def fetch_news_rss(url: str, max_items: int = 5) -> list:
    news_list = []
    try:
        feed = feedparser.parse(url)
        for item in feed.entries[:max_items]:
            news_list.append({
                "title": item.title,
                "link": item.link,
                "pubDate": item.published if 'published' in item else ''
            })
    except Exception:
        pass
    return news_list

# -------------------- Streamlit UI --------------------
st.title("Ultimate AI Market Analyzer — Replit Edition")

symbol = st.text_input("Enter NSE Stock Symbol (e.g., TCS, INFY):", "TCS")
if st.button("Analyze"):
    st.info("Fetching data...")
    df = get_historical_stock(symbol)
    if df.empty:
        st.error("Failed to fetch stock data.")
    else:
        ind = compute_indicators(df)
        score = composite_score(ind)
        rec = recommendation(score)

        st.metric(label="Composite Score", value=score)
        st.metric(label="Recommendation", value=rec)

        st.plotly_chart(plot_candlestick(df, ind), use_container_width=True)

        st.subheader("Latest News")
        news = fetch_news_rss(f"https://www.moneycontrol.com/rss/MCtopnews.xml")
        for n in news:
            st.markdown(f"[{n['title']}]({n['link']}) - {n['pubDate']}")