# ============================================================
# Ultimate AI Market Analyzer — Pure Python Version
# ============================================================

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import requests
import feedparser

# Optional TA-Lib fallback
try:
    import talib
    TA_LIB = True
except Exception:
    import pandas_ta as pta
    TA_LIB = False

# -------------------- Stock Data --------------------
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
                        ind
