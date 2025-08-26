"""
Ultimate AI Market Analyzer — Production-ready Streamlit app
- Multi-source, API-key-free price & history fetcher: NSE -> BSE (if scrip code) -> yfinance -> yahoo_fin -> Yahoo HTML scrape
- Lazy imports, safe_import, caching, retry/backoff, diagnostics
- No TA-Lib dependency (pandas_ta fallback)
- Designed to deploy cleanly on Streamlit Cloud
"""

import streamlit as st
import importlib
import traceback
import time
import random
from functools import wraps
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

st.set_page_config(page_title="Ultimate AI Market Analyzer", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:0.6rem}
.metric-card {border-radius:12px;padding:10px;background:#f8fafc;border:1px solid #e6eef6;margin-bottom:8px}
.section {border-radius:12px;padding:12px;background:#ffffffcc;border:1px solid #e6eef6}
.small-muted {color: #6b7280; font-size:12px}
</style>
""", unsafe_allow_html=True)

# -------------------- Utilities & diagnostics --------------------
def safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

if "diag" not in st.session_state:
    st.session_state.diag = {"last_exc": None, "providers": {}}

def flag_provider(name: str, ok: bool):
    st.session_state.diag["providers"][name] = ok

def record_exc():
    st.session_state.diag["last_exc"] = traceback.format_exc()

def retry_sync(max_retries=3, backoff_base=0.6):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = backoff_base
            last_exc = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    time.sleep(delay + random.random() * 0.1)
                    delay *= 2
            raise last_exc
        return wrapper
    return decorator

def fmt_money(x):
    try:
        if isinstance(x, (int, float)) and not np.isnan(x):
            if x > 1000:
                return f"₹{x:,.2f}"
            return f"₹{x:.2f}"
        return "—"
    except Exception:
        return "—"

# -------------------- Provider helpers --------------------
# NOTE: We avoid raising on import-time missing packages. Show diagnostics instead.
requests = safe_import("requests")
bs4 = safe_import("bs4")
yfinance = safe_import("yfinance")
yahoo_fin = safe_import("yahoo_fin.stock_info")  # may be None
pandas_ta = safe_import("pandas_ta")
plotly = safe_import("plotly.graph_objects")

flag_provider("requests", requests is not None)
flag_provider("bs4", bs4 is not None)
flag_provider("yfinance", yfinance is not None)
flag_provider("yahoo_fin", yahoo_fin is not None)
flag_provider("pandas_ta", pandas_ta is not None)
flag_provider("plotly", plotly is not None)

# -------------------- NSE fetcher (official endpoint, polite session) --------------------
@retry_sync(max_retries=2, backoff_base=0.8)
def get_price_nse(symbol: str) -> float:
    """
    Try to fetch price from NSE official endpoint (no API key).
    Returns float or np.nan.
    """
    if not requests or not bs4:
        flag_provider("nse", False)
        return float("nan")
    try:
        s = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        # initial request to get cookies
        s.get("https://www.nseindia.com", headers=headers, timeout=8)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}"
        resp = s.get(url, headers={**headers, "Referer": "https://www.nseindia.com"}, timeout=8)
        data = resp.json()
        # path may vary; best-effort
        last = data.get("priceInfo", {}).get("lastPrice", None)
        if last is None:
            # sometimes data nested in 'data'
            last = data.get("data", {}).get("lastPrice", None)
        if last is not None:
            flag_provider("nse", True)
            return float(last)
    except Exception:
        record_exc()
        flag_provider("nse", False)
    return float("nan")

# -------------------- BSE fetcher (requires numeric scrip code) --------------------
@retry_sync(max_retries=2, backoff_base=0.8)
def get_price_bse(scrip_code: str) -> float:
    """
    Fetch latest price from BSE API endpoint when scrip_code (numeric) is known.
    Example: Tata Technologies BSE code maybe '543527' (user-supplied mapping recommended).
    Returns float or np.nan.
    """
    if not requests:
        flag_provider("bse", False)
        return float("nan")
    try:
        # api endpoint expects scripcode param
        url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={scrip_code}&flag=0&fromdate=&todate=&seriesid="
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        data = resp.json()
        # data is often a list of datapoints; take last non-empty numeric close
        if isinstance(data, list) and len(data) > 0:
            last = data[-1]
            # some responses are lists of numbers [date, open, high, low, close, vol]
            if isinstance(last, list) and len(last) >= 5:
                close = last[4] if last[4] is not None else last[-2]
                flag_provider("bse", True)
                return float(close)
        # fallback: sometimes API returns dict with LastPrice
        if isinstance(data, dict) and "LastPrice" in data:
            flag_provider("bse", True)
            return float(data["LastPrice"])
    except Exception:
        record_exc()
        flag_provider("bse", False)
    return float("nan")

# -------------------- Yahoo fallback (yfinance) --------------------
@retry_sync(max_retries=2, backoff_base=0.6)
def get_price_yfinance(symbol: str) -> float:
    if not yfinance:
        flag_provider("yfinance_price", False)
        return float("nan")
    try:
        ticker = yfinance.Ticker(symbol + ".NS")
        hist = ticker.history(period="1d", interval="1d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            flag_provider("yfinance_price", True)
            return price
    except Exception:
        record_exc()
        flag_provider("yfinance_price", False)
    return float("nan")

# -------------------- yahoo_fin fallback (live price scraper) --------------------
@retry_sync(max_retries=2, backoff_base=0.6)
def get_price_yahoo_fin(symbol: str) -> float:
    if not yahoo_fin:
        flag_provider("yahoo_fin_price", False)
        return float("nan")
    try:
        price = yahoo_fin.get_live_price(symbol + ".NS")
        flag_provider("yahoo_fin_price", True)
        return float(price)
    except Exception:
        record_exc()
        flag_provider("yahoo_fin_price", False)
    return float("nan")

# -------------------- direct Yahoo html scrape (last resort) --------------------
@retry_sync(max_retries=2, backoff_base=0.6)
def get_price_yahoo_scrape(symbol: str) -> float:
    if not requests or not bs4:
        flag_provider("yahoo_scrape", False)
        return float("nan")
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}.NS"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        el = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        if el and el.text:
            flag_provider("yahoo_scrape", True)
            return float(el.text.replace(",", ""))
    except Exception:
        record_exc()
        flag_provider("yahoo_scrape", False)
    return float("nan")

# -------------------- Provider selection orchestration (auto resource changer) --------------------
def get_best_price(symbol: str, bse_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Try providers in preferred order; return dict with:
    {'price': float, 'provider': 'nse'|'bse'|'yfinance'|..., 'timings': {...}}
    The function measures elapsed time for successful calls and picks fastest success.
    """
    providers = []
    # 1) NSE
    providers.append(("nse", lambda: get_price_nse(symbol)))
    # 2) BSE if scrip code passed
    if bse_code:
        providers.append(("bse", lambda: get_price_bse(bse_code)))
    # 3) yfinance
    providers.append(("yfinance", lambda: get_price_yfinance(symbol)))
    # 4) yahoo_fin
    providers.append(("yahoo_fin", lambda: get_price_yahoo_fin(symbol)))
    # 5) yahoo_scrape
    providers.append(("yahoo_scrape", lambda: get_price_yahoo_scrape(symbol)))

    results = {}
    best = {"price": float("nan"), "provider": None, "elapsed": float("inf")}
    for name, func in providers:
        start = time.time()
        try:
            price = func()
            elapsed = time.time() - start
            results[name] = {"price": price, "elapsed": elapsed}
            # Accept if valid numeric
            if not np.isnan(price):
                # pick the first successful provider (prefer higher-ranked) but record elapsed
                if elapsed < best["elapsed"]:
                    best = {"price": price, "provider": name, "elapsed": elapsed}
                    # We do not break immediately — we allow later providers to be faster (rare)
        except Exception:
            results[name] = {"price": float("nan"), "elapsed": None}
    return {"price": best["price"], "provider": best["provider"], "timings": results}

# -------------------- Historical chain (NSE -> yfinance -> yahoo_fin -> scrape) --------------------
@st.cache_data(ttl=60*10)
def get_history(symbol: str, period: str = "6mo", interval: str = "1d", bse_code: Optional[str] = None) -> pd.DataFrame:
    # 1) try NSE historical if available
    try:
        # NSE sometimes provides series endpoints; we attempt a lightweight yfinance-first approach for reliability
        if yfinance:
            df = yfinance.Ticker(symbol + ".NS").history(period=period, interval=interval)
            if not df.empty:
                return df.rename(columns=lambda c: c.capitalize())
    except Exception:
        record_exc()
    # 2) try yahoo_fin get_data
    try:
        if yahoo_fin:
            df2 = yahoo_fin.get_data(symbol + ".NS")
            if not df2.empty:
                df2 = df2.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
                return df2
    except Exception:
        record_exc()
    # 3) fallback: empty df
    return pd.DataFrame()

# -------------------- Indicator computation (pandas_ta fallback) --------------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Try talib if available (rare on cloud)
    talib = safe_import("talib")
    try:
        if talib:
            ema50 = talib.EMA(close.values, timeperiod=50)
            ema200 = talib.EMA(close.values, timeperiod=200)
            rsi = float(talib.RSI(close.values, timeperiod=14)[-1])
            macd, macds, _ = talib.MACD(close.values)
            macd = float(macd[-1]) if len(macd) else np.nan
            macds = float(macds[-1]) if len(macds) else np.nan
            return {"EMA50": pd.Series(ema50, index=df.index), "EMA200": pd.Series(ema200, index=df.index),
                    "RSI": rsi, "MACD": macd, "MACD_signal": macds, "Hammer": 0, "Doji": 0, "Engulf": 0}
    except Exception:
        record_exc()

    # Fallback using pandas_ta or pandas ewm
    try:
        ind = {}
        ind["EMA50"] = close.ewm(span=50, adjust=False).mean()
        ind["EMA200"] = close.ewm(span=200, adjust=False).mean()
        if pandas_ta:
            rsi_series = pandas_ta.rsi(close, length=14)
            ind["RSI"] = float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan
            macd_df = pandas_ta.macd(close)
            cols = list(macd_df.columns) if isinstance(macd_df, pd.DataFrame) else []
            if len(cols) >= 2:
                ind["MACD"] = float(macd_df[cols[0]].iloc[-1])
                ind["MACD_signal"] = float(macd_df[cols[1]].iloc[-1])
            else:
                ind["MACD"] = np.nan
                ind["MACD_signal"] = np.nan
        else:
            # manual RSI approx
            delta = close.diff().dropna()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.ewm(com=13, adjust=False).mean()
            roll_down = down.ewm(com=13, adjust=False).mean()
            rs = roll_up / roll_down
            ind["RSI"] = float(100 - (100 / (1 + rs.iloc[-1]))) if not rs.empty else np.nan
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            ind["MACD"] = float(macd_line.iloc[-1])
            ind["MACD_signal"] = float(macd_signal.iloc[-1])

        # candle heuristics
        last_body = abs(close.iloc[-1] - open_.iloc[-1])
        candle_range = (high.iloc[-1] - low.iloc[-1]) if (high.iloc[-1] - low.iloc[-1]) != 0 else 1
        ind["Hammer"] = 1 if (last_body < candle_range * 0.25 and (close.iloc[-1] - low.iloc[-1]) < last_body * 1.5) else 0
        ind["Doji"] = 1 if last_body < candle_range * 0.1 else 0
        ind["Engulf"] = 0
        return ind
    except Exception:
        record_exc()
        return {}

# -------------------- Composite scoring & recommendation --------------------
def composite_score(ind: dict) -> int:
    score = 50
    try:
        rsi = ind.get("RSI", np.nan)
        if not np.isnan(rsi):
            score += 15 if rsi < 30 else (-15 if rsi > 70 else 0)
        ema50 = ind.get("EMA50")
        ema200 = ind.get("EMA200")
        if hasattr(ema50, "iloc") and hasattr(ema200, "iloc") and len(ema50) > 0 and len(ema200) > 0:
            score += 20 if ema50.iloc[-1] > ema200.iloc[-1] else -5
        macd = ind.get("MACD", np.nan)
        macd_sig = ind.get("MACD_signal", np.nan)
        if (not np.isnan(macd)) and (not np.isnan(macd_sig)):
            score += 15 if macd > macd_sig else -5
        if ind.get("Hammer", 0) != 0: score += 5
        if ind.get("Doji", 0) != 0: score += 2
        if ind.get("Engulf", 0) != 0: score += 5
    except Exception:
        pass
    return int(max(0, min(100, score)))

def recommendation_text(score: int) -> str:
    return "Buy" if score >= 70 else ("Hold" if score >= 40 else "Sell")

def heat_badge(score: int) -> str:
    return "🟢 Strong (Buy)" if score >= 70 else ("🟡 Neutral (Hold)" if score >= 40 else "🔴 Weak (Sell)")

# -------------------- Plotting --------------------
def plot_candles(df: pd.DataFrame, indicators: dict = None, show_ind: bool = True):
    go = plotly
    if not go:
        st.warning("Plotly not installed; install plotly to view charts.")
        return None
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(height=360, xaxis_rangeslider_visible=False)
        return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if show_ind and indicators:
        if "EMA50" in indicators and hasattr(indicators["EMA50"], "index"):
            fig.add_trace(go.Scatter(x=df.index, y=indicators["EMA50"], mode="lines", name="EMA50"))
        if "EMA200" in indicators and hasattr(indicators["EMA200"], "index"):
            fig.add_trace(go.Scatter(x=df.index, y=indicators["EMA200"], mode="lines", name="EMA200"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=26, b=8))
    return fig

# -------------------- SIP calculator --------------------
def sip_future_value(monthly: float, rate_annual: float, months: int) -> float:
    r = (rate_annual / 12) / 100.0
    if r == 0:
        return monthly * months
    return monthly * ((pow(1 + r, months) - 1) / r) * (1 + r)

# -------------------- News fetcher (RSS) --------------------
@st.cache_data(ttl=60*30)
@retry_sync(max_retries=2, backoff_base=0.7)
def fetch_rss(url: str, max_items: int = 5):
    if not requests or not bs4:
        return []
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:max_items]
        out = []
        for it in items:
            out.append({"title": it.title.text if it.title else "", "link": it.link.text if it.link else "", "pubDate": it.pubDate.text if it.pubDate else ""})
        return out
    except Exception:
        record_exc()
        return []

# -------------------- UI --------------------
st.title("Ultimate AI Market Analyzer — Production-ready (NSE/BSE aware)")
st.caption("Multi-source, API-key-free. Automatic provider selection. Designed to deploy cleanly on Streamlit Cloud.")

with st.sidebar:
    st.header("Settings")
    refresh_secs = st.slider("Auto-refresh (seconds)", 10, 180, 45, 5)
    lookback = st.selectbox("History lookback", ["3mo", "6mo", "1y", "3y"], index=1)
    timeframe = st.selectbox("Chart timeframe", ["1d", "5d", "1wk", "1mo"], index=0)
    show_ind = st.checkbox("Show EMA/MACD/RSI on charts", value=True)
    st.divider()
    st.header("Diagnostics")
    if st.button("Show diagnostics"):
        st.json(st.session_state.diag)
    st.caption("Providers: NSE/BSE -> yfinance -> yahoo_fin -> Yahoo scrape.")

col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter NSE symbol (no .NS), e.g., TATATECH", value="TATATECH").strip().upper()
    bse_code = st.text_input("Optional BSE scrip code (numeric) e.g., 543527", value="").strip()
    if st.button("Fetch latest & show chart"):
        with st.spinner("Fetching price and history..."):
            # fetch best price
            out = get_best_price(symbol, bse_code if bse_code else None)
            price = out["price"]
            provider = out["provider"]
            if np.isnan(price):
                st.warning("Price not available from current providers. Check diagnostics in sidebar.")
            else:
                st.metric(f"{symbol} Price ({provider})", fmt_money(price))
            # fetch history
            hist = get_history(symbol, period=lookback, interval="1d", bse_code=bse_code if bse_code else None)
            if hist is None or hist.empty:
                st.warning("Historical data not available. Try again or check diagnostics.")
            else:
                ind = compute_indicators(hist)
                score = composite_score(ind)
                rec = recommendation_text(score)
                badge = heat_badge(score)
                st.markdown(f"**Composite score:** {score}/100 — {rec} {badge}")
                fig = plot_candles(hist, ind, show_ind=show_ind)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if st.checkbox("Show last 10 rows"):
                    st.dataframe(hist[["Open", "High", "Low", "Close"]].tail(10))

with col2:
    st.subheader("Quick tools")
    st.markdown("**SIP Calculator**")
    m = st.number_input("Monthly SIP (₹)", value=5000, step=500)
    r = st.number_input("Expected annual return (%)", value=12.0, step=0.5)
    months = st.number_input("Duration (months)", value=60, min_value=1)
    fv = sip_future_value(m, r, months)
    st.markdown(f"**Future Value:** {fmt_money(fv)}")
    st.markdown("---")
    st.subheader("News")
    news = fetch_rss("https://finance.yahoo.com/rss/topstories", max_items=5)
    if news:
        for n in news:
            st.write(f"- [{n['title']}]({n['link']}) — {n['pubDate']}")
    else:
        st.info("No RSS items (maybe requests/bs4 not installed?). Check diagnostics.")

st.markdown("---")
st.caption("Notes: If you want BSE lookup automation (symbol -> scrip code), we can add a mapping/scraper to resolve scrip codes automatically. This was left optional to avoid fragile scraping of search endpoints. To enable additional concurrency (aiohttp) or TA-Lib, update requirements.txt and I will provide a Dockerfile for deterministic builds.")
# Paste your full AI Market Analyzer Streamlit code here