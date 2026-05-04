"""
Data engine — index constituent tickers and batch OHLCV ingestion.
"""
import io
import requests
import pandas as pd
import yfinance as yf

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*",
}

# Primary: plain CSV — no HTML parser required
_SP500_CSV  = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
# Secondary: Wikipedia HTML (needs lxml / html5lib / html.parser via bs4)
_SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_NIFTY500_CSV = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# Last-resort static list: top-50 S&P 500 names by market cap
_SP500_FALLBACK: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY",
    "AVGO", "TSLA", "JPM", "WMT", "V",    "UNH",  "XOM",  "MA",   "ORCL", "COST",
    "HD",   "PG",   "JNJ",  "ABBV", "BAC",  "NFLX", "KO",   "CRM",  "MRK",  "CVX",
    "AMD",  "PEP",  "TMO",  "ACN",  "LIN",  "MCD",  "ADBE", "IBM",  "GE",   "TXN",
    "PM",   "ISRG", "GS",   "AMGN", "DHR",  "BKNG", "RTX",  "NOW",  "CAT",  "INTU",
]

# Nifty 50 fallback used when the NSE CSV endpoint is unreachable
_NIFTY50_FALLBACK: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "TECHM.NS",
    "ADANIPORTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", "BRITANNIA.NS",
    "COALINDIA.NS", "HINDALCO.NS", "INDUSINDBK.NS", "M&M.NS", "TATACONSUM.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "BPCL.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS",
    "BAJAJ-AUTO.NS", "UPL.NS", "SHREECEM.NS", "TATAMOTORS.NS", "ADANIENT.NS",
]


def _clean_symbols(symbols: list) -> list[str]:
    """Strip whitespace and convert Yahoo Finance dot-notation to hyphens."""
    return [str(s).strip().replace(".", "-") for s in symbols if str(s).strip()]


def get_sp500_tickers() -> list[str]:
    """
    Return S&P 500 tickers, trying three sources in order:

    1. GitHub CSV  — plain CSV, no HTML parser required (most reliable)
    2. Wikipedia   — HTML scrape, needs lxml / html5lib / bs4
    3. Static list — top-50 hardcoded fallback, always works
    """
    # ── Source 1: GitHub open-data CSV (no HTML parser needed) ───────────────
    try:
        resp = requests.get(_SP500_CSV, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        df  = pd.read_csv(io.StringIO(resp.text))
        col = next((c for c in df.columns if c.strip().lower() in ("symbol", "ticker")), None)
        if col and not df.empty:
            return _clean_symbols(df[col].dropna().tolist())
    except Exception:
        pass

    # ── Source 2: Wikipedia HTML (requires an HTML parser package) ────────────
    try:
        html = requests.get(_SP500_WIKI, headers=_HEADERS, timeout=15).text
        for flavor in ("lxml", "html5lib", "html.parser"):
            try:
                for table in pd.read_html(io.StringIO(html), flavor=flavor):
                    if "Symbol" in table.columns:
                        return _clean_symbols(table["Symbol"].dropna().tolist())
            except Exception:
                continue
    except Exception:
        pass

    # ── Source 3: static fallback — always succeeds ───────────────────────────
    return _SP500_FALLBACK


def get_nifty500_tickers() -> list[str]:
    """
    Fetch Nifty 500 constituents from NSE archives CSV and append .NS suffix.
    Falls back to the Nifty 50 curated list when NSE is unreachable.
    """
    try:
        resp = requests.get(_NIFTY500_CSV, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return [f"{s.strip()}.NS" for s in df["Symbol"].tolist()]
    except Exception:
        return _NIFTY50_FALLBACK


def fetch_batch_ohlcv(
    tickers: list[str],
    period: str = "3mo",
    interval: str = "1d",
    chunk_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for all *tickers* in batched yf.download calls.

    Chunking avoids Yahoo Finance rate limits. threads=True parallelises
    within each chunk. Returns a dict keyed by ticker with flat OHLCV columns
    (no MultiIndex) so downstream code can access df["Close"] directly.
    """
    result: dict[str, pd.DataFrame] = {}

    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        try:
            raw = yf.download(
                chunk,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            continue

        if raw.empty:
            continue

        if len(chunk) == 1:
            # Single-ticker download may return flat or MultiIndex columns
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all")
            if not df.empty:
                result[chunk[0]] = df
        else:
            # Multi-ticker: raw[ticker] yields a DataFrame with standard columns
            for ticker in chunk:
                try:
                    df = raw[ticker].dropna(how="all")
                    if not df.empty:
                        result[ticker] = df
                except (KeyError, TypeError):
                    pass

    return result


# ── Single-ticker helpers (used by dashboard.py) ──────────────────────────────

def fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Return OHLCV DataFrame for a single *ticker*.

    Raises ValueError when no data is returned (bad ticker / delisted).
    Flattens MultiIndex columns produced by yfinance >= 0.2.x.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_multiple(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV sequentially for a small list of tickers.

    Returns a dict keyed by ticker; silently skips tickers with no data.
    For large universes use fetch_batch_ohlcv instead.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            result[ticker] = fetch_ohlcv(ticker, period=period, interval=interval)
        except ValueError:
            pass
    return result


def get_info(ticker: str) -> dict:
    """Return basic company metadata (name, sector, market cap)."""
    info = yf.Ticker(ticker).info
    return {
        "name":       info.get("longName", ticker),
        "sector":     info.get("sector", "N/A"),
        "industry":   info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "currency":   info.get("currency", "USD"),
    }
