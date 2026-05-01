"""
Data ingestion layer — fetches OHLCV data from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
from typing import Union


def fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Return OHLCV DataFrame for *ticker*.

    Raises ValueError when no data is returned (bad ticker / delisted).
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    # yfinance >= 0.2.x returns MultiIndex columns ("Close", "AAPL") for single-ticker
    # downloads. Flatten to simple string labels so downstream code can do df["Close"].
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_multiple(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for a list of tickers.

    Returns a dict keyed by ticker; skips tickers that returned no data.
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
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency", "USD"),
    }
