"""
Quant engine — computes technical indicators and screening signals.

Indicators are added as new columns on the input DataFrame (non-destructive copy).
"""
import numpy as np
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical indicators to *df* and return a new DataFrame."""
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # Trend
    df["EMA_20"]  = ta.trend.ema_indicator(close, window=20)
    df["EMA_50"]  = ta.trend.ema_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["MACD"]        = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    df["MACD_Hist"]   = ta.trend.macd_diff(close)

    # Momentum
    df["RSI"]  = ta.momentum.rsi(close, window=14)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Mid"]    = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()
    df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Volume
    df["OBV"]  = ta.volume.on_balance_volume(close, volume)
    df["VWAP"] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=14)

    return df


def screen(df: pd.DataFrame) -> dict[str, bool]:
    """
    Run a set of boolean screening checks on the latest row of *df*.

    Returns a dict of {signal_name: bool}.  df must have indicator columns
    (call add_indicators first).
    """
    last = df.iloc[-1]
    close = float(last["Close"])

    return {
        "RSI Oversold (<30)":           float(last["RSI"]) < 30,
        "RSI Overbought (>70)":         float(last["RSI"]) > 70,
        "Price > EMA 20":               close > float(last["EMA_20"]),
        "Price > EMA 50":               close > float(last["EMA_50"]),
        "Golden Cross (EMA20>EMA50)":   float(last["EMA_20"]) > float(last["EMA_50"]),
        "Price > SMA 200":              close > float(last["SMA_200"]),
        "MACD Bullish Crossover":       float(last["MACD"]) > float(last["MACD_Signal"]),
        "BB Squeeze (Width<0.1)":       float(last["BB_Width"]) < 0.1,
        "Price Near BB Lower (<2%)":    abs(close - float(last["BB_Lower"])) / close < 0.02,
        "Stoch Oversold (K<20)":        float(last["Stoch_K"]) < 20,
        "Stoch Overbought (K>80)":      float(last["Stoch_K"]) > 80,
    }


def analyze_ticker(data: pd.DataFrame) -> dict[str, object]:
    """
    Compute RSI-14 and Volume Spike signals from raw OHLCV *data*.

    Both indicators are computed with fully vectorized pandas/numpy operations —
    no Python-level loops over rows.

    Returns a dict with:
      rsi            float  — current RSI (last bar)
      rsi_signal     str    — "oversold" | "overbought" | "neutral"
      volume_spike   bool   — True when last bar volume > 2× its 20-day rolling mean
      volume_ratio   float  — last bar volume / 20-day average volume
      signals        list   — human-readable list of active signal strings
    """
    close  = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    # ── RSI-14 (Wilder / EMA smoothing, identical to TradingView) ────────────
    delta = close.diff()                          # price change series
    gain  = delta.clip(lower=0)                   # keep positive moves
    loss  = (-delta).clip(lower=0)                # keep magnitude of drops

    # First average: simple mean over the seed window; rest via Wilder EMA (α = 1/14)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)  # avoid ÷0; NaN propagates cleanly
    rsi = (100 - (100 / (1 + rs))).round(2)

    current_rsi = float(rsi.iloc[-1])
    if current_rsi < 30:
        rsi_signal = "oversold"
    elif current_rsi > 70:
        rsi_signal = "overbought"
    else:
        rsi_signal = "neutral"

    # ── Volume Spike — current bar > 200 % of rolling 20-day mean ────────────
    # min_periods=1 so the ratio is defined even near the start of the series;
    # the last bar is excluded from its own average (closed="left") to avoid
    # look-ahead bias on intraday data.
    vol_avg   = volume.shift(1).rolling(window=20, min_periods=1).mean()
    vol_ratio = (volume / vol_avg).round(4)       # vectorized division across all rows

    current_ratio = float(vol_ratio.iloc[-1])
    volume_spike  = current_ratio >= 2.0

    # ── Assemble output ───────────────────────────────────────────────────────
    active_signals: list[str] = []
    if rsi_signal == "oversold":
        active_signals.append(f"RSI oversold ({current_rsi:.1f})")
    elif rsi_signal == "overbought":
        active_signals.append(f"RSI overbought ({current_rsi:.1f})")
    if volume_spike:
        active_signals.append(f"Volume spike ({current_ratio:.1f}× avg)")

    return {
        "rsi":          current_rsi,
        "rsi_signal":   rsi_signal,
        "volume_spike": volume_spike,
        "volume_ratio": current_ratio,
        "signals":      active_signals,
    }


def latest_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Return key indicator values from the last row as a flat dict."""
    last = df.iloc[-1]
    return {
        "Close":    round(float(last["Close"]), 2),
        "RSI":      round(float(last["RSI"]), 1),
        "MACD":     round(float(last["MACD"]), 4),
        "ATR":      round(float(last["ATR"]), 2),
        "BB_Width": round(float(last["BB_Width"]), 4),
        "OBV":      int(last["OBV"]),
    }
