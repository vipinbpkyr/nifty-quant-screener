"""
Quant logic — Screener and Backtester for the Value + Momentum + Volume framework.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


# ── shared vectorized indicator helpers ──────────────────────────────────────
# Module-level so both Screener and Backtester use identical math.

def _wilder_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """Full Wilder RSI series via EWM (α = 1/window, adjust=False)."""
    alpha = 1.0 / window
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _vol_sma_ratio_series(volume: pd.Series, window: int = 20) -> pd.Series:
    """volume / rolling SMA. shift(1) prevents the current bar from inflating its own baseline."""
    sma = volume.shift(1).rolling(window=window, min_periods=1).mean()
    return volume / sma


def _sma_series(close: pd.Series, window: int = 200) -> pd.Series:
    """Simple Moving Average — NaN until *window* bars are available."""
    return close.rolling(window=window, min_periods=window).mean()


def _vol_trend_series(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """True when short-window avg volume exceeds long-window avg volume."""
    vol_short = volume.rolling(window=short, min_periods=short).mean()
    vol_long  = volume.rolling(window=long,  min_periods=long).mean()
    return vol_short > vol_long


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class ScreenerConfig:
    rsi_window:       int   = 14
    vol_sma_window:   int   = 20
    sma_trend_window: int   = 200   # price > SMA-N to require long-term uptrend
    vol_trend_short:  int   = 5     # short window for volume trend check
    max_pe:           float = 20.0
    min_pe:           float = 0.0   # exclude negative / zero PE (unprofitable)
    min_rsi:          float = 50.0
    vol_spike_mult:   float = 2.0
    pe_workers:       int   = 8


# ── backtest ──────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    ticker:         str
    n_signals:      int
    win_rate:       float      # 0.0 – 1.0
    avg_fwd_return: float      # mean forward return across all signal bars
    max_drawdown:   float      # most negative drawdown fraction, e.g. -0.12
    equity_curve:   pd.Series  # cumulative product indexed by signal dates


class Backtester:
    """
    Vectorized backtester for the RSI + Volume signal.

    For every historical bar where RSI > min_rsi AND volume ≥ vol_spike_mult × SMA-20,
    records the *fwd_days*-day forward close-to-close return.

    Signal bars within the last *fwd_days* rows are excluded — no realised outcome yet.
    All indicator math is identical to the Screener (same shared helpers), so a signal
    that fires in the live screen would also have fired historically under these rules.
    """

    def __init__(self, config: ScreenerConfig | None = None) -> None:
        self.cfg = config or ScreenerConfig()

    def _signal_returns(self, df: pd.DataFrame, fwd_days: int) -> pd.Series | None:
        """
        Detect all past signal bars in *df* and return their fwd_days-forward returns.

        Returns None when there are too few bars or no signal ever fired.
        Signal conditions mirror Screener.filter_stocks() exactly.
        """
        min_bars = max(
            self.cfg.rsi_window + self.cfg.vol_sma_window,
            self.cfg.sma_trend_window,
            self.cfg.vol_trend_short,
        ) + fwd_days
        if len(df) < min_bars:
            return None

        close  = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        rsi_s       = _wilder_rsi_series(close, self.cfg.rsi_window)
        vol_s       = _vol_sma_ratio_series(volume, self.cfg.vol_sma_window)
        sma200_s    = _sma_series(close, self.cfg.sma_trend_window)
        vol_trend_s = _vol_trend_series(volume, self.cfg.vol_trend_short, self.cfg.vol_sma_window)

        signal = (
            (rsi_s > self.cfg.min_rsi) &
            (vol_s >= self.cfg.vol_spike_mult) &
            (close > sma200_s) &
            vol_trend_s
        )
        # Mask out the last fwd_days bars: forward close is not yet available
        signal.iloc[-fwd_days:] = False

        idx = signal[signal].index
        if len(idx) == 0:
            return None

        fwd_ret = close.shift(-fwd_days) / close - 1
        trades  = fwd_ret.loc[idx].dropna()
        return trades if not trades.empty else None

    def run(self, ticker: str, df: pd.DataFrame, fwd_days: int = 5) -> BacktestResult | None:
        """Single-ticker backtest. Returns None when no signals fired."""
        trades = self._signal_returns(df, fwd_days)
        if trades is None:
            return None

        equity   = (1 + trades).cumprod()
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max

        return BacktestResult(
            ticker=ticker,
            n_signals=len(trades),
            win_rate=float((trades > 0).mean()),
            avg_fwd_return=float(trades.mean()),
            max_drawdown=float(drawdown.min()),
            equity_curve=equity,
        )

    def run_batch(
        self,
        data: dict[str, pd.DataFrame],
        fwd_days: int = 5,
    ) -> pd.DataFrame:
        """Per-ticker backtest summary sorted by Win Rate descending."""
        rows = []
        for ticker, df in data.items():
            r = self.run(ticker, df, fwd_days=fwd_days)
            if r is None:
                continue
            rows.append({
                "Ticker":         r.ticker,
                "Signals":        r.n_signals,
                "Win Rate %":     round(r.win_rate * 100, 1),
                "Avg 5D Ret %":   round(r.avg_fwd_return * 100, 2),
                "Max Drawdown %": round(abs(r.max_drawdown) * 100, 2),
            })
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values("Win Rate %", ascending=False)
            .reset_index(drop=True)
        )

    def aggregate_stats(
        self,
        data: dict[str, pd.DataFrame],
        fwd_days: int = 5,
    ) -> dict:
        """
        Pool all signal trades across every ticker into one chronological stream
        and return portfolio-level statistics.

        Multiple signals on the same date are averaged into a daily basket before
        compounding — this avoids double-counting correlated moves on the same day
        and gives a realistic one-unit-of-capital equity curve.

        Returns
        -------
        dict with keys:
          total_trades, win_rate, avg_fwd_return, max_drawdown  (display-ready numbers)
          equity_curve, drawdown_curve                          (pd.Series for charts)
        """
        all_rets: list[pd.Series] = []
        for df in data.values():
            trades = self._signal_returns(df, fwd_days)
            if trades is not None:
                all_rets.append(trades)

        if not all_rets:
            return {}

        combined = pd.concat(all_rets).sort_index()

        # Equal-weight daily basket: one capital unit deployed per day with a signal
        daily    = combined.groupby(level=0).mean()
        equity   = (1 + daily).cumprod()
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max

        # Prepend a 1.0 origin so the chart starts at a clean baseline
        t0       = equity.index[0] - pd.Timedelta(days=1)
        equity   = pd.concat([pd.Series([1.0], index=[t0]), equity])
        drawdown = pd.concat([pd.Series([0.0], index=[t0]), drawdown])

        return {
            "total_trades":   len(combined),
            "win_rate":       round(float((combined > 0).mean()) * 100, 1),
            "avg_fwd_return": round(float(combined.mean()) * 100, 2),
            "max_drawdown":   round(float(drawdown.min()) * 100, 2),   # e.g. -8.3
            "equity_curve":   equity,
            "drawdown_curve": drawdown,
        }


# ── screener ──────────────────────────────────────────────────────────────────

class Screener:
    """
    Three-factor screener: Value (PE < max_pe), Momentum (RSI > min_rsi),
    Volume (current vol ≥ vol_spike_mult × 20-day SMA).

    filter_stocks(data) runs in three passes:
      Pass 1 — vectorized RSI + Volume scan across all tickers  (fast)
      Pass 2 — concurrent PE fetch for Pass-1 survivors only    (minimises API calls)
      Pass 3 — PE value filter + Quant Score ranking
    """

    def __init__(self, config: ScreenerConfig | None = None) -> None:
        self.cfg   = config or ScreenerConfig()
        self.stats: dict = {}

    def _rsi(self, close: pd.Series) -> float:
        return float(_wilder_rsi_series(close, self.cfg.rsi_window).iloc[-1])

    def _vol_ratio(self, volume: pd.Series) -> float:
        return float(_vol_sma_ratio_series(volume, self.cfg.vol_sma_window).iloc[-1])

    @staticmethod
    def _fetch_pe(ticker: str) -> tuple[str, float | None]:
        try:
            info = yf.Ticker(ticker).info
            pe   = info.get("trailingPE") or info.get("forwardPE")
            return ticker, float(pe) if pe else None
        except Exception:
            return ticker, None

    def fetch_pe_batch(self, tickers: list[str]) -> dict[str, float | None]:
        """Fetch PE ratios concurrently using a thread pool."""
        out: dict[str, float | None] = {}
        with ThreadPoolExecutor(max_workers=self.cfg.pe_workers) as pool:
            futures = {pool.submit(self._fetch_pe, t): t for t in tickers}
            for fut in as_completed(futures):
                ticker, pe = fut.result()
                out[ticker] = pe
        return out

    def _score_ohlcv(self, ticker: str, df: pd.DataFrame) -> dict | None:
        min_bars = max(
            self.cfg.rsi_window + self.cfg.vol_sma_window,
            self.cfg.sma_trend_window,
            self.cfg.vol_trend_short,
        )
        if len(df) < min_bars:
            return None
        close  = df["Close"].astype(float)
        volume = df["Volume"].astype(float)
        sma200 = _sma_series(close, self.cfg.sma_trend_window)
        last_sma = sma200.iloc[-1]
        return {
            "Ticker":       ticker,
            "Close":        round(float(close.iloc[-1]), 2),
            "RSI(14)":      round(self._rsi(close), 1),
            "Vol Ratio":    round(self._vol_ratio(volume), 2),
            "Above SMA200": bool(pd.notna(last_sma) and close.iloc[-1] > last_sma),
            "Vol Trend Up": bool(_vol_trend_series(volume, self.cfg.vol_trend_short, self.cfg.vol_sma_window).iloc[-1]),
        }

    def filter_stocks(
        self,
        data: dict[str, pd.DataFrame],
        pe_data: dict[str, float | None] | None = None,
    ) -> pd.DataFrame:
        """
        Screen *data* against all three factors.

        Pass 1 (vectorized):  RSI > min_rsi  AND  Vol ≥ vol_spike_mult × SMA-20
        Pass 2 (concurrent):  fetch PE only for Pass-1 survivors
        Pass 3:               PE < max_pe  →  rank by Quant Score

        Populates self.stats with {total, rsi_vol_pass, final}.
        """
        rows = [
            row
            for ticker, df in data.items()
            if (row := self._score_ohlcv(ticker, df)) is not None
        ]
        if not rows:
            self.stats = {"total": 0, "rsi_vol_pass": 0, "final": 0}
            return pd.DataFrame()

        scored = pd.DataFrame(rows)
        rsi_vol_mask = (
            (scored["RSI(14)"]      > self.cfg.min_rsi) &
            (scored["Vol Ratio"]    >= self.cfg.vol_spike_mult) &
            (scored["Above SMA200"] == True) &
            (scored["Vol Trend Up"] == True)
        )
        candidates = scored[rsi_vol_mask].copy()
        self.stats  = {"total": len(scored), "rsi_vol_pass": len(candidates), "final": 0}

        if candidates.empty:
            return candidates

        if pe_data is None:
            pe_data = self.fetch_pe_batch(candidates["Ticker"].tolist())

        candidates["PE Ratio"] = candidates["Ticker"].map(pe_data)
        candidates = candidates[
            candidates["PE Ratio"].notna() &
            (candidates["PE Ratio"] > self.cfg.min_pe) &
            (candidates["PE Ratio"] < self.cfg.max_pe)
        ].copy()

        if candidates.empty:
            return candidates

        # ── Weighted Quant Score: RSI 30% | Volume 40% | Value 30% ──────────────
        _VOL_CAP = 5.0  # cap before scoring to neutralise illiquidity pumps

        # RSI — trapezoid: ramp 0→100 over [50,55], hold 100 over [55,65],
        #                   drop 100→0 over [65,75], zero above 75 (overbought)
        rsi = candidates["RSI(14)"]
        rsi_score = pd.Series(
            np.where(rsi <= 55, (rsi - 50).clip(lower=0) / 5,
            np.where(rsi <= 65, 1.0,
            np.where(rsi <= 75, (75 - rsi) / 10,
            0.0))) * 100,
            index=rsi.index,
        ).clip(0, 100)

        # Volume — linear from min-threshold to cap, outliers clipped at 5×
        vol      = candidates["Vol Ratio"].clip(upper=_VOL_CAP)
        vol_band = max(float(_VOL_CAP - self.cfg.vol_spike_mult), 1e-9)
        vol_score = ((vol - self.cfg.vol_spike_mult) / vol_band * 100).clip(0, 100)

        # Value — linear: lower PE → higher score within [0, max_pe]
        pe_score = ((self.cfg.max_pe - candidates["PE Ratio"]) / self.cfg.max_pe * 100).clip(0, 100)

        candidates["Quant Score"] = (
            0.30 * rsi_score +
            0.40 * vol_score +
            0.30 * pe_score
        ).round(1)

        candidates = (
            candidates
            .sort_values("Quant Score", ascending=False)
            .reset_index(drop=True)
        )
        candidates.index      = candidates.index + 1
        candidates.index.name = "Rank"
        self.stats["final"]   = len(candidates)

        return c

# ── Dashboard indicator suite (used by dashboard.py chart explorer) ───────────
# Consolidated here from the former quant_engine.py to eliminate the duplicate
# module. Uses the `ta` library for the full indicator set (EMA, MACD, BB, ATR,
# Stochastic, OBV, VWAP) which is richer than the screener's lean inline maths.

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical indicators to *df* and return a new DataFrame."""
    df     = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # Trend
    df["EMA_20"]      = ta.trend.ema_indicator(close, window=20)
    df["EMA_50"]      = ta.trend.ema_indicator(close, window=50)
    df["SMA_200"]     = ta.trend.sma_indicator(close, window=200)
    df["MACD"]        = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    df["MACD_Hist"]   = ta.trend.macd_diff(close)

    # Momentum
    df["RSI"] = ta.momentum.rsi(close, window=14)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Mid"]   = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["ATR"]      = ta.volatility.average_true_range(high, low, close, window=14)

    # Volume
    df["OBV"]  = ta.volume.on_balance_volume(close, volume)
    df["VWAP"] = ta.volume.volume_weighted_average_price(high, low, close, volume, window=14)

    return df


def screen(df: pd.DataFrame) -> dict[str, bool]:
    """
    Run boolean screening checks on the latest row of *df*.

    Returns {signal_name: bool}. Call add_indicators(df) first.
    """
    last  = df.iloc[-1]
    close = float(last["Close"])
    return {
        "RSI Oversold (<30)":          float(last["RSI"]) < 30,
        "RSI Overbought (>70)":        float(last["RSI"]) > 70,
        "Price > EMA 20":              close > float(last["EMA_20"]),
        "Price > EMA 50":              close > float(last["EMA_50"]),
        "Golden Cross (EMA20>EMA50)":  float(last["EMA_20"]) > float(last["EMA_50"]),
        "Price > SMA 200":             close > float(last["SMA_200"]),
        "MACD Bullish Crossover":      float(last["MACD"]) > float(last["MACD_Signal"]),
        "BB Squeeze (Width<0.1)":      float(last["BB_Width"]) < 0.1,
        "Price Near BB Lower (<2%)":   abs(close - float(last["BB_Lower"])) / close < 0.02,
        "Stoch Oversold (K<20)":       float(last["Stoch_K"]) < 20,
        "Stoch Overbought (K>80)":     float(last["Stoch_K"]) > 80,
    }


def latest_metrics(df: pd.DataFrame) -> dict:
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
