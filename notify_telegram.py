"""
notify_telegram.py — headless daily scanner + Telegram notification.

Runs the Screener and Backtester from quant_logic.py, formats the output into
a structured HTML message, and pushes it to a Telegram chat via the Bot API.

Intentionally decoupled from app.py:
  - imports only from data_engine and quant_logic
  - no Streamlit, no UI code
  - driven entirely by environment variables

Required environment variables (set as GitHub Secrets in the workflow):
  TELEGRAM_BOT_TOKEN  — bot token issued by @BotFather
  TELEGRAM_CHAT_ID    — numeric chat ID or @channel_username

Usage:
  python notify_telegram.py
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from data_engine import fetch_batch_ohlcv, get_nifty500_tickers, get_sp500_tickers
from quant_logic import Backtester, Screener, ScreenerConfig

# ── Scan configuration ────────────────────────────────────────────────────────

CONFIG = ScreenerConfig(
    max_pe=25.0,
    min_rsi=55.0,
    vol_spike_mult=2.0,
)

MARKETS: dict[str, object] = {
    "S&P 500":   get_sp500_tickers,
    "Nifty 500": get_nifty500_tickers,
}

PERIOD   = "6mo"   # needs enough history for a meaningful backtest
FWD_DAYS = 5       # days forward for backtest return measurement
TOP_N    = 10      # rows per message — keeps message under Telegram's 4 096-char limit

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_MSG = 4096


# ── Telegram layer ────────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, text: str) -> None:
    """POST a plain-text message to Telegram.

    No parse_mode — avoids 400 errors from HTML/Markdown special characters
    in ticker symbols or numeric values.  Logs Telegram's error body on failure
    so the exact reason is visible in GitHub Actions logs.
    """
    url  = _TG_API.format(token=token)
    resp = requests.post(
        url,
        json={
            "chat_id":                  chat_id,
            "text":                     text,
            "disable_web_page_preview": True,
        },
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Telegram API error {resp.status_code}: {resp.text}"
        )
    resp.raise_for_status()


# ── Scanner layer ─────────────────────────────────────────────────────────────

def run_market(ticker_fn) -> tuple[pd.DataFrame, dict]:
    """
    Fetch OHLCV, run Screener + Backtester, return (picks, bt_stats).

    Keeps data fetching and signal logic inside the modules that own them —
    this function is just an orchestration shim.
    """
    tickers  = ticker_fn()
    ohlcv    = fetch_batch_ohlcv(tickers, period=PERIOD)
    picks    = Screener(CONFIG).filter_stocks(ohlcv)
    bt_stats = Backtester(CONFIG).aggregate_stats(ohlcv, fwd_days=FWD_DAYS)
    return picks, bt_stats


# ── Formatter layer ───────────────────────────────────────────────────────────

def format_message(market: str, picks: pd.DataFrame, bt: dict) -> str:
    """
    Build a plain-text message — no HTML, no Markdown.

    Plain text is the safest Telegram format: no parser, no escaping rules,
    no 400 errors from special characters in ticker names or numeric values.

    Structure:
      Header    — market, timestamp, active thresholds
      Backtest  — win rate, avg return, max drawdown, trade count
      Picks     — top-N stocks, one line each
      Footer
    """
    ts  = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
    cfg = CONFIG
    div = "─" * 32

    lines: list[str] = [
        f"📊 Daily Quant Scan — {market}",
        ts,
        f"PE < {cfg.max_pe:.0f}  |  RSI > {cfg.min_rsi:.0f}  |  Vol >= {cfg.vol_spike_mult:.1f}x",
        div,
    ]

    # ── Backtest summary ──────────────────────────────────────────────────────
    if bt:
        sign = "+" if bt["avg_fwd_return"] >= 0 else ""
        lines += [
            f"📈 Backtest ({FWD_DAYS}-day return)",
            f"  Win Rate     {bt['win_rate']:.1f}%",
            f"  Avg Return   {sign}{bt['avg_fwd_return']:.2f}%",
            f"  Max Drawdown {bt['max_drawdown']:.1f}%",
            f"  Trades       {bt['total_trades']}",
            div,
        ]
    else:
        lines += ["Backtest skipped — insufficient history.", div]

    # ── Top picks ─────────────────────────────────────────────────────────────
    if picks.empty:
        lines.append("No stocks passed all three filters today.")
    else:
        top = picks.reset_index().head(TOP_N)
        lines.append(f"🏆 Top {len(top)} Picks")
        for _, row in top.iterrows():
            pe_str = f"PE {row['PE Ratio']:.0f}" if pd.notna(row.get("PE Ratio")) else "PE n/a"
            lines.append(
                f"  {row['Ticker']:<10}"
                f"  {row['Close']:>8.2f}"
                f"  RSI {row['RSI(14)']:>5.1f}"
                f"  Vol {row['Vol Ratio']:>4.1f}x"
                f"  {pe_str}"
                f"  Score {row['Quant Score']:.0f}"
            )

    lines += [div, "Nifty Quant Screener — automated daily run"]

    msg = "\n".join(lines)

    if len(msg) > _MAX_MSG:
        msg = msg[: _MAX_MSG - 20] + "\n...(truncated)"

    return msg


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set as "
            "environment variables (or GitHub Secrets).",
            file=sys.stderr,
        )
        sys.exit(1)

    exit_code = 0

    for market, ticker_fn in MARKETS.items():
        print(f"[{market}] Starting scan…")
        try:
            picks, bt_stats = run_market(ticker_fn)
            n = len(picks)
            print(f"[{market}] {n} picks found. Sending Telegram message…")
            msg = format_message(market, picks, bt_stats)
            send_telegram(token, chat_id, msg)
            print(f"[{market}] ✓ Sent ({len(msg)} chars).")
        except Exception as exc:
            print(f"[{market}] ✗ Failed: {exc}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
