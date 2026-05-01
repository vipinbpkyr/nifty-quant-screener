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
    """POST a single message to Telegram using HTML parse mode."""
    url  = _TG_API.format(token=token)
    resp = requests.post(
        url,
        json={
            "chat_id":                  chat_id,
            "text":                     text,
            "parse_mode":               "HTML",
            "disable_web_page_preview": True,
        },
        timeout=15,
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

def _h(value: object) -> str:
    """Escape HTML special characters for Telegram's HTML parse mode."""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_message(market: str, picks: pd.DataFrame, bt: dict) -> str:
    """
    Build a compact, mobile-readable HTML message.

    Structure:
      [Header]   market name, timestamp, active filter thresholds
      [Backtest] portfolio win rate, avg return, max drawdown, trade count
      [Picks]    top-N ranked stocks, one line each
      [Footer]   bot attribution
    """
    ts  = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
    cfg = CONFIG

    lines: list[str] = [
        f"📊 <b>Daily Quant Scan — {_h(market)}</b>",
        f"<i>{ts}</i>",
        f"<i>PE &lt; {cfg.max_pe:.0f}  ·  RSI &gt; {cfg.min_rsi:.0f}"
        f"  ·  Vol ≥ {cfg.vol_spike_mult:.1f}×</i>",
        "",
    ]

    # ── Backtest summary ──────────────────────────────────────────────────────
    if bt:
        sign = "+" if bt["avg_fwd_return"] >= 0 else ""
        lines += [
            f"<b>📈 Signal Backtest ({FWD_DAYS}-day return)</b>",
            f"  Win Rate    <code>{bt['win_rate']:.1f}%</code>",
            f"  Avg Return  <code>{sign}{bt['avg_fwd_return']:.2f}%</code>",
            f"  Max Drawdown <code>{bt['max_drawdown']:.1f}%</code>",
            f"  Trades      <code>{bt['total_trades']}</code>",
            "",
        ]
    else:
        lines += ["<i>Backtest skipped — insufficient history.</i>", ""]

    # ── Top picks ─────────────────────────────────────────────────────────────
    if picks.empty:
        lines.append("❌ <i>No stocks passed all three filters today.</i>")
    else:
        top = picks.reset_index().head(TOP_N)
        lines.append(f"<b>🏆 Top {len(top)} Picks</b>")
        for _, row in top.iterrows():
            pe_str    = f"PE {row['PE Ratio']:.0f}" if pd.notna(row.get("PE Ratio")) else "PE —"
            score_str = f"{row['Quant Score']:.0f}"
            lines.append(
                f"  <b>{_h(row['Ticker'])}</b>"
                f"  {row['Close']:.2f}"
                f"  RSI {row['RSI(14)']:.0f}"
                f"  Vol {row['Vol Ratio']:.1f}×"
                f"  {pe_str}"
                f"  ⭐{score_str}"
            )

    lines += ["", "<i>🤖 Nifty Quant Screener — automated daily run</i>"]

    msg = "\n".join(lines)

    # Telegram hard limit is 4 096 characters
    if len(msg) > _MAX_MSG:
        msg = msg[: _MAX_MSG - 40] + "\n…\n<i>(message truncated)</i>"

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
