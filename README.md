# Nifty Quant Screener

A Python-based quantitative stock screener for **S&P 500** and **Nifty 500** built on a **Value + Momentum + Volume** framework. Includes a Streamlit dashboard, a vectorized backtester, and a GitHub Actions workflow that sends daily picks to Telegram.

---

## Features

- **Three-factor screen** — PE Ratio (Value), RSI-14 (Momentum), Volume spike vs 20-day SMA
- **Vectorized backtester** — 5-day forward return analysis with Win Rate and Max Drawdown
- **Two Streamlit apps** — index-wide screener (`app.py`) and single-ticker chart explorer (`dashboard.py`)
- **Daily automation** — GitHub Actions fires at 09:00 IST and pushes results to Telegram
- **Composite Quant Score** — equal-weight ranking across all three factors

---

## Project Structure

```
nifty-quant-screener/
│
├── app.py                          # Streamlit index screener + backtest tab
├── dashboard.py                    # Streamlit single-ticker chart explorer
│
├── data_engine.py                  # Batch OHLCV ingestion, S&P 500 / Nifty 500 tickers
├── data_provider.py                # Single-ticker OHLCV (used by dashboard.py)
│
├── quant_logic.py                  # Screener + Backtester classes (index screener)
├── quant_engine.py                 # Indicator suite (used by dashboard.py)
│
├── notify_telegram.py              # Headless daily scan + Telegram notification
│
├── launcher.py                     # Python launcher for both Streamlit apps
├── run.bat                         # Windows double-click launcher
│
├── requirements.txt
├── .gitignore
└── .github/
    └── workflows/
        └── daily_scan.yml          # GitHub Actions schedule (03:30 UTC / 09:00 IST)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run both Streamlit apps

**Windows — double-click:**
```
run.bat
```

**Or from the terminal:**
```bash
python launcher.py
```

| App | URL |
|---|---|
| Index Screener + Backtest | http://localhost:8502 |
| Single-Ticker Chart Explorer | http://localhost:8501 |

### 3. Run either app individually

```bash
streamlit run app.py        # index screener
streamlit run dashboard.py  # single-ticker explorer
```

---

## Screener Logic

### Filters (`quant_logic.py`)

| Factor | Signal | Default threshold |
|---|---|---|
| Value | PE Ratio | < 25 |
| Momentum | RSI-14 (Wilder EWM) | > 55 |
| Volume | Current vol / 20-day SMA | ≥ 2× |

All thresholds are adjustable from the Streamlit sidebar.

### Quant Score (0–100)

Equal-weight composite of all three factors — used to rank final picks:

```
RSI Score   = (RSI - 50) / 50 × 100       clipped [0, 100]
Volume Score = (Vol Ratio / multiplier) × 50  clipped [0, 100]
Value Score  = (max_PE - PE) / max_PE × 100   clipped [0, 100]

Quant Score = (RSI Score + Volume Score + Value Score) / 3
```

### Backtester (`quant_logic.py — Backtester`)

For every historical bar where the RSI + Volume signal fired (last 6 months), records the N-day forward close-to-close return. Reports:

- **Win Rate** — % of signal trades with positive return
- **Avg N-Day Return** — mean forward return across all signal bars
- **Max Drawdown** — peak-to-trough decline of the portfolio equity curve
- **Equity Curve** — equal-weight daily basket, one unit of capital

No look-ahead bias: `shift(1)` on the volume baseline, signal bars in the last N rows excluded.

---

## Daily Telegram Alerts

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) — copy the token
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Add both as GitHub Secrets:

```
Repository → Settings → Secrets and variables → Actions → New repository secret

TELEGRAM_BOT_TOKEN   your_bot_token
TELEGRAM_CHAT_ID     your_chat_id
```

### Test locally

```bash
export TELEGRAM_BOT_TOKEN=your_token
export TELEGRAM_CHAT_ID=your_chat_id
python notify_telegram.py
```

On Windows (PowerShell):
```powershell
$env:TELEGRAM_BOT_TOKEN = "your_token"
$env:TELEGRAM_CHAT_ID   = "your_chat_id"
python notify_telegram.py
```

### Trigger manually on GitHub

```
Actions → Daily Stock Scan & Telegram Alert → Run workflow
```

### Schedule

The workflow runs automatically at **03:30 UTC (09:00 IST)** every day via:

```yaml
on:
  schedule:
    - cron: '30 3 * * *'
  workflow_dispatch:
```

---

## Architecture

```
data_engine.py          ← data layer   (tickers, batch OHLCV)
    ↓
quant_logic.py          ← logic layer  (Screener, Backtester)
    ↓                       ↓
app.py                  notify_telegram.py
(Streamlit UI)          (headless CLI — no UI imports)
```

`notify_telegram.py` imports only from `data_engine` and `quant_logic` — completely decoupled from the Streamlit UI so it runs cleanly in GitHub Actions.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned packages

| Package | Purpose |
|---|---|
| `yfinance` | OHLCV data and PE ratios |
| `pandas` | Data manipulation |
| `numpy` | Vectorized indicator math |
| `ta` | Technical indicator library (dashboard) |
| `streamlit` | Web dashboard UI |
| `plotly` | Interactive charts |
| `requests` | HTTP (ticker lists, Telegram API) |
| `lxml` | HTML parsing for S&P 500 ticker scrape |
