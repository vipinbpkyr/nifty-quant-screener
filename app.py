"""
Streamlit dashboard — Value + Momentum + Volume Stock Screener.

Run:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_engine import get_sp500_tickers, get_nifty500_tickers, fetch_batch_ohlcv
from quant_logic import Screener, ScreenerConfig, Backtester

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Stock Screener",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Quant Screener")
    st.caption("Value + Momentum + Volume")

    st.header("Market")
    market = st.selectbox("Index Universe", ["S&P 500", "Nifty 500"])

    st.divider()
    st.header("Screener Thresholds")
    max_pe = st.slider(
        "Max PE Ratio  (Value)",
        5, 50, 20, step=1,
        help="Stocks with PE at or above this value are excluded.",
    )
    min_rsi = st.slider(
        "Min RSI-14  (Momentum)",
        30, 80, 50, step=1,
        help="Only stocks whose 14-day RSI exceeds this level pass.",
    )
    vol_mult = st.slider(
        "Min Volume Ratio  (Volume)",
        1.0, 6.0, 2.0, step=0.5,
        help="Current volume must be at least this multiple of the 20-day SMA.",
    )

    st.divider()
    st.header("Backtest Settings")
    fwd_days = st.number_input(
        "Forward return window (days)",
        min_value=1, max_value=20, value=5, step=1,
        help="Number of trading days after each signal to measure the return.",
    )

    st.divider()
    st.header("Data Settings")
    period = st.selectbox("Lookback period", ["1mo", "3mo", "6mo"], index=2,
                          help="6mo recommended — backtest needs sufficient signal history.")
    chunk_size = st.number_input(
        "Download chunk size",
        min_value=10, max_value=100, value=50, step=10,
        help="Tickers per batch API call — reduce if hitting Yahoo rate limits.",
    )

    st.divider()
    run = st.button("▶  Run Screener", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Top Quant Picks")
st.caption("Stocks passing all three quantitative filters, ranked by composite Quant Score.")

if not run:
    st.info("Configure your filters in the sidebar and click **▶ Run Screener** to start.")
    st.stop()

# ── Execute scan + backtest ───────────────────────────────────────────────────
config   = ScreenerConfig(
    max_pe=float(max_pe),
    min_rsi=float(min_rsi),
    vol_spike_mult=float(vol_mult),
)
screener  = Screener(config)
backtester = Backtester(config)

with st.status("Scanning…", expanded=True) as scan_status:
    st.write(f"⬇ Loading {market} constituent tickers…")
    tickers = get_sp500_tickers() if market == "S&P 500" else get_nifty500_tickers()
    st.write(f"  ↳ {len(tickers)} tickers loaded.")

    st.write(f"⬇ Downloading OHLCV data in chunks of {int(chunk_size)}…")
    ohlcv = fetch_batch_ohlcv(tickers, period=period, chunk_size=int(chunk_size))
    st.write(f"  ↳ {len(ohlcv)} tickers with valid data.")

    st.write("⚙ Pass 1 — RSI + Volume screen (vectorized)…")
    st.write("⚙ Pass 2 — PE fetch for survivors (concurrent)…")
    st.write("⚙ Pass 3 — Value filter + Quant Score ranking…")
    results = screener.filter_stocks(ohlcv)

    # Backtest runs on the same OHLCV — fully vectorized, no API calls
    st.write(f"📊 Backtesting RSI + Volume signal ({fwd_days}-day forward returns)…")
    bt_stats  = backtester.aggregate_stats(ohlcv, fwd_days=int(fwd_days))
    bt_detail = backtester.run_batch(ohlcv, fwd_days=int(fwd_days))

    scan_status.update(
        label=f"Scan complete — {screener.stats.get('final', 0)} picks found.",
        state="complete",
        expanded=False,
    )

st.divider()

# ── Summary strip (above tabs) ────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers Scanned",  screener.stats.get("total", len(ohlcv)))
c2.metric("Passed RSI + Vol", screener.stats.get("rsi_vol_pass", "—"))
c3.metric("Final Picks",      screener.stats.get("final", 0))
c4.metric(
    "Avg Quant Score",
    f"{results['Quant Score'].mean():.1f}" if not results.empty else "—",
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Top Quant Picks", f"🧪 Backtest — {fwd_days}-Day Signal Return"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCREENER RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if results.empty:
        st.warning(
            "No stocks passed all three filters. "
            "Try relaxing thresholds — increase Max PE, lower Min RSI, "
            "or reduce the Volume Ratio."
        )
    else:
        st.subheader(f"{market}  ·  {screener.stats['final']} stocks matched")
        st.caption(
            f"0 < PE < {max_pe}  ·  RSI(14) > {min_rsi}  ·  "
            f"Volume ≥ {vol_mult}× SMA-20  ·  Price > SMA-200  ·  "
            f"Vol Trend Up  ·  Period: {period}"
        )
        st.dataframe(
            results.reset_index(),
            use_container_width=True,
            height=min(40 * len(results) + 60, 650),
            column_config={
                "Rank":        st.column_config.NumberColumn("Rank", width="small"),
                "Ticker":      st.column_config.TextColumn("Ticker"),
                "Close":       st.column_config.NumberColumn("Price", format="%.2f"),
                "RSI(14)":     st.column_config.ProgressColumn(
                                   "RSI(14)", min_value=0, max_value=100, format="%.1f"
                               ),
                "Vol Ratio":   st.column_config.NumberColumn("Vol / SMA20", format="%.2f×"),
                "PE Ratio":    st.column_config.NumberColumn("PE", format="%.1f"),
                "Quant Score": st.column_config.ProgressColumn(
                                   "Quant Score", min_value=0, max_value=100, format="%.1f"
                               ),
            },
            hide_index=True,
        )
        st.download_button(
            "⬇  Download CSV",
            data=results.reset_index().to_csv(index=False).encode(),
            file_name=f"quant_picks_{market.replace(' ', '_').lower()}_{period}.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    if not bt_stats:
        st.warning("Insufficient historical data to run a backtest. Try a longer period (6mo).")
        st.stop()

    st.subheader("Portfolio-Level Statistics")
    st.caption(
        f"Tickers where RSI(14) > {min_rsi}, Volume ≥ {vol_mult}× SMA-20, "
        f"Price > SMA-200, and volume trending up — all fired historically.  "
        f"Each signal's {fwd_days}-day forward return is recorded.  "
        f"Equity curve: equal-weight daily basket, one unit of capital."
    )

    # ── Backtest metrics strip ────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)

    win_rate_val = bt_stats["win_rate"]
    win_delta    = f"{'above' if win_rate_val >= 50 else 'below'} random"
    b1.metric("Total Signal Trades",  bt_stats["total_trades"])
    b2.metric(
        "Win Rate",
        f"{win_rate_val:.1f}%",
        delta=win_delta,
        delta_color="normal" if win_rate_val >= 50 else "inverse",
    )
    b3.metric(
        f"Avg {fwd_days}-Day Return",
        f"{bt_stats['avg_fwd_return']:+.2f}%",
        delta_color="normal" if bt_stats["avg_fwd_return"] >= 0 else "inverse",
    )
    b4.metric(
        "Max Drawdown",
        f"{bt_stats['max_drawdown']:.2f}%",
        delta_color="off",
    )

    # ── Equity curve + drawdown chart ─────────────────────────────────────────
    eq = bt_stats["equity_curve"]
    dd = bt_stats["drawdown_curve"] * 100   # convert to percentage

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        subplot_titles=("Cumulative Equity Curve (equal-weight daily basket)", "Drawdown (%)"),
    )

    # Equity curve — colour the fill based on whether above or below starting capital
    eq_color    = "#26a69a"
    eq_fill     = "rgba(38,166,154,0.15)"
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=eq.values,
            name="Equity",
            mode="lines",
            line=dict(color=eq_color, width=2),
            fill="tozeroy",
            fillcolor=eq_fill,
            hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Starting capital reference line
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(180,180,180,0.5)", row=1, col=1)

    # Drawdown — filled red below zero
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values,
            name="Drawdown",
            mode="lines",
            line=dict(color="#ef5350", width=1),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.2)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %",        tickformat=".1f",  row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,80,80,0.3)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,80,80,0.3)")

    st.plotly_chart(fig, use_container_width=True)

    # ── Per-ticker breakdown table ────────────────────────────────────────────
    st.subheader("Per-Ticker Signal Breakdown")
    st.caption("Individual ticker results — only tickers where the signal fired at least once.")

    if bt_detail.empty:
        st.info("No individual ticker results available.")
    else:
        st.dataframe(
            bt_detail,
            use_container_width=True,
            height=min(40 * len(bt_detail) + 60, 550),
            column_config={
                "Ticker":         st.column_config.TextColumn("Ticker"),
                "Signals":        st.column_config.NumberColumn("# Signals", width="small"),
                "Win Rate %":     st.column_config.ProgressColumn(
                                      "Win Rate %", min_value=0, max_value=100, format="%.1f"
                                  ),
                "Avg 5D Ret %":   st.column_config.NumberColumn(
                                      f"Avg {fwd_days}D Ret %", format="%+.2f"
                                  ),
                "Max Drawdown %": st.column_config.NumberColumn(
                                      "Max DD %", format="%.2f"
                                  ),
            },
            hide_index=True,
        )

        st.download_button(
            "⬇  Download Backtest CSV",
            data=bt_detail.to_csv(index=False).encode(),
            file_name=f"backtest_{market.replace(' ', '_').lower()}_{fwd_days}d.csv",
            mime="text/csv",
        )
