"""
Streamlit dashboard — interactive stock screener UI.

Run:
    streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_engine import fetch_ohlcv, fetch_multiple, get_info
from quant_logic import add_indicators, screen, latest_metrics

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Screener", layout="wide", page_icon="📈")
st.title("📈 Nifty Quant Screener")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Single Ticker", "Watchlist Scan"])
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TICKER MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Ticker":
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()

    if st.button("Analyse", type="primary") and ticker:
        with st.spinner(f"Fetching {ticker}…"):
            try:
                df_raw  = fetch_ohlcv(ticker, period=period, interval=interval)
                df      = add_indicators(df_raw)
                signals = screen(df)
                metrics = latest_metrics(df)
                info    = get_info(ticker)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        # Company info strip
        st.subheader(info["name"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Sector", info["sector"])
        c2.metric("Industry", info["industry"])
        cap = info["market_cap"]
        c3.metric("Market Cap", f"${cap/1e9:.1f}B" if cap else "N/A")

        # Key metrics
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close",    f"${metrics['Close']}")
        m2.metric("RSI",      metrics["RSI"])
        m3.metric("MACD",     metrics["MACD"])
        m4.metric("ATR",      metrics["ATR"])
        m5.metric("BB Width", metrics["BB_Width"])

        # Candlestick + indicators chart
        st.divider()
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20],
            vertical_spacing=0.03,
            subplot_titles=("Price & Bollinger Bands", "RSI (14)", "MACD"),
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
            low=df["Low"].squeeze(), close=df["Close"].squeeze(), name="OHLC",
        ), row=1, col=1)

        for col, color, dash in [
            ("BB_Upper", "rgba(200,200,200,0.4)", "dot"),
            ("BB_Mid",   "rgba(200,200,200,0.8)", "dash"),
            ("BB_Lower", "rgba(200,200,200,0.4)", "dot"),
            ("EMA_20",   "#26c6da", "solid"),
            ("EMA_50",   "#ef5350", "solid"),
        ]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col].squeeze(), name=col,
                line=dict(color=color, dash=dash, width=1),
            ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(), name="RSI",
                                 line=dict(color="#ab47bc")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        colors = ["green" if v >= 0 else "red" for v in df["MACD_Hist"].squeeze()]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"].squeeze(), name="MACD Hist",
                             marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(), name="MACD",
                                 line=dict(color="#42a5f5")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"].squeeze(), name="Signal",
                                 line=dict(color="#ffa726")), row=3, col=1)

        fig.update_layout(height=700, showlegend=True,
                          xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Signals
        st.divider()
        st.subheader("Screening Signals")
        cols = st.columns(3)
        for i, (signal, triggered) in enumerate(signals.items()):
            cols[i % 3].metric(signal, "✅ YES" if triggered else "⬜ NO")

# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST SCAN MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    default_watchlist = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, NFLX, AMD"
    raw = st.text_area("Tickers (comma-separated)", value=default_watchlist, height=80)
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    if st.button("Run Scan", type="primary") and tickers:
        with st.spinner(f"Scanning {len(tickers)} tickers…"):
            data = fetch_multiple(tickers, period=period, interval=interval)

        rows = []
        progress = st.progress(0)
        for i, (t, df_raw) in enumerate(data.items()):
            df  = add_indicators(df_raw)
            sig = screen(df)
            met = latest_metrics(df)
            rows.append({"Ticker": t, **met, **{k: ("✅" if v else "⬜") for k, v in sig.items()}})
            progress.progress((i + 1) / len(data))

        if rows:
            result_df = pd.DataFrame(rows).set_index("Ticker")
            st.dataframe(result_df, use_container_width=True)
            csv = result_df.to_csv().encode()
            st.download_button("Download CSV", csv, "scan_results.csv", "text/csv")
        else:
            st.warning("No valid data returned for any ticker.")
