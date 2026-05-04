"""
data_provider.py — DEPRECATED.

All functions have been consolidated into data_engine.py.
This shim re-exports them for backwards compatibility.
"""
from data_engine import fetch_ohlcv, fetch_multiple, get_info  # noqa: F401
