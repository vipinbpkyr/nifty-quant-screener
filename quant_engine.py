"""
quant_engine.py — DEPRECATED.

All functions have been consolidated into quant_logic.py.
This shim re-exports them for backwards compatibility.
"""
from quant_logic import add_indicators, screen, latest_metrics  # noqa: F401
