"""Interactive Streamlit dashboard for the backtester.

Run with:
    streamlit run dashboard.py

Allows real-time parameter tuning and instant comparison across strategies,
underliers, and cost models — all without touching code or config files.
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mft.analytics import (
    compute_metrics, daily_pnl, drawdown, round_trips,
    plot_equity, plot_drawdown, plot_daily_pnl, plot_position_timeline,
    combined_equity,
)
from mft.data import Dataset
from mft.engine import BacktestEngine
from mft.portfolio import CostModel, VolatilityScaledCostModel
from mft.strategy import (
    NearestStraddle, TimeWeightedStraddle, WidenedStrangle,
    STRATEGY_REGISTRY,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Index-Option Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Index-Option Backtester Dashboard")
st.caption("Interactive parameter tuning and strategy comparison")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    data_root = st.text_input("Data root", value="allData")
    try:
        dataset = Dataset(data_root)
        available_dates = dataset.dates
    except Exception:
        st.error(f"Cannot find data at `{data_root}`. Place the `allData` folder at the repo root.")
        st.stop()

    st.subheader("Strategy")
    strategy_name = st.selectbox(
        "Strategy",
        list(STRATEGY_REGISTRY.keys()),
        format_func=lambda s: s.replace("_", " ").title(),
    )

    # Strategy-specific parameters
    if strategy_name == "nearest_straddle":
        hysteresis = st.slider("Hysteresis (pts)", 0.0, 100.0, 0.0, 1.0,
                               help="Anti-whipsaw band: roll only after the futures "
                                    "moves past this threshold.")
        strategy = NearestStraddle(hysteresis=hysteresis)

    elif strategy_name == "time_weighted_straddle":
        interval = st.slider("Rebalance interval (seconds)", 1, 600, 60, 1)
        hysteresis = st.slider("Hysteresis (pts)", 0.0, 100.0, 0.0, 1.0)
        strategy = TimeWeightedStraddle(rebalance_interval_s=interval,
                                        hysteresis=hysteresis)

    elif strategy_name == "widened_strangle":
        width = st.slider("Strike width offset", 0, 10, 1, 1,
                          help="How many strikes away from ATM to place each leg.")
        hysteresis = st.slider("Hysteresis (pts)", 0.0, 100.0, 0.0, 1.0)
        strategy = WidenedStrangle(width=width, hysteresis=hysteresis)

    st.subheader("Underliers")
    underliers = st.multiselect(
        "Select underliers",
        ["NIFTY", "BANKNIFTY", "FINNIFTY"],
        default=["NIFTY", "BANKNIFTY"],
    )

    st.subheader("Date range")
    max_days = len(available_dates)
    n_days = st.slider("Trading days", 1, max_days, max_days)

    st.subheader("Execution costs")
    cost_type = st.radio("Cost model", ["Frictionless", "Static", "Volatility-scaled"])
    if cost_type == "Static":
        slippage = st.number_input("Slippage (pts/unit)", 0.0, 10.0, 0.5, 0.1)
        fee_rate = st.number_input("Fee rate", 0.0, 0.01, 0.0003, 0.0001, format="%.4f")
        cost_model = CostModel(per_unit_slippage=slippage, fee_rate=fee_rate)
    elif cost_type == "Volatility-scaled":
        base_slip = st.number_input("Base slippage (pts)", 0.0, 10.0, 0.5, 0.1)
        vol_mult = st.number_input("Vol multiplier", 0.0, 1.0, 0.1, 0.01)
        fee_rate = st.number_input("Fee rate", 0.0, 0.01, 0.0003, 0.0001, format="%.4f")
        cost_model = VolatilityScaledCostModel(
            base_slippage=base_slip, vol_multiplier=vol_mult, fee_rate=fee_rate)
    else:
        cost_model = CostModel()

    lot_size = st.number_input("Lot size (contract multiplier)", 1.0, 100.0, 1.0, 1.0,
                               help="Use 50 for NIFTY rupee PnL, 15 for BANKNIFTY, etc.")

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if not run_btn:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to start.")
    st.stop()

if not underliers:
    st.warning("Select at least one underlier.")
    st.stop()

dates = available_dates[:n_days]

# Run the backtest
results = {}
with st.spinner("Running backtest..."):
    progress = st.progress(0)
    for idx, underlier in enumerate(underliers):
        engine = BacktestEngine(dataset, lot_size=lot_size, cost_model=cost_model)
        strat = copy.deepcopy(strategy)
        result = engine.run(strat, underlier, dates=dates)
        results[underlier] = result
        progress.progress((idx + 1) / len(underliers))
    progress.empty()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
st.header("📊 Summary Metrics")
metric_data = []
for name, r in results.items():
    m = compute_metrics(r)
    metric_data.append(m.as_row())

summary_df = pd.DataFrame(metric_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Top-level KPI cards
cols = st.columns(len(underliers) + 1)
for i, (name, r) in enumerate(results.items()):
    m = compute_metrics(r)
    cols[i].metric(name, f"{m.final_pnl:,.1f} pts", delta=f"DD: {m.max_drawdown:,.1f}")

# Combined
comb_eq = combined_equity(results)
if comb_eq is not None:
    final_combined = float(comb_eq.iloc[-1])
    cols[-1].metric("Combined", f"{final_combined:,.1f} pts")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
st.header("📈 Equity Curves")
fig, ax = plt.subplots(figsize=(12, 4))
plot_equity(results, ax)
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Drawdown")
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_drawdown(results, ax)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Daily PnL")
    fig, ax = plt.subplots(figsize=(6, 3))
    plot_daily_pnl(results, ax)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Position timeline for a selected day
# ---------------------------------------------------------------------------
st.header("🎯 Position Timeline")
day_options = [d.isoformat() for d in dates]
selected_day = st.selectbox("Select day", day_options, index=0)
selected_underlier = st.selectbox("Select underlier", list(results.keys()), index=0)

if selected_underlier in results:
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_position_timeline(results[selected_underlier], selected_day, ax)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------
st.header("📋 Trade Log")
trade_underlier = st.selectbox("Underlier (trades)", list(results.keys()), index=0,
                               key="trade_ul")
if trade_underlier in results:
    trades = results[trade_underlier].trades
    if not trades.empty:
        st.dataframe(trades, use_container_width=True, height=300)
        st.caption(f"{len(trades)} fills total")

        # Round-trip analysis
        rt = round_trips(trades)
        if not rt.empty:
            st.subheader("Round-trip analysis")
            st.dataframe(rt, use_container_width=True, height=200)
            avg_hold = rt["hold_s"].mean()
            win_rate = (rt["pnl"] > 0).mean() * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg hold time", f"{avg_hold:.0f}s")
            col2.metric("Win rate", f"{win_rate:.1f}%")
            col3.metric("Avg PnL/trade", f"{rt['pnl'].mean():.2f} pts")
    else:
        st.info("No trades executed.")
