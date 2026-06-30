"""Independent reconciliation of the engine's PnL.

The engine accumulates PnL incrementally through average-cost bookkeeping. To
trust those numbers we recompute them here in two completely different ways that
share no code with ``Portfolio``:

1. a single cash-flow identity over the trade log, and
2. a full per-second equity curve rebuilt from the trade log plus the raw price
   series.

If both agree with the engine to floating-point tolerance, the accounting is
correct by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analytics import round_trips
from .data import Dataset, DayMarket, session_index
from .engine import BacktestResult


@dataclass
class Reconciliation:
    underlier: str
    engine_pnl: float
    cashflow_pnl: float
    roundtrip_pnl: float
    curve_max_abs_diff: float

    def passed(self, tol: float = 1e-6) -> bool:
        return (
            abs(self.cashflow_pnl - self.engine_pnl) <= tol
            and abs(self.roundtrip_pnl - self.engine_pnl) <= tol
            and self.curve_max_abs_diff <= tol
        )

    def as_row(self) -> dict:
        return {
            "underlier": self.underlier,
            "engine_pnl": round(self.engine_pnl, 6),
            "cashflow_pnl": round(self.cashflow_pnl, 6),
            "roundtrip_pnl": round(self.roundtrip_pnl, 6),
            "curve_max_abs_diff": float(f"{self.curve_max_abs_diff:.2e}"),
            "reconciled": self.passed(),
        }


def cashflow_pnl(result: BacktestResult) -> float:
    """With a flat book at the end, total PnL is just the negative of every
    cash flow (buys cost cash, sells return it) minus fees."""
    t = result.trades
    if t.empty:
        return 0.0
    gross = -(t["signed_qty"] * t["price"]).sum() * result.lot_size
    return float(gross - t["cost"].sum())


def roundtrip_pnl(result: BacktestResult) -> float:
    rt = round_trips(result.trades)
    if rt.empty:
        return 0.0
    return float(rt["pnl"].sum() * result.lot_size - result.trades["cost"].sum())


def independent_equity_curve(result: BacktestResult, dataset: Dataset) -> pd.Series:
    """Rebuild the cumulative per-second equity from scratch: positions and cash
    are derived directly from the fills, marked against freshly loaded price
    series. Uses none of the engine's incremental bookkeeping."""
    t = result.trades
    if t.empty:
        return result.mtm["equity"] * 0.0

    lot = result.lot_size
    pieces: list[pd.Series] = []
    carry = 0.0  # cumulative realized PnL handed over from previous days
    t = t.assign(date=t["timestamp"].dt.normalize())

    for day, day_trades in t.groupby("date"):
        d = day.date()
        idx = session_index(d)
        n = len(idx)
        start = idx[0]
        market = DayMarket(dataset.day_dir(d), result.underlier, d)

        cash = np.zeros(n)
        position_value = np.zeros(n)
        for sym, sym_trades in day_trades.groupby("symbol"):
            pos_delta = np.zeros(n)
            cash_delta = np.zeros(n)
            for _, f in sym_trades.iterrows():
                sec = int((f["timestamp"] - start).total_seconds())
                pos_delta[sec] += f["signed_qty"]
                cash_delta[sec] += -f["signed_qty"] * f["price"] * lot + -f["cost"]
            pos = np.cumsum(pos_delta)
            price = np.nan_to_num(market.series(sym))
            position_value += np.where(pos != 0, pos * price * lot, 0.0)
            cash += np.cumsum(cash_delta)

        equity = carry + cash + position_value
        carry = float(equity[-1])
        pieces.append(pd.Series(equity, index=idx))

    return pd.concat(pieces)


def reconcile(result: BacktestResult, dataset: Dataset) -> Reconciliation:
    indep = independent_equity_curve(result, dataset)
    diff = float((indep - result.mtm["equity"]).abs().max())
    return Reconciliation(
        underlier=result.underlier,
        engine_pnl=result.final_pnl,
        cashflow_pnl=cashflow_pnl(result),
        roundtrip_pnl=roundtrip_pnl(result),
        curve_max_abs_diff=diff,
    )
