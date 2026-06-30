"""PnL attribution for the straddle.

A long straddle's PnL over a holding segment splits cleanly, with no Greeks
required, into two economically meaningful pieces:

* **directional** -- the change in the legs' *intrinsic* value as the futures
  moves (|F - K| for the pair), i.e. what the move was worth, and
* **time / volatility (extrinsic)** -- everything left over: the change in the
  options' time value, which for a continuously-held long straddle is the theta
  bleed that the brief's strategy is structurally exposed to.

Per leg:  ``pnl = (intrinsic_exit - intrinsic_entry) + (extrinsic change)``.
Summed across every leg this equals realised PnL (ex-fees), so the decomposition
is exact, not a model approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analytics import round_trips
from .data import Dataset, DayMarket, session_index
from .engine import BacktestResult


def _intrinsic(opt_type: str, futures: float, strike: int) -> float:
    return max(futures - strike, 0.0) if opt_type == "CE" else max(strike - futures, 0.0)


@dataclass
class Attribution:
    underlier: str
    directional: float
    extrinsic: float
    fees: float

    @property
    def total(self) -> float:
        return self.directional + self.extrinsic - self.fees

    def as_row(self) -> dict:
        return {
            "underlier": self.underlier,
            "directional_pnl": round(self.directional, 2),
            "extrinsic_pnl": round(self.extrinsic, 2),
            "fees": round(self.fees, 2),
            "total_pnl": round(self.total, 2),
        }


def attribute(result: BacktestResult, dataset: Dataset) -> tuple[Attribution, pd.DataFrame]:
    rt = round_trips(result.trades)
    if rt.empty:
        return Attribution(result.underlier, 0.0, 0.0, 0.0), rt

    fut_cache: dict = {}

    def futures_at(d, ts) -> float:
        if d not in fut_cache:
            market = DayMarket(dataset.day_dir(d), result.underlier, d)
            fut_cache[d] = (session_index(d)[0], market.futures)
        start, arr = fut_cache[d]
        sec = int((ts - start).total_seconds())
        sec = min(max(sec, 0), len(arr) - 1)
        return float(arr[sec])

    rows = []
    for _, leg in rt.iterrows():
        d = leg["entry"].date()
        f_entry = futures_at(d, leg["entry"])
        f_exit = futures_at(d, leg["exit"])
        intr_entry = _intrinsic(leg["opt_type"], f_entry, leg["strike"])
        intr_exit = _intrinsic(leg["opt_type"], f_exit, leg["strike"])
        sign = np.sign(leg["qty"])
        directional = (intr_exit - intr_entry) * sign
        total = leg["exit_px"] - leg["entry_px"]
        total *= sign
        rows.append({
            "date": d, "strike": leg["strike"], "opt_type": leg["opt_type"],
            "directional": directional, "extrinsic": total - directional,
            "leg_pnl": total,
        })

    legs = pd.DataFrame(rows)
    lot = result.lot_size
    attribution = Attribution(
        underlier=result.underlier,
        directional=float(legs["directional"].sum() * lot),
        extrinsic=float(legs["extrinsic"].sum() * lot),
        fees=float(result.trades["cost"].sum()),
    )
    return attribution, legs


def plot_attribution(attributions: list[Attribution], ax) -> None:
    names = [a.underlier for a in attributions]
    directional = [a.directional for a in attributions]
    extrinsic = [a.extrinsic for a in attributions]
    x = np.arange(len(names))
    ax.bar(x, directional, label="Directional (intrinsic)", color="steelblue")
    ax.bar(x, extrinsic, bottom=directional, label="Time / vol (extrinsic)", color="indianred")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("PnL (points)")
    ax.set_title("PnL attribution: directional vs time-decay")
    ax.legend(loc="best", fontsize=8)
