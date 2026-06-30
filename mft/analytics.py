from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestResult

TRADING_DAYS = 252


def daily_pnl(mtm: pd.DataFrame) -> pd.Series:
    """End-of-day equity differences -> realized daily PnL."""
    eod = mtm["equity"].groupby(mtm.index.date).last()
    eod.index = pd.to_datetime(list(eod.index))
    return eod.diff().fillna(eod.iloc[0] if len(eod) else 0.0)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity - equity.cummax()


def round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair each opening fill with its offsetting close per symbol (FIFO) to
    get realized leg-level PnL and holding time."""
    rows: list[dict] = []
    if trades.empty:
        return pd.DataFrame(rows)
    for sym, grp in trades.groupby("symbol"):
        open_lots: list[tuple] = []  # (timestamp, qty, price)
        for _, t in grp.sort_values("timestamp").iterrows():
            qty, price = int(t["signed_qty"]), float(t["price"])
            while qty != 0 and open_lots and (open_lots[0][1] > 0) != (qty > 0):
                o_ts, o_qty, o_px = open_lots[0]
                matched = min(abs(o_qty), abs(qty))
                direction = 1 if o_qty > 0 else -1
                rows.append({
                    "symbol": sym, "strike": int(t["strike"]), "opt_type": t["opt_type"],
                    "entry": o_ts, "exit": t["timestamp"], "qty": matched * direction,
                    "entry_px": o_px, "exit_px": price,
                    "pnl": (price - o_px) * matched * direction,
                    "hold_s": (t["timestamp"] - o_ts).total_seconds(),
                })
                if matched == abs(o_qty):
                    open_lots.pop(0)
                else:
                    open_lots[0] = (o_ts, o_qty - direction * matched, o_px)
                qty += direction * matched
            if qty != 0:
                open_lots.append((t["timestamp"], qty, price))
    return pd.DataFrame(rows)


@dataclass
class Metrics:
    underlier: str
    final_pnl: float
    n_fills: int
    n_rolls: int
    max_drawdown: float
    sharpe_like: float
    avg_daily_pnl: float
    pnl_std: float
    total_fees: float

    def as_row(self) -> dict:
        return {
            "underlier": self.underlier,
            "final_pnl": round(self.final_pnl, 2),
            "fills": self.n_fills,
            "rolls": self.n_rolls,
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_like": round(self.sharpe_like, 3),
            "avg_daily_pnl": round(self.avg_daily_pnl, 2),
            "pnl_std": round(self.pnl_std, 2),
            "fees": round(self.total_fees, 2),
        }


def compute_metrics(result: BacktestResult) -> Metrics:
    mtm = result.mtm
    dpnl = daily_pnl(mtm)
    std = float(dpnl.std(ddof=0))
    sharpe = float(dpnl.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    n_rolls = int((result.trades["reason"] == "roll_in").sum()) if not result.trades.empty else 0
    return Metrics(
        underlier=result.underlier,
        final_pnl=result.final_pnl,
        n_fills=len(result.trades),
        n_rolls=n_rolls,
        max_drawdown=float(drawdown(mtm["equity"]).min()),
        sharpe_like=sharpe,
        avg_daily_pnl=float(dpnl.mean()),
        pnl_std=std,
        total_fees=float(result.trades["cost"].sum()) if not result.trades.empty else 0.0,
    )


def combined_equity(results: dict[str, BacktestResult]) -> pd.Series:
    """Sum per-underlier equity curves on their shared 1-second grid."""
    eq = None
    for r in results.values():
        s = r.mtm["equity"]
        eq = s if eq is None else eq.add(s, fill_value=0.0)
    return eq


# --------------------------------------------------------------------------
# Plotting (matplotlib). Each helper takes an Axes so the notebook controls
# layout; all are intentionally dependency-light.
# --------------------------------------------------------------------------
def plot_equity(results: dict[str, BacktestResult], ax) -> None:
    for name, r in results.items():
        ax.plot(r.mtm.index, r.mtm["equity"], label=name, lw=1.0)
    combined = combined_equity(results)
    ax.plot(combined.index, combined.values, label="Combined", color="black", lw=1.4)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title("Cumulative mark-to-market PnL (index points)")
    ax.set_ylabel("PnL (points)")
    ax.legend(loc="best", fontsize=8)


def plot_drawdown(results: dict[str, BacktestResult], ax) -> None:
    combined = combined_equity(results)
    dd = drawdown(combined)
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.4)
    ax.set_title("Drawdown of combined PnL (points)")
    ax.set_ylabel("Drawdown")


def plot_daily_pnl(results: dict[str, BacktestResult], ax) -> None:
    frame = pd.DataFrame({name: daily_pnl(r.mtm) for name, r in results.items()})
    frame.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Daily PnL by underlier (points)")
    ax.set_xticklabels([d.strftime("%m-%d") for d in frame.index], rotation=45, fontsize=7)
    ax.set_ylabel("PnL (points)")


def plot_position_timeline(result: BacktestResult, day, ax) -> None:
    """Held strike vs futures for a single day, showing the rolls."""
    day = pd.Timestamp(day)
    mask = result.mtm.index.normalize() == day.normalize()
    sub = result.mtm[mask]
    ax.plot(sub.index, sub["futures"], color="steelblue", lw=1.0, label="Futures")
    ax.step(sub.index, sub["held_strike"], color="darkorange", lw=1.2,
            where="post", label="Held strike")
    ax.set_title(f"{result.underlier}: futures vs held strike ({day:%Y-%m-%d})")
    ax.set_ylabel("Price / strike")
    ax.legend(loc="best", fontsize=8)
