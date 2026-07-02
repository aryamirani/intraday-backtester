"""Parameter grid search with parallel execution.

Sweeps a range of strategy parameters (e.g. hysteresis values) over the
backtest in parallel using ``concurrent.futures.ProcessPoolExecutor``, then
summarises the results and optionally plots a parameter sensitivity chart.

Usage::

    python -m mft.optimize --data-root allData --underlier NIFTY \\
           --param hysteresis --values 0 2 5 10 20 50 --workers 4

Or programmatically::

    from mft.optimize import grid_search
    results = grid_search("allData", "NIFTY", "hysteresis", [0, 5, 10, 20])
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import compute_metrics, daily_pnl, drawdown
from .data import Dataset
from .engine import BacktestEngine
from .portfolio import CostModel
from .strategy import NearestStraddle, TimeWeightedStraddle, STRATEGY_REGISTRY


@dataclass
class GridResult:
    param_name: str
    param_value: float
    final_pnl: float
    max_drawdown: float
    sharpe_like: float
    n_rolls: int
    avg_daily_pnl: float


def _run_single(data_root: str, underlier: str, strategy_name: str,
                param_name: str, param_value: float,
                dates: list[date] | None = None) -> GridResult:
    """Worker function for a single parameter value (must be picklable)."""
    dataset = Dataset(data_root)
    run_dates = dates or dataset.dates

    strat_cls = STRATEGY_REGISTRY[strategy_name]
    strategy = strat_cls(**{param_name: param_value})

    engine = BacktestEngine(dataset)
    result = engine.run(strategy, underlier, dates=run_dates)
    metrics = compute_metrics(result)

    return GridResult(
        param_name=param_name,
        param_value=param_value,
        final_pnl=metrics.final_pnl,
        max_drawdown=metrics.max_drawdown,
        sharpe_like=metrics.sharpe_like,
        n_rolls=metrics.n_rolls,
        avg_daily_pnl=metrics.avg_daily_pnl,
    )


def grid_search(
    data_root: str,
    underlier: str,
    param_name: str,
    param_values: list[float],
    strategy_name: str = "nearest_straddle",
    max_workers: int = 4,
    dates: list[date] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a parameter sweep in parallel and return a summary DataFrame."""
    results: list[GridResult] = []
    t0 = time.time()

    if verbose:
        print(f"Grid search: {underlier} | {param_name} = {param_values} "
              f"| workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_single, data_root, underlier, strategy_name,
                        param_name, v, dates): v
            for v in param_values
        }
        for fut in as_completed(futures):
            val = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                if verbose:
                    print(f"  {param_name}={val:>8.1f}  →  PnL={res.final_pnl:>10.1f}  "
                          f"DD={res.max_drawdown:>10.1f}  Sharpe={res.sharpe_like:>6.3f}  "
                          f"rolls={res.n_rolls}")
            except Exception as e:
                print(f"  {param_name}={val}: FAILED ({e})")

    elapsed = time.time() - t0
    if verbose:
        print(f"Completed {len(results)} runs in {elapsed:.1f}s")

    df = pd.DataFrame([
        {
            param_name: r.param_value,
            "final_pnl": r.final_pnl,
            "max_drawdown": r.max_drawdown,
            "sharpe_like": r.sharpe_like,
            "n_rolls": r.n_rolls,
            "avg_daily_pnl": r.avg_daily_pnl,
        }
        for r in results
    ]).sort_values(param_name).reset_index(drop=True)

    return df


def plot_sensitivity(df: pd.DataFrame, param_name: str, ax=None) -> None:
    """Plot parameter sensitivity: PnL and drawdown vs parameter value."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = ax

    # PnL vs parameter
    axes[0].plot(df[param_name], df["final_pnl"], "o-", color="steelblue", lw=1.5)
    axes[0].axhline(0, color="grey", lw=0.6, ls="--")
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Final PnL (pts)")
    axes[0].set_title(f"PnL vs {param_name}")

    # Max drawdown vs parameter
    axes[1].plot(df[param_name], df["max_drawdown"], "o-", color="firebrick", lw=1.5)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("Max Drawdown (pts)")
    axes[1].set_title(f"Drawdown vs {param_name}")

    # Sharpe vs parameter
    axes[2].plot(df[param_name], df["sharpe_like"], "o-", color="seagreen", lw=1.5)
    axes[2].axhline(0, color="grey", lw=0.6, ls="--")
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel("Sharpe-like")
    axes[2].set_title(f"Sharpe vs {param_name}")

    if ax is None:
        plt.tight_layout()


def _cli() -> None:
    p = argparse.ArgumentParser(description="Parameter grid search.")
    p.add_argument("--data-root", default="allData")
    p.add_argument("--underlier", default="NIFTY")
    p.add_argument("--strategy", default="nearest_straddle",
                   choices=list(STRATEGY_REGISTRY))
    p.add_argument("--param", default="hysteresis",
                   help="parameter name to sweep")
    p.add_argument("--values", nargs="+", type=float,
                   default=[0, 2, 5, 10, 15, 20, 30, 50])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--out", default="results/grid_search.csv")
    args = p.parse_args()

    ds = Dataset(args.data_root)
    dates = ds.dates[:args.days] if args.days else None

    df = grid_search(args.data_root, args.underlier, args.param, args.values,
                     strategy_name=args.strategy, max_workers=args.workers,
                     dates=dates)

    Path(args.out).parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    _cli()
