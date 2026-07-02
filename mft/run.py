from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .analytics import compute_metrics
from .data import Dataset
from .engine import BacktestEngine
from .portfolio import CostModel
from .strategy import NearestStraddle, TimeWeightedStraddle, WidenedStrangle, Strategy

LOT_SIZE = {"NIFTY": 50, "BANKNIFTY": 15, "FINNIFTY": 40}


def run_all(data_root: str | Path = "allData",
            underliers: tuple[str, ...] = ("NIFTY", "BANKNIFTY", "FINNIFTY"),
            out_dir: str | Path = "results",
            lot_size: float = 1.0,
            slippage: float = 0.0,
            fee_rate: float = 0.0,
            hysteresis: float = 0.0,
            dates: list[date] | None = None,
            verbose: bool = True,
            strategy: Strategy | None = None,
            cost_model: CostModel | None = None,
            max_position: int = 1) -> dict:
    dataset = Dataset(data_root)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    try:
        from tqdm import tqdm
        progress = (lambda it: tqdm(it, desc="days", leave=False)) if verbose else None
    except ImportError:
        progress = None

    if cost_model is None:
        cost_model = CostModel(per_unit_slippage=slippage, fee_rate=fee_rate)
    results = {}
    metric_rows = []
    for underlier in underliers:
        engine = BacktestEngine(dataset, lot_size=lot_size, cost_model=cost_model,
                                max_position=max_position)
        # Use the provided strategy or create a default NearestStraddle.
        # Each underlier gets its own strategy instance to avoid shared state.
        if strategy is not None:
            import copy
            strat = copy.deepcopy(strategy)
        else:
            strat = NearestStraddle(hysteresis=hysteresis)
        if verbose:
            print(f"Running {underlier} with {type(strat).__name__} ...")
        result = engine.run(strat, underlier, dates=dates, progress=progress)
        results[underlier] = result

        result.mtm.to_parquet(out / f"mtm_{underlier}.parquet")
        result.trades.to_parquet(out / f"trades_{underlier}.parquet")
        m = compute_metrics(result)
        metric_rows.append(m.as_row())
        if verbose:
            print(f"  {underlier}: final PnL {m.final_pnl:,.1f} pts, "
                  f"{m.n_rolls} rolls, max DD {m.max_drawdown:,.1f}")

    summary = pd.DataFrame(metric_rows)
    summary.to_csv(out / "summary.csv", index=False)
    if verbose:
        print("\n" + summary.to_string(index=False))
    return results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run the nearest-straddle backtest.")
    p.add_argument("--data-root", default="allData")
    p.add_argument("--out", default="results")
    p.add_argument("--underliers", nargs="+", default=["NIFTY", "BANKNIFTY", "FINNIFTY"])
    p.add_argument("--lot-size", type=float, default=1.0)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--fee-rate", type=float, default=0.0)
    p.add_argument("--hysteresis", type=float, default=0.0)
    p.add_argument("--days", type=int, default=None,
                   help="limit to first N trading days (smoke test)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML or JSON config file (overrides other flags)")
    args = p.parse_args()

    if args.config:
        from .config import RunConfig
        cfg = RunConfig.from_file(args.config)
        dataset = Dataset(cfg.data_root)
        dates = dataset.dates[:args.days] if args.days else None
        run_all(cfg.data_root, cfg.underliers, cfg.out_dir, cfg.lot_size,
                dates=dates, strategy=cfg.strategy, cost_model=cfg.cost_model,
                max_position=cfg.max_position)
    else:
        dataset = Dataset(args.data_root)
        dates = dataset.dates[: args.days] if args.days else None
        run_all(args.data_root, tuple(args.underliers), args.out, args.lot_size,
                args.slippage, args.fee_rate, args.hysteresis, dates)


if __name__ == "__main__":
    _cli()

