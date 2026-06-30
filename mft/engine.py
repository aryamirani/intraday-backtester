from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .core import MarketSnapshot, Side
from .data import Dataset, DayMarket
from .instruments import parse_instrument
from .portfolio import CostModel, Portfolio
from .strategy import Strategy


@dataclass
class BacktestResult:
    underlier: str
    mtm: pd.DataFrame          # per-second equity / pnl / exposure curve
    trades: pd.DataFrame       # one row per fill
    lot_size: float

    @property
    def final_pnl(self) -> float:
        return float(self.mtm["equity"].iloc[-1]) if len(self.mtm) else 0.0


class BacktestEngine:
    """Runs one strategy over one underlier across a sequence of days. The
    portfolio (and hence the equity curve) is cumulative across days; positions
    are always flat between days because the engine flattens at each close."""

    def __init__(self, dataset: Dataset, lot_size: float = 1.0,
                 cost_model: CostModel | None = None, max_position: int = 1):
        self.dataset = dataset
        self.lot_size = lot_size
        self.cost_model = cost_model
        self.max_position = max_position

    def run(self, strategy: Strategy, underlier: str,
            dates: list[date] | None = None, progress=None) -> BacktestResult:
        dates = dates or self.dataset.dates
        portfolio = Portfolio(self.lot_size, self.cost_model, self.max_position)

        frames: list[pd.DataFrame] = []
        iterator = progress(dates) if progress else dates
        for d in iterator:
            frames.append(self._run_day(strategy, portfolio, underlier, d))

        mtm = pd.concat(frames) if frames else pd.DataFrame()
        trades = pd.DataFrame(
            [(f.timestamp, f.symbol, f.signed_qty, f.price, f.cost, f.reason)
             for f in portfolio.fills],
            columns=["timestamp", "symbol", "signed_qty", "price", "cost", "reason"],
        )
        if not trades.empty:
            strikes = trades["symbol"].map(lambda s: parse_instrument(s).strike)
            trades.insert(2, "strike", strikes)
            trades.insert(3, "opt_type", trades["symbol"].map(lambda s: parse_instrument(s).opt_type))
        return BacktestResult(underlier, mtm, trades, self.lot_size)

    def _run_day(self, strategy: Strategy, portfolio: Portfolio,
                 underlier: str, d: date) -> pd.DataFrame:
        market = DayMarket(self.dataset.day_dir(d), underlier, d)
        strategy.on_day_start(market)
        n = len(market.index)

        equity = np.empty(n)
        realized = np.empty(n)
        unrealized = np.empty(n)
        n_pos = np.empty(n, dtype="int32")
        held = np.full(n, np.nan)

        for i in range(n):
            market.i = i
            ts = market.index[i].to_pydatetime()
            is_close = i == n - 1
            snap = MarketSnapshot(ts, market, is_session_close=is_close)

            for order in strategy.on_step(snap, portfolio):
                self._execute(portfolio, market, order.symbol, order.signed_qty,
                              i, ts, order.reason)

            if is_close:
                self._flatten(portfolio, market, i, ts)

            mark_fn = lambda sym, _i=i: market.price_at(sym, _i)
            realized[i] = portfolio.realized_pnl
            unrealized[i] = portfolio.unrealized_pnl(mark_fn)
            equity[i] = realized[i] + unrealized[i]
            open_syms = portfolio.open_symbols()
            n_pos[i] = len(open_syms)
            if open_syms:
                held[i] = parse_instrument(open_syms[0]).strike

        return pd.DataFrame(
            {
                "underlier": underlier,
                "equity": equity,
                "realized": realized,
                "unrealized": unrealized,
                "n_positions": n_pos,
                "held_strike": held,
                "futures": market.futures,
            },
            index=market.index,
        )

    def _execute(self, portfolio: Portfolio, market: DayMarket, symbol: str,
                 signed_qty: int, i: int, ts, reason: str) -> None:
        mark = market.price_at(symbol, i)
        if not np.isfinite(mark):
            return  # leg not tradable yet; strategy will retry next second
        resulting = portfolio.position(symbol) + signed_qty
        if abs(resulting) > self.max_position:
            return  # position cap guard
        portfolio.fill(ts, symbol, signed_qty, mark, reason)

    def _flatten(self, portfolio: Portfolio, market: DayMarket, i: int, ts) -> None:
        for sym in portfolio.open_symbols():
            qty = portfolio.position(sym)
            mark = market.price_at(sym, i)
            if np.isfinite(mark):
                portfolio.fill(ts, sym, int(-qty), mark, reason="eod_flatten")
