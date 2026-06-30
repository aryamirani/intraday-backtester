from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .core import Fill, Position


@dataclass
class CostModel:
    """Execution frictions. Defaults to frictionless since the brief is about
    measuring the strategy, not making it profitable. ``per_unit_slippage`` is
    added to the price on buys and subtracted on sells; ``fee_rate`` is charged
    on traded notional."""

    per_unit_slippage: float = 0.0
    fee_rate: float = 0.0

    def execution_price(self, mark: float, signed_qty: int) -> float:
        direction = 1 if signed_qty > 0 else -1
        return mark + direction * self.per_unit_slippage

    def fee(self, price: float, signed_qty: int) -> float:
        return self.fee_rate * abs(signed_qty) * price


class Portfolio:
    """Tracks signed positions, average cost, realized cash and unrealized
    mark-to-market. PnL is in index points per ``lot_size`` unit; set
    ``lot_size`` to the contract multiplier for a rupee view."""

    def __init__(self, lot_size: float = 1.0, cost_model: CostModel | None = None,
                 max_position: int = 1):
        self.lot_size = lot_size
        self.costs = cost_model or CostModel()
        self.max_position = max_position
        self.positions: dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.fills: list[Fill] = []

    def position(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.quantity if p else 0.0

    def fill(self, timestamp: datetime, symbol: str, signed_qty: int, mark: float,
             reason: str = "") -> Fill:
        price = self.costs.execution_price(mark, signed_qty)
        fee = self.costs.fee(price, signed_qty) * self.lot_size

        pos = self.positions.setdefault(symbol, Position(symbol))
        prev_qty, avg = pos.quantity, pos.avg_price
        new_qty = prev_qty + signed_qty

        opening_same_way = prev_qty == 0 or (prev_qty > 0) == (signed_qty > 0)
        if opening_same_way:
            pos.avg_price = (avg * prev_qty + price * signed_qty) / new_qty if new_qty else 0.0
        else:
            closed = min(abs(signed_qty), abs(prev_qty))
            direction = 1 if prev_qty > 0 else -1
            self.realized_pnl += (price - avg) * closed * direction * self.lot_size
            if abs(signed_qty) > abs(prev_qty):  # flipped through zero
                pos.avg_price = price

        pos.quantity = new_qty
        if new_qty == 0:
            pos.avg_price = 0.0

        self.realized_pnl -= fee
        self.total_fees += fee
        f = Fill(timestamp, symbol, signed_qty, price, fee, reason)
        self.fills.append(f)
        return f

    def unrealized_pnl(self, mark_fn) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            if pos.quantity:
                total += (mark_fn(sym) - pos.avg_price) * pos.quantity * self.lot_size
        return total

    def equity(self, mark_fn) -> float:
        return self.realized_pnl + self.unrealized_pnl(mark_fn)

    def open_symbols(self) -> list[str]:
        return [s for s, p in self.positions.items() if p.quantity]
