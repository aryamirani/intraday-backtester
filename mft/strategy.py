from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .core import MarketSnapshot, Order, Side
from .portfolio import Portfolio


class Strategy(ABC):
    """Strategy interface. The engine is otherwise agnostic: implement
    ``on_step`` to look at the market plus your current book and return the
    orders you want filled this second. ``on_day_start`` is an optional hook
    for resetting per-day state."""

    def on_day_start(self, market) -> None:  # noqa: B027 - optional hook
        pass

    @abstractmethod
    def on_step(self, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[Order]:
        ...


class NearestStraddle(Strategy):
    """Hold a long straddle (1x CE + 1x PE) on the strike nearest the futures
    price. When the nearest strike changes, roll: sell the held pair and buy the
    new pair. Positions are flattened by the engine at the close.

    ``hysteresis`` is an optional, off-by-default guard: a roll only triggers
    once the futures has moved at least this many points past the midpoint
    between the held and the candidate strike. With the default of 0 the
    behaviour is exactly "always pick the nearest strike, roll on change"."""

    def __init__(self, hysteresis: float = 0.0):
        self.hysteresis = hysteresis
        self.held_strike: int | None = None

    def on_day_start(self, market) -> None:
        self.held_strike = None

    def _target_strike(self, market) -> int | None:
        fut = market.futures_price
        nearest = market.nearest_strike(fut)
        if nearest is None:
            return self.held_strike
        if self.held_strike is None or self.hysteresis <= 0:
            return nearest
        if nearest == self.held_strike:
            return self.held_strike
        # Only roll once price has cleared the midpoint by the hysteresis band.
        midpoint = (self.held_strike + nearest) / 2.0
        if abs(fut - midpoint) < self.hysteresis:
            return self.held_strike
        return nearest

    def on_step(self, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[Order]:
        market = snapshot.market
        target = self._target_strike(market)
        if target is None or target == self.held_strike:
            return []

        orders: list[Order] = []
        if self.held_strike is not None:
            for opt in ("CE", "PE"):
                orders.append(Order(market.symbol(self.held_strike, opt), Side.SELL,
                                    reason="roll_out"))

        # Only enter legs that have actually printed; otherwise leave the roll
        # to a later second so we never fill on a price that doesn't exist yet.
        new_legs = []
        for opt in ("CE", "PE"):
            if np.isfinite(market.option_price(target, opt)):
                new_legs.append(Order(market.symbol(target, opt), Side.BUY, reason="roll_in"))
        if len(new_legs) < 2:
            # Defer the whole roll until both legs are tradable.
            return []

        orders.extend(new_legs)
        self.held_strike = target
        return orders
