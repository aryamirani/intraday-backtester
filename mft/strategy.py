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


# ---------------------------------------------------------------------------
# Additional pluggable strategies — demonstrating engine agnosticism.
# ---------------------------------------------------------------------------

class TimeWeightedStraddle(Strategy):
    """A straddle strategy that only rebalances at fixed intervals rather than
    on every tick.

    Instead of continuously chasing the nearest strike, this strategy only
    checks whether to roll every ``rebalance_interval_s`` seconds (default:
    60 = one minute). Within a bucket the position is held even if the futures
    moves past another strike. This reduces churn dramatically and shows the
    engine handles arbitrary rebalance cadences.

    Parameters
    ----------
    rebalance_interval_s : int
        Minimum number of seconds between rebalance checks.
    hysteresis : float
        Optional anti-whipsaw band (same meaning as NearestStraddle).
    """

    def __init__(self, rebalance_interval_s: int = 60, hysteresis: float = 0.0):
        self.interval = rebalance_interval_s
        self.hysteresis = hysteresis
        self.held_strike: int | None = None
        self._seconds_since_last: int = 0

    def on_day_start(self, market) -> None:
        self.held_strike = None
        self._seconds_since_last = self.interval  # trigger immediately at open

    def on_step(self, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[Order]:
        self._seconds_since_last += 1
        if self._seconds_since_last < self.interval and self.held_strike is not None:
            return []
        self._seconds_since_last = 0

        market = snapshot.market
        fut = market.futures_price
        nearest = market.nearest_strike(fut)
        if nearest is None:
            return []

        # Hysteresis guard
        if self.held_strike is not None and nearest != self.held_strike and self.hysteresis > 0:
            midpoint = (self.held_strike + nearest) / 2.0
            if abs(fut - midpoint) < self.hysteresis:
                return []

        if nearest == self.held_strike:
            return []

        orders: list[Order] = []
        if self.held_strike is not None:
            for opt in ("CE", "PE"):
                orders.append(Order(market.symbol(self.held_strike, opt), Side.SELL,
                                    reason="tw_roll_out"))

        new_legs = []
        for opt in ("CE", "PE"):
            if np.isfinite(market.option_price(nearest, opt)):
                new_legs.append(Order(market.symbol(nearest, opt), Side.BUY,
                                      reason="tw_roll_in"))
        if len(new_legs) < 2:
            return []

        orders.extend(new_legs)
        self.held_strike = nearest
        return orders


class WidenedStrangle(Strategy):
    """Hold a strangle (OTM CE + OTM PE) offset by ``width`` strikes from ATM.

    Instead of buying the nearest-strike call and put (a straddle), this
    strategy buys a call at the Nth strike *above* the nearest and a put at the
    Nth strike *below*. ``width=0`` collapses to an ATM straddle. ``width=1``
    gives a one-strike-wide strangle. This demonstrates the engine's ability to
    handle asymmetric leg selection with no engine changes.
    """

    def __init__(self, width: int = 1, hysteresis: float = 0.0):
        self.width = width
        self.hysteresis = hysteresis
        self.held_ce_strike: int | None = None
        self.held_pe_strike: int | None = None

    def on_day_start(self, market) -> None:
        self.held_ce_strike = None
        self.held_pe_strike = None

    def _target_strikes(self, market) -> tuple[int | None, int | None]:
        """Return (ce_strike, pe_strike) offset from ATM by self.width."""
        fut = market.futures_price
        nearest = market.nearest_strike(fut)
        if nearest is None:
            return self.held_ce_strike, self.held_pe_strike

        strikes = market.strikes
        atm_idx = int(np.searchsorted(strikes, nearest))
        # Clamp indices to valid range
        ce_idx = min(atm_idx + self.width, len(strikes) - 1)
        pe_idx = max(atm_idx - self.width, 0)
        return int(strikes[ce_idx]), int(strikes[pe_idx])

    def on_step(self, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[Order]:
        market = snapshot.market
        ce_target, pe_target = self._target_strikes(market)
        if ce_target is None or pe_target is None:
            return []
        if ce_target == self.held_ce_strike and pe_target == self.held_pe_strike:
            return []

        orders: list[Order] = []
        # Close existing legs
        if self.held_ce_strike is not None:
            orders.append(Order(market.symbol(self.held_ce_strike, "CE"), Side.SELL,
                                reason="strangle_roll_out"))
        if self.held_pe_strike is not None:
            orders.append(Order(market.symbol(self.held_pe_strike, "PE"), Side.SELL,
                                reason="strangle_roll_out"))

        # Open new legs — only if both are tradable
        ce_sym = market.symbol(ce_target, "CE")
        pe_sym = market.symbol(pe_target, "PE")
        if not (np.isfinite(market.price_at(ce_sym)) and np.isfinite(market.price_at(pe_sym))):
            return []

        orders.append(Order(ce_sym, Side.BUY, reason="strangle_roll_in"))
        orders.append(Order(pe_sym, Side.BUY, reason="strangle_roll_in"))
        self.held_ce_strike = ce_target
        self.held_pe_strike = pe_target
        return orders


# ---------------------------------------------------------------------------
# Strategy registry — enables config-driven strategy construction.
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "nearest_straddle": NearestStraddle,
    "time_weighted_straddle": TimeWeightedStraddle,
    "widened_strangle": WidenedStrangle,
}
