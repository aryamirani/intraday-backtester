from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import DayMarket


class Side(Enum):
    BUY = 1
    SELL = -1


@dataclass(frozen=True, slots=True)
class Order:
    """A market order for a single instrument. The engine fills it at the
    instrument's prevailing 1-second mark."""

    symbol: str
    side: Side
    quantity: int = 1
    reason: str = ""  # free-form tag, useful for inspecting why a trade happened

    @property
    def signed_qty(self) -> int:
        return self.side.value * self.quantity


@dataclass(frozen=True, slots=True)
class Fill:
    timestamp: datetime
    symbol: str
    signed_qty: int
    price: float
    cost: float
    reason: str


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0


@dataclass(slots=True)
class MarketSnapshot:
    """What a strategy sees at one instant. ``market`` exposes the prevailing
    futures price, the tradable strike grid and per-leg option prices for the
    underlier being traded at the current second."""

    timestamp: datetime
    market: "DayMarket"
    is_session_close: bool = False
