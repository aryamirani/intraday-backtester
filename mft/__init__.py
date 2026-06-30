"""A small, strategy-agnostic event-driven backtester for index options.

The engine knows nothing about any particular strategy: strategies observe the
market and emit orders, the engine fills them, tracks positions and cash, and
records a mark-to-market curve. Swapping in a new strategy means implementing a
single method.
"""

from .instruments import Instrument, parse_instrument
from .core import Order, Side, Fill, Position, MarketSnapshot
from .portfolio import Portfolio
from .engine import BacktestEngine, BacktestResult
from .strategy import Strategy, NearestStraddle

__all__ = [
    "Instrument",
    "parse_instrument",
    "Order",
    "Side",
    "Fill",
    "Position",
    "MarketSnapshot",
    "Portfolio",
    "BacktestEngine",
    "BacktestResult",
    "Strategy",
    "NearestStraddle",
]
