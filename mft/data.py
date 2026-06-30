from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import is_option_name, option_symbol, parse_instrument

SESSION_START = time(9, 15, 0)
SESSION_END = time(15, 30, 0)
COLUMNS = ["date", "time", "price", "volume", "oi"]

FUTURES_DIRNAME = "Futures (Continuous)"
OPTIONS_DIRNAME = "Options"


def session_index(d: date) -> pd.DatetimeIndex:
    """1-second grid spanning the trading session for a given date."""
    start = datetime.combine(d, SESSION_START)
    end = datetime.combine(d, SESSION_END)
    return pd.date_range(start, end, freq="1s")


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, dtype={"date": str, "time": str})
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H:%M:%S")
    df.index = ts
    return df


def load_price_series(path: Path, index: pd.DatetimeIndex) -> np.ndarray:
    """Last-traded price aligned to a 1-second grid.

    Multiple ticks within the same second collapse to the last one; seconds
    without a trade carry the previous traded price forward. Seconds before the
    instrument's first trade are NaN (so the engine can defer entry until the
    leg actually prints)."""
    raw = _read_raw(path)
    per_second = raw["price"].groupby(raw.index.floor("1s")).last()
    aligned = per_second.reindex(index).ffill()
    return aligned.to_numpy(dtype="float64")


def date_from_dirname(name: str) -> date:
    # NSE_YYYYMMDD
    stamp = name.split("_")[-1]
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))


class Dataset:
    """Discovers day folders under ``root`` and exposes their dates in order."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._days = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("NSE_")),
            key=lambda p: p.name,
        )

    @property
    def day_dirs(self) -> list[Path]:
        return self._days

    @property
    def dates(self) -> list[date]:
        return [date_from_dirname(p.name) for p in self._days]

    def day_dir(self, d: date) -> Path:
        target = f"NSE_{d:%Y%m%d}"
        for p in self._days:
            if p.name == target:
                return p
        raise KeyError(f"No data folder for {d}")


def _expiries_for(options_dir: Path, underlier: str) -> dict[str, set[str]]:
    """Map ``YYMMDD`` expiry -> set of option types present, restricted to the
    given underlier (exact match, so NIFTY does not swallow BANKNIFTY)."""
    out: dict[str, set[str]] = {}
    for f in options_dir.glob(f"{underlier}*.csv"):
        if not is_option_name(f.name):
            continue
        inst = parse_instrument(f.name)
        if inst.underlier != underlier:
            continue
        key = f"{inst.expiry:%y%m%d}"
        out.setdefault(key, set()).add(inst.opt_type)
    return out


def select_nearest_expiry(expiries: list[date], today: date) -> date:
    """The expiry closest to ``today`` that has not already passed; if all have
    passed (shouldn't happen intraday), fall back to the closest by distance."""
    upcoming = [e for e in expiries if e >= today]
    if upcoming:
        return min(upcoming)
    return min(expiries, key=lambda e: abs((e - today).days))


class DayMarket:
    """Per-(underlier, day) market view handed to the strategy and used by the
    engine for fills and marks. Option legs are loaded lazily and cached, so a
    run only ever touches the handful of near-ATM strikes it actually trades."""

    def __init__(self, day_dir: Path, underlier: str, d: date):
        self.day_dir = day_dir
        self.underlier = underlier
        self.date = d
        self.index = session_index(d)
        self.i = 0  # current second, advanced by the engine

        self._options_dir = day_dir / OPTIONS_DIRNAME
        self._series_cache: dict[str, np.ndarray] = {}

        fut_path = day_dir / FUTURES_DIRNAME / f"{underlier}-I.csv"
        self.futures = load_price_series(fut_path, self.index)

        expiry_map = _expiries_for(self._options_dir, underlier)
        expiry_dates = [parse_instrument(f"{underlier}{e}0CE").expiry for e in expiry_map]
        self.expiry = select_nearest_expiry(expiry_dates, d)
        self.expiry_code = f"{self.expiry:%y%m%d}"

        # Only strikes available as BOTH a call and a put are tradable as a straddle.
        ce = self._strikes_for("CE")
        pe = self._strikes_for("PE")
        self.strikes = np.array(sorted(set(ce) & set(pe)), dtype="int64")

    def _strikes_for(self, opt_type: str) -> list[int]:
        prefix = f"{self.underlier}{self.expiry_code}"
        out = []
        for f in self._options_dir.glob(f"{prefix}*{opt_type}.csv"):
            inst = parse_instrument(f.name)
            if inst.underlier == self.underlier and inst.opt_type == opt_type:
                out.append(inst.strike)
        return out

    # ---- price access -------------------------------------------------
    def series(self, symbol: str) -> np.ndarray:
        cached = self._series_cache.get(symbol)
        if cached is None:
            path = self._options_dir / f"{symbol}.csv"
            cached = load_price_series(path, self.index)
            self._series_cache[symbol] = cached
        return cached

    def price_at(self, symbol: str, i: int | None = None) -> float:
        i = self.i if i is None else i
        return float(self.series(symbol)[i])

    # ---- convenience for strategies -----------------------------------
    @property
    def timestamp(self) -> datetime:
        return self.index[self.i].to_pydatetime()

    @property
    def futures_price(self) -> float:
        return float(self.futures[self.i])

    def symbol(self, strike: int, opt_type: str) -> str:
        return option_symbol(self.underlier, self.expiry_code, strike, opt_type)

    def nearest_strike(self, price: float) -> int | None:
        if not np.isfinite(price) or self.strikes.size == 0:
            return None
        pos = int(np.searchsorted(self.strikes, price))
        if pos == 0:
            return int(self.strikes[0])
        if pos >= self.strikes.size:
            return int(self.strikes[-1])
        low, high = self.strikes[pos - 1], self.strikes[pos]
        # Ties resolve to the higher strike (round-half-up); this only matters
        # at the exact midpoint and is otherwise immaterial.
        return int(high if (price - low) >= (high - price) else low)

    def option_price(self, strike: int, opt_type: str) -> float:
        return self.price_at(self.symbol(strike, opt_type))
