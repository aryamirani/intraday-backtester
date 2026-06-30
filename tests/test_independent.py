"""Independent verification of the backtester.

The point of this file is to trust *nothing* from the `mft` package on the
quantities being checked. It contains a second, from-scratch implementation that
shares no code with the engine:

* its own CSV reader (the `csv` module, not pandas),
* its own 1-second resampling / forward-fill,
* its own expiry + strike selection (its own regex and globbing),
* its own strategy loop and cash-flow accounting.

If this naive oracle reproduces the engine's trades fill-for-fill and its PnL to
1e-9, the engine, the strategy and the data layer are all corroborated by a
genuinely separate code path. We then add mutation ("teeth") tests that corrupt
inputs and assert the reconciliation *fails*, proving the checks are not vacuous.

Run directly (`python tests/test_independent.py`) or via `pytest -v`.
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mft.data import Dataset
from mft.engine import BacktestEngine
from mft.strategy import NearestStraddle
from mft import reconcile

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(_PROJECT_ROOT / "allData")
FUTURES_DIR = "Futures (Continuous)"   # hard-coded on purpose, not imported
OPTIONS_DIR = "Options"
SESSION_START = time(9, 15, 0)
SESSION_END = time(15, 30, 0)


# --------------------------------------------------------------------------
# A completely independent reference implementation.
# --------------------------------------------------------------------------
def _grid(d: date) -> list[datetime]:
    start = datetime.combine(d, SESSION_START)
    end = datetime.combine(d, SESSION_END)
    n = int((end - start).total_seconds()) + 1
    return [start + timedelta(seconds=s) for s in range(n)]


def _read_series(path: Path, grid: list[datetime]) -> np.ndarray:
    """Last-traded price per second, forward-filled, via the stdlib csv reader.

    Deliberately avoids pandas so a bug in the package loader cannot hide here.
    """
    last_in_second: dict[datetime, float] = {}
    if path.exists():
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                # date, time, price, volume, oi
                ts = datetime.strptime(f"{row[0]} {row[1]}", "%Y%m%d %H:%M:%S")
                last_in_second[ts] = float(row[2])  # later rows overwrite -> last wins

    out = np.full(len(grid), np.nan)
    carry = math.nan
    for i, t in enumerate(grid):
        if t in last_in_second:
            carry = last_in_second[t]
        out[i] = carry
    return out


_OPT_RE_TMPL = r"^{u}(\d{{6}})(\d+)(CE|PE)\.csv$"


def _discover(day_dir: Path, underlier: str):
    opt_dir = day_dir / OPTIONS_DIR
    rx = re.compile(_OPT_RE_TMPL.format(u=underlier))
    expiries: dict[str, dict[str, set]] = {}
    for f in opt_dir.glob(f"{underlier}*.csv"):
        m = rx.match(f.name)
        if not m:
            continue
        exp, strike, opt = m.group(1), int(m.group(2)), m.group(3)
        expiries.setdefault(exp, {"CE": set(), "PE": set()})[opt].add(strike)
    return expiries


def _nearest_expiry(expiries: list[str], today: date) -> str:
    def to_date(e: str) -> date:
        return date(2000 + int(e[:2]), int(e[2:4]), int(e[4:6]))

    upcoming = [e for e in expiries if to_date(e) >= today]
    pool = upcoming if upcoming else expiries
    return min(pool, key=lambda e: (abs((to_date(e) - today).days), to_date(e)))


def _nearest_strike(strikes: list[int], price: float):
    if not math.isfinite(price) or not strikes:
        return None
    pos = np.searchsorted(strikes, price)
    if pos == 0:
        return strikes[0]
    if pos >= len(strikes):
        return strikes[-1]
    low, high = strikes[pos - 1], strikes[pos]
    return high if (price - low) >= (high - price) else low  # tie -> higher


def reference_day(day_dir: Path, underlier: str, d: date):
    """Independent single-day simulation. Returns (trades, day_pnl)."""
    grid = _grid(d)
    n = len(grid)
    futures = _read_series(day_dir / FUTURES_DIR / f"{underlier}-I.csv", grid)

    expiries = _discover(day_dir, underlier)
    exp = _nearest_expiry(list(expiries), d)
    strikes = sorted(expiries[exp]["CE"] & expiries[exp]["PE"])

    opt_cache: dict[str, np.ndarray] = {}

    def series(strike: int, opt: str) -> np.ndarray:
        sym = f"{underlier}{exp}{strike}{opt}"
        if sym not in opt_cache:
            opt_cache[sym] = _read_series(day_dir / OPTIONS_DIR / f"{sym}.csv", grid)
        return opt_cache[sym]

    def sym_of(strike, opt):
        return f"{underlier}{exp}{strike}{opt}"

    held = None
    pos: dict[str, int] = {}
    cash = 0.0
    trades = []

    def trade(ts, sym, q, px):
        nonlocal cash
        cash += -q * px
        pos[sym] = pos.get(sym, 0) + q
        trades.append((ts, sym, q, round(px, 6)))

    for i in range(n):
        ts = grid[i]
        target = _nearest_strike(strikes, futures[i])
        if target is not None and target != held:
            ce, pe = series(target, "CE")[i], series(target, "PE")[i]
            if math.isfinite(ce) and math.isfinite(pe):
                if held is not None:
                    for opt in ("CE", "PE"):
                        s = sym_of(held, opt)
                        if pos.get(s, 0):
                            trade(ts, s, -pos[s], series(held, opt)[i])
                trade(ts, sym_of(target, "CE"), 1, ce)
                trade(ts, sym_of(target, "PE"), 1, pe)
                held = target

        if i == n - 1:  # end-of-day flatten
            for s in list(pos):
                if pos[s]:
                    strike = int(re.match(_OPT_RE_TMPL.format(u=underlier), s + ".csv").group(2))
                    opt = s[-2:]
                    px = series(strike, opt)[i]
                    if math.isfinite(px):
                        trade(ts, s, -pos[s], px)

    return trades, cash


def reference_backtest(dataset: Dataset, underlier: str, dates):
    all_trades = []
    pnl = 0.0
    for d in dates:
        trades, day_pnl = reference_day(dataset.day_dir(d), underlier, d)
        all_trades.extend(trades)
        pnl += day_pnl
    return all_trades, pnl


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def _engine_trades(result):
    rows = []
    for _, t in result.trades.iterrows():
        rows.append((t["timestamp"].to_pydatetime(), t["symbol"],
                     int(t["signed_qty"]), round(float(t["price"]), 6)))
    return rows


def test_loader_matches_independent_reader():
    """The package's pandas loader must equal a stdlib-only reader, instrument
    by instrument -- so the shared data path in `reconcile` is itself trusted."""
    from mft.data import load_price_series, session_index

    ds = Dataset(DATA_ROOT)
    d = ds.dates[0]
    grid = _grid(d)
    idx = session_index(d)
    day_dir = ds.day_dir(d)

    samples = [day_dir / FUTURES_DIR / "NIFTY-I.csv",
               day_dir / FUTURES_DIR / "BANKNIFTY-I.csv"]
    # add a few traded option legs discovered from a real engine run
    eng = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=[d])
    for sym in eng.trades["symbol"].unique()[:5]:
        samples.append(day_dir / OPTIONS_DIR / f"{sym}.csv")

    for path in samples:
        ref = _read_series(path, grid)
        pkg = load_price_series(path, idx)
        both_nan = np.isnan(ref) & np.isnan(pkg)
        assert np.allclose(ref[~both_nan], pkg[~both_nan], atol=0, rtol=0), path.name
        assert (np.isnan(ref) == np.isnan(pkg)).all(), f"NaN mask mismatch {path.name}"
    print("loader matches stdlib reader on", len(samples), "instruments")


def test_reference_matches_engine():
    """The independent oracle must reproduce the engine fill-for-fill and PnL."""
    ds = Dataset(DATA_ROOT)
    for underlier in ("NIFTY", "BANKNIFTY"):
        dates = ds.dates[:3]
        eng = BacktestEngine(ds).run(NearestStraddle(), underlier, dates=dates)
        ref_trades, ref_pnl = reference_backtest(ds, underlier, dates)
        eng_trades = _engine_trades(eng)

        assert len(ref_trades) == len(eng_trades), (
            f"{underlier}: trade count {len(ref_trades)} vs {len(eng_trades)}")
        mismatches = [(r, e) for r, e in zip(ref_trades, eng_trades) if r != e]
        assert not mismatches, f"{underlier}: first mismatch {mismatches[0]}"
        assert abs(ref_pnl - eng.final_pnl) < 1e-9, (
            f"{underlier}: PnL {ref_pnl} vs {eng.final_pnl}")
        print(f"{underlier}: {len(eng_trades)} trades reproduced exactly, "
              f"PnL {eng.final_pnl:.4f} == {ref_pnl:.4f}")


def test_reconcile_passes_on_truth():
    ds = Dataset(DATA_ROOT)
    eng = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:3])
    rec = reconcile.reconcile(eng, ds)
    assert rec.passed(), rec.as_row()
    print("reconcile passes on untouched result:", rec.as_row())


def test_reconcile_has_teeth():
    """If the checks were vacuous they would pass on corrupted data. They must
    not: perturbing a price, dropping a fill, or corrupting the recorded curve
    each has to break reconciliation."""
    ds = Dataset(DATA_ROOT)
    eng = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
    assert reconcile.reconcile(eng, ds).passed()

    # 1) perturb a single fill price -> cash-flow identity must break
    bad = deepcopy(eng)
    bad.trades.loc[bad.trades.index[10], "price"] += 7.0
    r1 = reconcile.reconcile(bad, ds)
    assert not r1.passed(), "perturbed price slipped through"

    # 2) drop a fill entirely -> positions no longer net to flat
    bad2 = deepcopy(eng)
    bad2.trades = bad2.trades.drop(bad2.trades.index[5]).reset_index(drop=True)
    r2 = reconcile.reconcile(bad2, ds)
    assert not r2.passed(), "dropped fill slipped through"

    # 3) corrupt the engine's recorded equity curve -> rebuilt curve must diverge
    bad3 = deepcopy(eng)
    bad3.mtm.iloc[1000, bad3.mtm.columns.get_loc("equity")] += 50.0
    r3 = reconcile.reconcile(bad3, ds)
    assert r3.curve_max_abs_diff > 1.0, "curve corruption undetected"
    print("teeth confirmed: price/fill/curve corruptions all detected")


def test_invariants():
    """Structural guarantees, recomputed straight from the fills."""
    ds = Dataset(DATA_ROOT)
    eng = BacktestEngine(ds).run(NearestStraddle(), "BANKNIFTY", dates=ds.dates[:3])
    t = eng.trades

    # never exceed the position cap per symbol
    running = {}
    for _, f in t.iterrows():
        running[f["symbol"]] = running.get(f["symbol"], 0) + f["signed_qty"]
        assert abs(running[f["symbol"]]) <= 1
    # flat at the end of every day
    t = t.assign(day=t["timestamp"].dt.normalize())
    for day, grp in t.groupby("day"):
        net = grp.groupby("symbol")["signed_qty"].sum()
        assert (net == 0).all(), f"not flat on {day}: {net[net != 0].to_dict()}"
    # every roll-in adds exactly a CE and a PE on the same strike
    rolls = t[t["reason"] == "roll_in"]
    by_time = rolls.groupby("timestamp")["opt_type"].apply(lambda s: set(s))
    assert all(s == {"CE", "PE"} for s in by_time), "a roll did not enter a full straddle"
    print(f"invariants hold: {len(t)} fills, cap respected, flat daily, straddle complete")


if __name__ == "__main__":
    for fn in [test_loader_matches_independent_reader, test_reference_matches_engine,
               test_reconcile_passes_on_truth, test_reconcile_has_teeth, test_invariants]:
        print(f"\n=== {fn.__name__} ===")
        fn()
    print("\nALL INDEPENDENT CHECKS PASSED")
