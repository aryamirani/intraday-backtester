# Index-Option Backtester

A small, **strategy-agnostic** event-driven backtester for NSE index options, with a
nearest-ATM straddle strategy run over 21 trading days (Nov 2022) on NIFTY, BANKNIFTY, and FINNIFTY.

The design goal is separation of concerns: the engine never knows what the strategy is
doing. A strategy looks at the market each second and returns orders; the engine fills
them, enforces the position cap, tracks signed positions and realized/unrealized PnL, and
records a per-second mark-to-market curve plus a full trade log. Plugging in a new
strategy is a single method.

```
          observes market                      emits orders
  Strategy  ----------->  MarketSnapshot  ----------------->  BacktestEngine
                                                                    |
                                fills / marks / position cap        v
                                                               Portfolio  -->  MTM curve + trades
```

## The strategy (as specified)

Every second, pick the strike nearest the front-month futures (`*-I.csv`), hold 1x CE +
1x PE, and **roll** (sell the held pair, buy the new pair) whenever the nearest strike
changes. All positions are flattened at the close; days are processed serially and PnL is
cumulative across them. Profitability is explicitly *not* the objective — the point is to
measure the strategy faithfully. (The steady theta bleed of a perpetually-long straddle is
the expected, and observed, result.)

## Layout

| File | Responsibility |
|------|----------------|
| `mft/instruments.py` | Parse `UNDERLIER+YYMMDD+STRIKE+CE/PE` filenames. |
| `mft/data.py` | Tick CSV -> 1-second forward-filled price series; nearest-expiry selection; lazy per-strike loading (`DayMarket`). |
| `mft/core.py` | `Order`, `Fill`, `Position`, `MarketSnapshot`. |
| `mft/portfolio.py` | Signed positions, average cost, realized/unrealized PnL, configurable slippage/fees and lot size. |
| `mft/engine.py` | The strategy-agnostic 1-second event loop, day-end flatten, recording. |
| `mft/strategy.py` | `Strategy` interface + `NearestStraddle`. |
| `mft/analytics.py` | Metrics (drawdown, Sharpe-like, round-trips, hold time) and matplotlib helpers. |
| `mft/run.py` | Orchestrates the full multi-day run and writes `results/`. |
| `report.ipynb` | The narrative report: cumulative PnL, drawdown, daily PnL, position timeline, trade microstructure, and a "new strategy in ~10 lines" demo. |

## Run it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m mft.run                 # full 21-day NIFTY, BANKNIFTY, and FINNIFTY run -> results/
python -m mft.run --days 1        # 1-day smoke test
jupyter notebook report.ipynb     # the report
```

Useful flags: `--lot-size` (contract multiplier for a rupee view), `--slippage`,
`--fee-rate` (transaction costs), `--hysteresis` (optional anti-whipsaw band; default 0
gives the exact "always-nearest" behaviour).

## Design notes / assumptions

- **1-second grid.** Multiple ticks per second collapse to the last; gaps forward-fill the
  last traded price. A freshly-selected leg with no print yet is left untraded until it
  prints, so we never fill on a price that doesn't exist.
- **Tradable strikes** are those present as *both* a call and a put for the chosen expiry.
- **Costs** default to zero (frictionless) per the brief, but slippage, fees and lot size
  are first-class engine parameters, so a realistic rupee run is one flag away.
- **PnL is in index points** for a 1-unit position.
