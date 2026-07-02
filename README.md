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

The `allData` folder (which is git-ignored) should be placed at the root of the project directory. The folder structure should look like this:

```text
intraday-backtester/
├── allData/
│   ├── NSE_20221101/
│   │   ├── Futures (Continuous)/
│   │   │   ├── BANKNIFTY-I.csv
│   │   │   ├── NIFTY-I.csv
│   │   │   └── ...
│   │   └── Options/
│   │       ├── BANKNIFTY22110339500CE.csv
│   │       ├── NIFTY22110317500PE.csv
│   │       └── ...
│   ├── NSE_20221102/
│   └── ...
├── mft/
├── configs/
├── tests/
├── dashboard.py
├── README.md
├── report.ipynb
└── ...
```

| File | Responsibility |
|------|----------------|
| `mft/instruments.py` | Parse `UNDERLIER+YYMMDD+STRIKE+CE/PE` filenames. |
| `mft/data.py` | Tick CSV -> 1-second forward-filled price series; nearest-expiry selection; lazy per-strike loading (`DayMarket`). |
| `mft/core.py` | `Order`, `Fill`, `Position`, `MarketSnapshot`. |
| `mft/portfolio.py` | Signed positions, average cost, realized/unrealized PnL, configurable slippage/fees, lot size, and **volatility-scaled cost model**. |
| `mft/engine.py` | The strategy-agnostic 1-second event loop, day-end flatten, recording. |
| `mft/strategy.py` | `Strategy` interface + `NearestStraddle` + `TimeWeightedStraddle` + `WidenedStrangle` + strategy registry. |
| `mft/config.py` | **YAML/JSON config loader** — instantiate any registered strategy and cost model from a config file. |
| `mft/optimize.py` | **Parallel parameter grid search** — sweep hysteresis (or any parameter) across N values using `ProcessPoolExecutor`. |
| `mft/analytics.py` | Metrics (drawdown, Sharpe-like, round-trips, hold time) and matplotlib helpers. |
| `mft/reconcile.py` | Recomputes PnL two ways independent of the engine (cash-flow identity + rebuilt per-second curve) and checks both agree to tolerance. |
| `mft/attribution.py` | Exact, Greeks-free split of straddle PnL into directional (intrinsic) vs time/volatility (theta) components. |
| `mft/run.py` | Orchestrates the full multi-day run; supports both CLI flags and `--config` for YAML-driven runs. |
| `dashboard.py` | **Interactive Streamlit dashboard** — real-time parameter tuning with instant visual feedback. |
| `report.ipynb` | The narrative report: reconciliation, cumulative PnL, drawdown, daily PnL, position timeline, trade microstructure, PnL attribution, and a "new strategy in ~10 lines" demo. |
| `tests/test_independent.py` | An independent oracle (its own CSV reader, resampling, strategy and accounting) that reproduces the engine fill-for-fill, plus mutation and invariant tests. |

## Run it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m mft.run                 # full 21-day NIFTY, BANKNIFTY, and FINNIFTY run -> results/
python -m mft.run --days 1        # 1-day smoke test
jupyter notebook report.ipynb     # the report
```

### Config-driven runs

Instead of CLI flags, pass a YAML config to run any registered strategy:

```bash
python -m mft.run --config configs/nearest_straddle.yaml
python -m mft.run --config configs/time_weighted.yaml
python -m mft.run --config configs/widened_strangle.yaml
python -m mft.run --config configs/realistic_costs.yaml
```

### Interactive dashboard

```bash
streamlit run dashboard.py
```

The Streamlit dashboard lets you:
- Select any strategy and tune its parameters with sliders
- Choose underliers, date range, and cost model interactively
- View equity curves, drawdown, daily PnL, position timelines, and trade logs — all updated in real time

### Parameter grid search

Sweep a parameter in parallel to find sensitivity:

```bash
python -m mft.optimize --underlier NIFTY --param hysteresis --values 0 5 10 20 50 --workers 4
```

Useful flags: `--lot-size` (contract multiplier for a rupee view), `--slippage`,
`--fee-rate` (transaction costs), `--hysteresis` (optional anti-whipsaw band; default 0
gives the exact "always-nearest" behaviour).

## Pluggable strategies

The engine is **strategy-agnostic**. Three strategies ship out of the box:

| Strategy | Description | Key parameter |
|----------|-------------|---------------|
| `NearestStraddle` | ATM straddle, roll on every strike change | `hysteresis` |
| `TimeWeightedStraddle` | ATM straddle, roll only every N seconds | `rebalance_interval_s` |
| `WidenedStrangle` | OTM strangle, legs offset N strikes from ATM | `width` |

Adding a new strategy is a single class implementing `on_step()`. Register it in `STRATEGY_REGISTRY` and it's immediately available in configs, the CLI, and the dashboard.

## Realistic execution modelling

Two cost models are available:

| Model | Description |
|-------|-------------|
| `CostModel` (static) | Fixed slippage + proportional fee. |
| `VolatilityScaledCostModel` | Dynamic slippage that widens with recent futures volatility, simulating real bid-ask spread behaviour. |

## Design notes / assumptions

- **1-second grid.** Multiple ticks per second collapse to the last; gaps forward-fill the
  last traded price. A freshly-selected leg with no print yet is left untraded until it
  prints, so we never fill on a price that doesn't exist.
- **Tradable strikes** are those present as *both* a call and a put for the chosen expiry.
- **Costs** default to zero (frictionless) per the brief, but slippage, fees and lot size
  are first-class engine parameters, so a realistic rupee run is one flag away.
- **PnL is in index points** for a 1-unit position.

## Commit history

**1. `assignment codebase`** — The full backtesting framework and the report. This commit
builds the strategy-agnostic, event-driven engine and everything it depends on: a data layer
that turns raw tick CSVs into 1-second forward-filled price series, selects the nearest expiry
and lazily loads only the strikes a run actually touches; a portfolio that tracks signed
positions, average cost and realized/unrealized PnL with configurable lot size, slippage and
fees; and the engine itself, which walks the session second by second, asks the strategy for
orders, fills them, enforces the position cap, flattens at the close and records a per-second
mark-to-market curve and a full trade log.

On top of that sits the `NearestStraddle` strategy (hold the ATM call and put, roll when the
nearest strike changes) and an analytics module that turns the recorded output into metrics
(drawdown, Sharpe-like, turnover, holding time) and plots. `run.py` drives the whole thing over
all 21 trading days for NIFTY, BANKNIFTY and FINNIFTY, and `report.ipynb` presents the results:
cumulative PnL, drawdown, daily PnL, the held-strike-vs-futures timeline, trade microstructure,
and a short demo showing a new strategy plugged into the same engine in a few lines.

**2. `independent verification and testing`** — Everything needed to *trust* the numbers.
This commit adds `reconcile.py`, which recomputes the engine's PnL two ways that share none of
its incremental bookkeeping — a single cash-flow identity over the trade log and a full
per-second equity curve rebuilt from the fills — and `attribution.py`, which decomposes the
straddle's PnL exactly (no Greeks) into a directional (intrinsic) component and a time/volatility
component, making the theta bleed explicit. Both are surfaced in the report.

It also adds `tests/test_independent.py`, a from-scratch reference implementation that shares no
code with the package: its own stdlib CSV reader, its own resampling, its own expiry/strike
selection, and its own strategy loop and accounting. It reproduces the engine fill-for-fill and
to floating-point PnL across multiple days and underliers, confirms the package's loader matches
an independent reader instrument by instrument, and includes mutation ("teeth") tests that corrupt
prices, drop fills and perturb the recorded curve to prove the reconciliation actually fails when
something is wrong — alongside structural invariant checks (position cap, flat at every close,
complete straddle on every roll).

Run the suite with `pytest -v` (or `python tests/test_independent.py`).

**3. `enhancements`** — Multi-strategy pluggability, config-driven runs,
interactive dashboard, parallel parameter optimization, and realistic execution modelling.
This commit adds two additional strategies (`TimeWeightedStraddle` and `WidenedStrangle`) that
plug into the engine with zero changes, proving agnosticism. A YAML/JSON config loader
(`mft/config.py`) lets strategies be instantiated from config files. A Streamlit dashboard
(`dashboard.py`) provides interactive parameter tuning with instant visual feedback. A parallel
parameter grid search (`mft/optimize.py`) sweeps any strategy parameter using
`ProcessPoolExecutor`. And a `VolatilityScaledCostModel` in `portfolio.py` dynamically adjusts
slippage based on recent futures volatility, simulating realistic market microstructure.

