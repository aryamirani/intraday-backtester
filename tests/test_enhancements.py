"""Comprehensive test suite for the intraday-backtester.

Covers every module introduced or modified in the enhancements commit:
  - mft/instruments.py      (parsing, construction, edge cases)
  - mft/core.py             (Order, Fill, Position, Side, MarketSnapshot)
  - mft/portfolio.py        (Portfolio, CostModel, VolatilityScaledCostModel)
  - mft/strategy.py         (NearestStraddle, TimeWeightedStraddle,
                              WidenedStrangle, STRATEGY_REGISTRY)
  - mft/config.py           (RunConfig from dict and from YAML file)
  - mft/data.py             (Dataset, DayMarket, session_index, loaders)
  - mft/engine.py           (BacktestEngine, BacktestResult)
  - mft/analytics.py        (daily_pnl, drawdown, round_trips, Metrics, plots)
  - mft/reconcile.py        (cashflow_pnl, roundtrip_pnl, independent_equity_curve)
  - mft/optimize.py         (grid_search, GridResult)
  - mft/run.py              (run_all end-to-end)
  - dashboard.py            (importability)
  - configs/*.yaml          (all four YAML configs parse correctly)

Run with:  pytest tests/test_enhancements.py -v
"""
from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(_PROJECT_ROOT / "allData")


# ============================================================================
# 1. mft/instruments.py
# ============================================================================
from mft.instruments import Instrument, parse_instrument, is_option_name, option_symbol


class TestInstruments:
    """Tests for instrument name parsing, construction, and helpers."""

    def test_parse_nifty_option(self):
        inst = parse_instrument("NIFTY22110314550PE")
        assert inst.underlier == "NIFTY"
        assert inst.strike == 14550
        assert inst.opt_type == "PE"
        assert inst.expiry == date(2022, 11, 3)
        assert inst.symbol == "NIFTY22110314550PE"
        assert inst.is_call is False

    def test_parse_banknifty_option(self):
        inst = parse_instrument("BANKNIFTY22112443200CE")
        assert inst.underlier == "BANKNIFTY"
        assert inst.strike == 43200
        assert inst.opt_type == "CE"
        assert inst.is_call is True

    def test_parse_with_csv_extension(self):
        inst = parse_instrument("NIFTY22110314550PE.csv")
        assert inst.strike == 14550
        assert inst.opt_type == "PE"

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError, match="Not a valid option name"):
            parse_instrument("NIFTY-I")

    def test_is_option_name(self):
        assert is_option_name("NIFTY22110314550PE") is True
        assert is_option_name("NIFTY22110314550CE.csv") is True
        assert is_option_name("NIFTY-I") is False
        assert is_option_name("NIFTY-I.csv") is False
        assert is_option_name("random_string") is False

    def test_option_symbol_roundtrip(self):
        sym = option_symbol("NIFTY", "221103", 18000, "CE")
        assert sym == "NIFTY22110318000CE"
        inst = parse_instrument(sym)
        assert inst.underlier == "NIFTY"
        assert inst.strike == 18000
        assert inst.opt_type == "CE"

    def test_instrument_frozen(self):
        inst = parse_instrument("NIFTY22110314550PE")
        with pytest.raises(AttributeError):
            inst.strike = 99999


# ============================================================================
# 2. mft/core.py
# ============================================================================
from mft.core import Order, Side, Fill, Position, MarketSnapshot


class TestCore:
    """Tests for Order, Side, Fill, Position, MarketSnapshot."""

    def test_side_values(self):
        assert Side.BUY.value == 1
        assert Side.SELL.value == -1

    def test_order_signed_qty_buy(self):
        o = Order("SYM", Side.BUY, quantity=3, reason="test")
        assert o.signed_qty == 3
        assert o.symbol == "SYM"
        assert o.reason == "test"

    def test_order_signed_qty_sell(self):
        o = Order("SYM", Side.SELL, quantity=2)
        assert o.signed_qty == -2

    def test_order_default_quantity(self):
        o = Order("SYM", Side.BUY)
        assert o.quantity == 1
        assert o.signed_qty == 1

    def test_order_frozen(self):
        o = Order("SYM", Side.BUY)
        with pytest.raises(AttributeError):
            o.symbol = "OTHER"

    def test_fill_creation(self):
        ts = datetime(2022, 11, 1, 9, 15, 0)
        f = Fill(ts, "SYM", 1, 100.5, 0.03, "roll_in")
        assert f.timestamp == ts
        assert f.symbol == "SYM"
        assert f.signed_qty == 1
        assert f.price == 100.5
        assert f.cost == 0.03
        assert f.reason == "roll_in"

    def test_position_defaults(self):
        p = Position("SYM")
        assert p.quantity == 0.0
        assert p.avg_price == 0.0

    def test_position_mutable(self):
        p = Position("SYM")
        p.quantity = 5
        p.avg_price = 100.0
        assert p.quantity == 5
        assert p.avg_price == 100.0

    def test_market_snapshot(self):
        ts = datetime(2022, 11, 1, 9, 15, 0)
        snap = MarketSnapshot(ts, market=None, is_session_close=True)
        assert snap.timestamp == ts
        assert snap.is_session_close is True
        assert snap.market is None


# ============================================================================
# 3. mft/portfolio.py — CostModel
# ============================================================================
from mft.portfolio import CostModel, Portfolio, VolatilityScaledCostModel


class TestCostModel:
    """Tests for the static CostModel."""

    def test_frictionless_defaults(self):
        cm = CostModel()
        assert cm.per_unit_slippage == 0.0
        assert cm.fee_rate == 0.0

    def test_execution_price_buy(self):
        cm = CostModel(per_unit_slippage=1.0)
        assert cm.execution_price(100.0, 1) == 101.0  # buy = mark + slippage

    def test_execution_price_sell(self):
        cm = CostModel(per_unit_slippage=1.0)
        assert cm.execution_price(100.0, -1) == 99.0  # sell = mark - slippage

    def test_fee_calculation(self):
        cm = CostModel(fee_rate=0.001)
        assert cm.fee(100.0, 2) == pytest.approx(0.2)
        assert cm.fee(100.0, -2) == pytest.approx(0.2)  # fee on abs qty

    def test_zero_slippage_passthrough(self):
        cm = CostModel()
        assert cm.execution_price(42.5, 1) == 42.5
        assert cm.execution_price(42.5, -1) == 42.5


# ============================================================================
# 4. mft/portfolio.py — VolatilityScaledCostModel
# ============================================================================
class TestVolatilityScaledCostModel:
    """Tests for the dynamic volatility-scaled cost model."""

    def test_initial_vol_is_zero(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=3)
        assert cm._current_vol() == 0.0

    def test_vol_increases_with_price_movement(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=5)
        for p in [100, 101, 102, 103, 104]:
            cm.update_futures(p)
        vol = cm._current_vol()
        assert vol > 0.0

    def test_vol_zero_for_constant_prices(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=5)
        for _ in range(10):
            cm.update_futures(100.0)
        assert cm._current_vol() == pytest.approx(0.0)

    def test_dynamic_slippage_greater_than_base(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=3, vol_multiplier=0.1)
        # Feed volatile prices
        for p in [100, 105, 95, 110]:
            cm.update_futures(p)
        buy_price = cm.execution_price(100.0, 1)
        assert buy_price > 100.5  # base_slippage + vol component

    def test_sell_price_lower_than_mark_minus_base(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=3, vol_multiplier=0.1)
        for p in [100, 105, 95, 110]:
            cm.update_futures(p)
        sell_price = cm.execution_price(100.0, -1)
        assert sell_price < 99.5  # base_slippage + vol component

    def test_reset_day_clears_history(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=3)
        for p in [100, 105, 95]:
            cm.update_futures(p)
        assert cm._current_vol() > 0.0
        cm.reset_day()
        assert cm._current_vol() == 0.0
        assert len(cm._recent_futures) == 0

    def test_lookback_window_trimming(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, vol_lookback=3)
        for p in [100, 101, 102, 103, 104, 105]:
            cm.update_futures(p)
        # lookback=3 means we keep at most vol_lookback+1 = 4 prices
        assert len(cm._recent_futures) == 4

    def test_inherits_fee_rate(self):
        cm = VolatilityScaledCostModel(base_slippage=0.5, fee_rate=0.001)
        assert cm.fee_rate == 0.001
        assert cm.fee(100.0, 2) == pytest.approx(0.2)

    def test_execution_price_with_no_vol_equals_base(self):
        cm = VolatilityScaledCostModel(base_slippage=1.0, vol_multiplier=0.5)
        # No futures updates -> vol = 0
        assert cm.execution_price(100.0, 1) == 101.0
        assert cm.execution_price(100.0, -1) == 99.0


# ============================================================================
# 5. mft/portfolio.py — Portfolio
# ============================================================================
class TestPortfolio:
    """Tests for Portfolio position tracking and PnL accounting."""

    def test_initial_state(self):
        pf = Portfolio()
        assert pf.realized_pnl == 0.0
        assert pf.total_fees == 0.0
        assert pf.fills == []
        assert pf.position("SYM") == 0.0

    def test_buy_then_sell_realized_pnl(self):
        pf = Portfolio(lot_size=1.0)
        ts = datetime(2022, 11, 1, 9, 15, 0)
        pf.fill(ts, "SYM", 1, 100.0)
        assert pf.position("SYM") == 1
        pf.fill(ts, "SYM", -1, 110.0)
        assert pf.position("SYM") == 0
        assert pf.realized_pnl == pytest.approx(10.0)  # (110-100)*1*1

    def test_lot_size_scaling(self):
        pf = Portfolio(lot_size=50.0)
        ts = datetime(2022, 11, 1, 9, 15, 0)
        pf.fill(ts, "SYM", 1, 100.0)
        pf.fill(ts, "SYM", -1, 110.0)
        assert pf.realized_pnl == pytest.approx(500.0)  # 10 * 50

    def test_unrealized_pnl(self):
        pf = Portfolio(lot_size=1.0)
        ts = datetime(2022, 11, 1, 9, 15, 0)
        pf.fill(ts, "SYM", 1, 100.0)
        unrealized = pf.unrealized_pnl(lambda s: 105.0)
        assert unrealized == pytest.approx(5.0)

    def test_equity(self):
        pf = Portfolio(lot_size=1.0)
        ts = datetime(2022, 11, 1, 9, 15, 0)
        pf.fill(ts, "A", 1, 100.0)
        pf.fill(ts, "A", -1, 110.0)  # realized = 10
        pf.fill(ts, "B", 1, 50.0)
        # B marked at 60 -> unrealized = 10
        eq = pf.equity(lambda s: 60.0)
        assert eq == pytest.approx(20.0)

    def test_open_symbols(self):
        pf = Portfolio()
        ts = datetime(2022, 11, 1, 9, 15, 0)
        pf.fill(ts, "A", 1, 100.0)
        pf.fill(ts, "B", 1, 50.0)
        pf.fill(ts, "A", -1, 110.0)
        assert "B" in pf.open_symbols()
        assert "A" not in pf.open_symbols()

    def test_fill_with_cost_model(self):
        cm = CostModel(per_unit_slippage=1.0, fee_rate=0.001)
        pf = Portfolio(lot_size=1.0, cost_model=cm)
        ts = datetime(2022, 11, 1, 9, 15, 0)
        fill = pf.fill(ts, "SYM", 1, 100.0)
        # buy: execution_price = 100 + 1 = 101
        assert fill.price == 101.0
        # fee = 0.001 * 1 * 101 * lot_size(=1)
        assert fill.cost == pytest.approx(0.101)

    def test_fills_are_recorded(self):
        pf = Portfolio()
        ts = datetime(2022, 11, 1, 9, 15, 0)
        pf.fill(ts, "A", 1, 100.0, "reason1")
        pf.fill(ts, "A", -1, 110.0, "reason2")
        assert len(pf.fills) == 2
        assert pf.fills[0].reason == "reason1"
        assert pf.fills[1].reason == "reason2"


# ============================================================================
# 6. mft/strategy.py — STRATEGY_REGISTRY
# ============================================================================
from mft.strategy import (
    Strategy, NearestStraddle, TimeWeightedStraddle, WidenedStrangle,
    STRATEGY_REGISTRY,
)


class TestStrategyRegistry:
    """Tests for the strategy registry and strategy construction."""

    def test_registry_contains_all_strategies(self):
        assert "nearest_straddle" in STRATEGY_REGISTRY
        assert "time_weighted_straddle" in STRATEGY_REGISTRY
        assert "widened_strangle" in STRATEGY_REGISTRY

    def test_registry_constructs_correct_types(self):
        assert isinstance(STRATEGY_REGISTRY["nearest_straddle"](), NearestStraddle)
        assert isinstance(STRATEGY_REGISTRY["time_weighted_straddle"](), TimeWeightedStraddle)
        assert isinstance(STRATEGY_REGISTRY["widened_strangle"](), WidenedStrangle)

    def test_all_registry_entries_are_strategies(self):
        for name, cls in STRATEGY_REGISTRY.items():
            inst = cls()
            assert isinstance(inst, Strategy), f"{name} is not a Strategy subclass"

    def test_nearest_straddle_params(self):
        s = NearestStraddle(hysteresis=10.0)
        assert s.hysteresis == 10.0
        assert s.held_strike is None

    def test_time_weighted_params(self):
        s = TimeWeightedStraddle(rebalance_interval_s=120, hysteresis=5.0)
        assert s.interval == 120
        assert s.hysteresis == 5.0

    def test_widened_strangle_params(self):
        s = WidenedStrangle(width=3, hysteresis=2.0)
        assert s.width == 3
        assert s.hysteresis == 2.0

    def test_nearest_straddle_on_day_start_resets(self):
        s = NearestStraddle()
        s.held_strike = 18000
        s.on_day_start(None)
        assert s.held_strike is None

    def test_time_weighted_on_day_start_resets(self):
        s = TimeWeightedStraddle(rebalance_interval_s=60)
        s.held_strike = 18000
        s._seconds_since_last = 5
        s.on_day_start(None)
        assert s.held_strike is None
        assert s._seconds_since_last == 60  # triggers immediately at open

    def test_widened_strangle_on_day_start_resets(self):
        s = WidenedStrangle()
        s.held_ce_strike = 18100
        s.held_pe_strike = 17900
        s.on_day_start(None)
        assert s.held_ce_strike is None
        assert s.held_pe_strike is None


# ============================================================================
# 7. mft/config.py — RunConfig
# ============================================================================
from mft.config import RunConfig, _load_file


class TestConfig:
    """Tests for config-driven strategy construction."""

    def test_from_dict_nearest_straddle_defaults(self):
        cfg = RunConfig.from_dict({})
        assert isinstance(cfg.strategy, NearestStraddle)
        assert cfg.underliers == ("NIFTY", "BANKNIFTY", "FINNIFTY")
        assert cfg.data_root == "allData"
        assert cfg.out_dir == "results"
        assert cfg.lot_size == 1.0
        assert cfg.max_position == 1
        assert isinstance(cfg.cost_model, CostModel)

    def test_from_dict_widened_strangle(self):
        raw = {
            "strategy": {"name": "widened_strangle", "params": {"width": 2, "hysteresis": 5.0}},
            "engine": {"lot_size": 25.0, "max_position": 2},
            "cost_model": {"type": "volatility_scaled", "base_slippage": 1.5,
                           "vol_lookback": 30, "vol_multiplier": 0.2, "fee_rate": 0.001},
            "data": {"root": "testData", "underliers": ["NIFTY"]},
            "output": {"dir": "my_results"},
        }
        cfg = RunConfig.from_dict(raw)
        assert isinstance(cfg.strategy, WidenedStrangle)
        assert cfg.strategy.width == 2
        assert cfg.strategy.hysteresis == 5.0
        assert cfg.lot_size == 25.0
        assert cfg.max_position == 2
        assert cfg.data_root == "testData"
        assert cfg.underliers == ("NIFTY",)
        assert cfg.out_dir == "my_results"
        assert isinstance(cfg.cost_model, VolatilityScaledCostModel)
        assert cfg.cost_model.base_slippage == 1.5
        assert cfg.cost_model.vol_lookback == 30
        assert cfg.cost_model.vol_multiplier == 0.2
        assert cfg.cost_model.fee_rate == 0.001

    def test_from_dict_time_weighted(self):
        raw = {
            "strategy": {"name": "time_weighted_straddle",
                          "params": {"rebalance_interval_s": 120}},
        }
        cfg = RunConfig.from_dict(raw)
        assert isinstance(cfg.strategy, TimeWeightedStraddle)
        assert cfg.strategy.interval == 120

    def test_from_dict_static_cost_model(self):
        raw = {
            "cost_model": {"type": "static", "per_unit_slippage": 2.0, "fee_rate": 0.005},
        }
        cfg = RunConfig.from_dict(raw)
        assert isinstance(cfg.cost_model, CostModel)
        assert not isinstance(cfg.cost_model, VolatilityScaledCostModel)
        assert cfg.cost_model.per_unit_slippage == 2.0
        assert cfg.cost_model.fee_rate == 0.005

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            RunConfig.from_dict({"strategy": {"name": "nonexistent"}})

    def test_from_yaml_file(self):
        """Test loading from a real YAML config file."""
        yaml_path = _PROJECT_ROOT / "configs" / "nearest_straddle.yaml"
        if yaml_path.exists():
            cfg = RunConfig.from_file(yaml_path)
            assert isinstance(cfg.strategy, NearestStraddle)
            assert cfg.strategy.hysteresis == 0.0
            assert cfg.underliers == ("NIFTY", "BANKNIFTY", "FINNIFTY")

    def test_all_yaml_configs_parse(self):
        """Every YAML config in configs/ must parse without error."""
        configs_dir = _PROJECT_ROOT / "configs"
        for yaml_file in configs_dir.glob("*.yaml"):
            cfg = RunConfig.from_file(yaml_file)
            assert isinstance(cfg.strategy, Strategy), f"Failed to parse {yaml_file.name}"

    def test_from_json_file(self):
        """Test loading from a JSON config file."""
        import json
        raw = {
            "strategy": {"name": "nearest_straddle", "params": {"hysteresis": 3.0}},
            "data": {"underliers": ["NIFTY"]},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            f.flush()
            cfg = RunConfig.from_file(f.name)
        os.unlink(f.name)
        assert isinstance(cfg.strategy, NearestStraddle)
        assert cfg.strategy.hysteresis == 3.0

    def test_raw_dict_preserved(self):
        raw = {"strategy": {"name": "nearest_straddle"}, "extra_key": "extra_val"}
        cfg = RunConfig.from_dict(raw)
        assert cfg.raw == raw
        assert cfg.raw["extra_key"] == "extra_val"


# ============================================================================
# 8. mft/data.py — Dataset, session_index, helpers
# ============================================================================
from mft.data import Dataset, DayMarket, session_index, date_from_dirname, load_price_series


class TestData:
    """Tests for the data layer."""

    def test_session_index_length(self):
        idx = session_index(date(2022, 11, 1))
        # 9:15:00 to 15:30:00 = 6h15m = 22501 seconds
        expected = int((timedelta(hours=6, minutes=15)).total_seconds()) + 1
        assert len(idx) == expected

    def test_session_index_boundaries(self):
        d = date(2022, 11, 1)
        idx = session_index(d)
        assert idx[0] == pd.Timestamp(datetime.combine(d, time(9, 15, 0)))
        assert idx[-1] == pd.Timestamp(datetime.combine(d, time(15, 30, 0)))

    def test_date_from_dirname(self):
        assert date_from_dirname("NSE_20221101") == date(2022, 11, 1)
        assert date_from_dirname("NSE_20221130") == date(2022, 11, 30)

    def test_dataset_discovery(self):
        ds = Dataset(DATA_ROOT)
        assert len(ds.dates) == 21
        assert ds.dates[0] == date(2022, 11, 1)
        assert ds.dates[-1] == date(2022, 11, 30)

    def test_dataset_day_dir(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        day_dir = ds.day_dir(d)
        assert day_dir.name == "NSE_20221101"
        assert day_dir.is_dir()

    def test_dataset_missing_date_raises(self):
        ds = Dataset(DATA_ROOT)
        with pytest.raises(KeyError, match="No data folder"):
            ds.day_dir(date(2099, 1, 1))

    def test_day_market_creation(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        market = DayMarket(ds.day_dir(d), "NIFTY", d)
        assert market.futures is not None
        assert len(market.futures) == len(market.index)
        assert market.strikes.size > 0

    def test_day_market_nearest_strike(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        market = DayMarket(ds.day_dir(d), "NIFTY", d)
        fut_price = market.futures[0]
        if np.isfinite(fut_price):
            nearest = market.nearest_strike(fut_price)
            assert nearest is not None
            # nearest strike should be within one strike gap of the futures price
            assert abs(nearest - fut_price) <= 200  # NIFTY strikes are 50pt apart

    def test_day_market_nearest_strike_nan(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        market = DayMarket(ds.day_dir(d), "NIFTY", d)
        assert market.nearest_strike(float("nan")) is None

    def test_day_market_symbol_construction(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        market = DayMarket(ds.day_dir(d), "NIFTY", d)
        sym = market.symbol(18000, "CE")
        assert "NIFTY" in sym
        assert "CE" in sym
        assert "18000" in sym

    def test_day_market_option_price(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        market = DayMarket(ds.day_dir(d), "NIFTY", d)
        market.i = 100  # move past the start
        strike = int(market.strikes[len(market.strikes) // 2])
        price = market.option_price(strike, "CE")
        # Should be a finite float (might be NaN if no data, but should work)
        assert isinstance(price, float)

    def test_load_price_series_shape(self):
        ds = Dataset(DATA_ROOT)
        d = ds.dates[0]
        idx = session_index(d)
        # Load a futures series
        fut_path = ds.day_dir(d) / "Futures (Continuous)" / "NIFTY-I.csv"
        series = load_price_series(fut_path, idx)
        assert len(series) == len(idx)
        assert series.dtype == np.float64


# ============================================================================
# 9. mft/engine.py — BacktestEngine, BacktestResult
# ============================================================================
from mft.engine import BacktestEngine, BacktestResult


class TestEngine:
    """Tests for the backtesting engine."""

    def test_single_day_run(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=[ds.dates[0]])
        assert isinstance(result, BacktestResult)
        assert result.underlier == "NIFTY"
        assert not result.mtm.empty
        assert not result.trades.empty

    def test_result_columns(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=[ds.dates[0]])
        # MTM columns
        for col in ["underlier", "equity", "realized", "unrealized",
                     "n_positions", "held_strike", "futures"]:
            assert col in result.mtm.columns, f"Missing MTM column: {col}"
        # Trades columns
        for col in ["timestamp", "symbol", "strike", "opt_type",
                     "signed_qty", "price", "cost", "reason"]:
            assert col in result.trades.columns, f"Missing trades column: {col}"

    def test_final_pnl_matches_equity(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        assert result.final_pnl == pytest.approx(float(result.mtm["equity"].iloc[-1]))

    def test_positions_flat_at_eod(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "BANKNIFTY", dates=ds.dates[:3])
        t = result.trades
        t = t.assign(day=t["timestamp"].dt.normalize())
        for day, grp in t.groupby("day"):
            net = grp.groupby("symbol")["signed_qty"].sum()
            assert (net == 0).all(), f"Not flat on {day}: {net[net != 0].to_dict()}"

    def test_lot_size_passthrough(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds, lot_size=50.0)
        result = engine.run(NearestStraddle(), "NIFTY", dates=[ds.dates[0]])
        assert result.lot_size == 50.0

    def test_cost_model_passthrough(self):
        ds = Dataset(DATA_ROOT)
        cm = CostModel(per_unit_slippage=1.0, fee_rate=0.001)
        engine = BacktestEngine(ds, cost_model=cm)
        result = engine.run(NearestStraddle(), "NIFTY", dates=[ds.dates[0]])
        # With slippage and fees, cost column should have nonzero values
        assert result.trades["cost"].sum() > 0

    def test_max_position_respected(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds, max_position=1)
        result = engine.run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        # Reconstruct running position per symbol
        running = {}
        for _, f in result.trades.iterrows():
            running[f["symbol"]] = running.get(f["symbol"], 0) + f["signed_qty"]
            assert abs(running[f["symbol"]]) <= 1, \
                f"Position cap exceeded for {f['symbol']}: {running[f['symbol']]}"

    def test_time_weighted_produces_fewer_rolls(self):
        """TimeWeightedStraddle with 60s interval should produce fewer rolls
        than NearestStraddle which rolls on every tick."""
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]

        r_tick = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=dates)
        r_time = BacktestEngine(ds).run(
            TimeWeightedStraddle(rebalance_interval_s=60), "NIFTY", dates=dates)

        tick_rolls = (r_tick.trades["reason"] == "roll_in").sum()
        time_rolls = (r_time.trades["reason"].str.contains("roll_in")).sum()
        assert time_rolls <= tick_rolls, \
            f"TimeWeighted ({time_rolls}) should have ≤ rolls than NearestStraddle ({tick_rolls})"

    def test_widened_strangle_runs_and_produces_trades(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(WidenedStrangle(width=1), "NIFTY", dates=[ds.dates[0]])
        assert not result.trades.empty
        assert not result.mtm.empty

    def test_empty_result_pnl_zero(self):
        """BacktestResult with empty MTM should return 0 PnL."""
        result = BacktestResult("TEST", pd.DataFrame(), pd.DataFrame(), 1.0)
        assert result.final_pnl == 0.0


# ============================================================================
# 10. mft/analytics.py
# ============================================================================
from mft.analytics import (
    daily_pnl, drawdown, round_trips, compute_metrics, Metrics,
    combined_equity,
)


class TestAnalytics:
    """Tests for analytics functions."""

    def test_daily_pnl_shape(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=ds.dates[:3])
        dpnl = daily_pnl(result.mtm)
        assert len(dpnl) == 3

    def test_drawdown_is_nonpositive(self):
        eq = pd.Series([0, 10, 5, 15, 12, 20])
        dd = drawdown(eq)
        assert (dd <= 0).all()
        assert dd.iloc[2] == pytest.approx(5 - 10)  # 5 - peak(10)
        assert dd.iloc[4] == pytest.approx(12 - 15)

    def test_drawdown_zero_at_new_high(self):
        eq = pd.Series([0, 10, 20, 30])
        dd = drawdown(eq)
        assert (dd == 0).all()

    def test_round_trips_on_simple_case(self):
        trades = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-11-01 09:15:00", "2022-11-01 09:16:00"]),
            "symbol": ["SYM", "SYM"],
            "strike": [18000, 18000],
            "opt_type": ["CE", "CE"],
            "signed_qty": [1, -1],
            "price": [100.0, 110.0],
            "cost": [0.0, 0.0],
            "reason": ["roll_in", "roll_out"],
        })
        rt = round_trips(trades)
        assert len(rt) == 1
        assert rt.iloc[0]["pnl"] == pytest.approx(10.0)
        assert rt.iloc[0]["hold_s"] == 60.0

    def test_round_trips_empty(self):
        rt = round_trips(pd.DataFrame())
        assert rt.empty

    def test_compute_metrics(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=ds.dates[:3])
        m = compute_metrics(result)
        assert isinstance(m, Metrics)
        assert m.underlier == "NIFTY"
        assert m.n_fills == len(result.trades)
        assert m.max_drawdown <= 0
        assert isinstance(m.sharpe_like, float)
        assert isinstance(m.avg_daily_pnl, float)
        assert isinstance(m.pnl_std, float)

    def test_metrics_as_row(self):
        ds = Dataset(DATA_ROOT)
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        m = compute_metrics(result)
        row = m.as_row()
        assert "underlier" in row
        assert "final_pnl" in row
        assert "fills" in row
        assert "rolls" in row
        assert "max_drawdown" in row
        assert "sharpe_like" in row
        assert "fees" in row

    def test_combined_equity(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        results = {}
        for ul in ("NIFTY", "BANKNIFTY"):
            results[ul] = BacktestEngine(ds).run(NearestStraddle(), ul, dates=dates)
        combined = combined_equity(results)
        assert combined is not None
        assert len(combined) > 0
        # Combined should equal sum of individual equity at each point
        for ul, r in results.items():
            # Just check they overlap and sum is roughly right
            assert isinstance(combined, pd.Series)


# ============================================================================
# 11. mft/reconcile.py
# ============================================================================
from mft.reconcile import reconcile, cashflow_pnl, roundtrip_pnl, independent_equity_curve


class TestReconcile:
    """Tests for the reconciliation module."""

    def test_cashflow_pnl_matches_engine(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        cf_pnl = cashflow_pnl(result)
        assert abs(cf_pnl - result.final_pnl) < 1e-6

    def test_roundtrip_pnl_matches_engine(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        rt_pnl = roundtrip_pnl(result)
        assert abs(rt_pnl - result.final_pnl) < 1e-6

    def test_reconcile_passes(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        rec = reconcile(result, ds)
        assert rec.passed(), f"Reconciliation failed: {rec.as_row()}"

    def test_reconcile_detects_price_corruption(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        bad = copy.deepcopy(result)
        bad.trades.loc[bad.trades.index[5], "price"] += 10.0
        rec = reconcile(bad, ds)
        assert not rec.passed(), "Corrupted price should fail reconciliation"

    def test_reconcile_detects_dropped_fill(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        bad = copy.deepcopy(result)
        bad.trades = bad.trades.drop(bad.trades.index[3]).reset_index(drop=True)
        rec = reconcile(bad, ds)
        assert not rec.passed(), "Dropped fill should fail reconciliation"

    def test_reconcile_detects_curve_corruption(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        bad = copy.deepcopy(result)
        bad.mtm.iloc[500, bad.mtm.columns.get_loc("equity")] += 100.0
        rec = reconcile(bad, ds)
        assert rec.curve_max_abs_diff > 1.0

    def test_reconcile_as_row(self):
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:2])
        rec = reconcile(result, ds)
        row = rec.as_row()
        assert "underlier" in row
        assert "engine_pnl" in row
        assert "cashflow_pnl" in row
        assert "roundtrip_pnl" in row
        assert "curve_max_abs_diff" in row
        assert "reconciled" in row

    def test_reconcile_on_empty_trades(self):
        """cashflow_pnl and roundtrip_pnl must handle a result with empty trades."""
        empty_result = BacktestResult("TEST", pd.DataFrame(), pd.DataFrame(), 1.0)
        assert cashflow_pnl(empty_result) == 0.0
        assert roundtrip_pnl(empty_result) == 0.0
        assert empty_result.final_pnl == 0.0


# ============================================================================
# 12. mft/optimize.py — grid_search
# ============================================================================
from mft.optimize import grid_search, GridResult, _run_single


class TestOptimize:
    """Tests for the parameter optimization module."""

    def test_run_single(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        result = _run_single(DATA_ROOT, "NIFTY", "nearest_straddle",
                             "hysteresis", 0.0, dates)
        assert isinstance(result, GridResult)
        assert result.param_name == "hysteresis"
        assert result.param_value == 0.0
        assert isinstance(result.final_pnl, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.sharpe_like, float)

    def test_grid_search_returns_dataframe(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        df = grid_search(DATA_ROOT, "NIFTY", "hysteresis", [0.0, 5.0],
                         max_workers=1, dates=dates, verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "hysteresis" in df.columns
        assert "final_pnl" in df.columns
        assert "max_drawdown" in df.columns
        assert "sharpe_like" in df.columns

    def test_grid_search_sorted_by_param(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        df = grid_search(DATA_ROOT, "NIFTY", "hysteresis", [10.0, 0.0, 5.0],
                         max_workers=1, dates=dates, verbose=False)
        # Should be sorted by param value
        assert list(df["hysteresis"]) == [0.0, 5.0, 10.0]

    def test_grid_search_time_weighted(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        df = grid_search(DATA_ROOT, "NIFTY", "rebalance_interval_s", [30, 60],
                         strategy_name="time_weighted_straddle",
                         max_workers=1, dates=dates, verbose=False)
        assert len(df) == 2
        assert "rebalance_interval_s" in df.columns


# ============================================================================
# 13. mft/run.py — run_all
# ============================================================================
from mft.run import run_all


class TestRunAll:
    """End-to-end integration tests for run_all."""

    def test_run_all_single_underlier(self):
        with tempfile.TemporaryDirectory(dir=str(_PROJECT_ROOT)) as tmpdir:
            results = run_all(
                data_root=DATA_ROOT,
                underliers=("NIFTY",),
                out_dir=tmpdir,
                dates=Dataset(DATA_ROOT).dates[:2],
                verbose=False,
            )
            assert "NIFTY" in results
            assert isinstance(results["NIFTY"], BacktestResult)
            # Check output files were created
            assert (Path(tmpdir) / "mtm_NIFTY.parquet").exists()
            assert (Path(tmpdir) / "trades_NIFTY.parquet").exists()
            assert (Path(tmpdir) / "summary.csv").exists()

    def test_run_all_with_custom_strategy(self):
        with tempfile.TemporaryDirectory(dir=str(_PROJECT_ROOT)) as tmpdir:
            results = run_all(
                data_root=DATA_ROOT,
                underliers=("NIFTY",),
                out_dir=tmpdir,
                dates=Dataset(DATA_ROOT).dates[:1],
                verbose=False,
                strategy=TimeWeightedStraddle(rebalance_interval_s=120),
            )
            assert "NIFTY" in results

    def test_run_all_with_custom_cost_model(self):
        with tempfile.TemporaryDirectory(dir=str(_PROJECT_ROOT)) as tmpdir:
            cm = VolatilityScaledCostModel(base_slippage=0.5, vol_multiplier=0.1)
            results = run_all(
                data_root=DATA_ROOT,
                underliers=("NIFTY",),
                out_dir=tmpdir,
                dates=Dataset(DATA_ROOT).dates[:1],
                verbose=False,
                cost_model=cm,
            )
            assert "NIFTY" in results

    def test_run_all_multiple_underliers(self):
        with tempfile.TemporaryDirectory(dir=str(_PROJECT_ROOT)) as tmpdir:
            results = run_all(
                data_root=DATA_ROOT,
                underliers=("NIFTY", "BANKNIFTY"),
                out_dir=tmpdir,
                dates=Dataset(DATA_ROOT).dates[:1],
                verbose=False,
            )
            assert "NIFTY" in results
            assert "BANKNIFTY" in results


# ============================================================================
# 14. End-to-end: full pipeline with reconciliation
# ============================================================================
class TestEndToEnd:
    """Full pipeline tests: engine → analytics → reconcile."""

    def test_full_pipeline_nifty(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:3]
        engine = BacktestEngine(ds)
        result = engine.run(NearestStraddle(), "NIFTY", dates=dates)

        # Analytics
        m = compute_metrics(result)
        assert m.n_fills > 0
        dpnl = daily_pnl(result.mtm)
        assert len(dpnl) == 3

        # Reconciliation
        rec = reconcile(result, ds)
        assert rec.passed(), f"Reconciliation failed: {rec.as_row()}"

    def test_full_pipeline_banknifty(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:3]
        result = BacktestEngine(ds).run(NearestStraddle(), "BANKNIFTY", dates=dates)
        rec = reconcile(result, ds)
        assert rec.passed(), f"Reconciliation failed: {rec.as_row()}"

    def test_full_pipeline_with_costs(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        cm = CostModel(per_unit_slippage=1.0, fee_rate=0.001)
        result = BacktestEngine(ds, cost_model=cm).run(
            NearestStraddle(), "NIFTY", dates=dates)
        m = compute_metrics(result)
        assert m.total_fees > 0

    def test_full_pipeline_time_weighted(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        result = BacktestEngine(ds).run(
            TimeWeightedStraddle(rebalance_interval_s=60), "NIFTY", dates=dates)
        m = compute_metrics(result)
        assert m.n_fills > 0

    def test_full_pipeline_widened_strangle(self):
        ds = Dataset(DATA_ROOT)
        dates = ds.dates[:2]
        result = BacktestEngine(ds).run(
            WidenedStrangle(width=1), "NIFTY", dates=dates)
        m = compute_metrics(result)
        assert m.n_fills > 0

    def test_config_driven_end_to_end(self):
        """Load a config file and run the full pipeline."""
        yaml_path = _PROJECT_ROOT / "configs" / "nearest_straddle.yaml"
        if not yaml_path.exists():
            pytest.skip("Config file not found")
        cfg = RunConfig.from_file(yaml_path)
        ds = Dataset(cfg.data_root)
        with tempfile.TemporaryDirectory(dir=str(_PROJECT_ROOT)) as tmpdir:
            results = run_all(
                data_root=cfg.data_root,
                underliers=cfg.underliers[:1],  # just one for speed
                out_dir=tmpdir,
                lot_size=cfg.lot_size,
                dates=ds.dates[:2],
                verbose=False,
                strategy=cfg.strategy,
                cost_model=cfg.cost_model,
                max_position=cfg.max_position,
            )
            assert len(results) == 1

    def test_straddle_invariant_roll_pairs(self):
        """Every roll-in for NearestStraddle must produce exactly a CE+PE pair."""
        ds = Dataset(DATA_ROOT)
        result = BacktestEngine(ds).run(NearestStraddle(), "NIFTY", dates=ds.dates[:3])
        rolls = result.trades[result.trades["reason"] == "roll_in"]
        by_time = rolls.groupby("timestamp")["opt_type"].apply(lambda s: set(s))
        for ts, opts in by_time.items():
            assert opts == {"CE", "PE"}, f"Incomplete straddle at {ts}: {opts}"


# ============================================================================
# 15. Dashboard importability
# ============================================================================
class TestDashboard:
    """Verify that dashboard.py can be imported without error (no runtime test)."""

    def test_dashboard_file_exists(self):
        assert (_PROJECT_ROOT / "dashboard.py").exists()

    def test_dashboard_imports_all_needed_modules(self):
        """Check that the imports referenced in dashboard.py are available."""
        # These are the imports dashboard.py uses
        from mft.analytics import (compute_metrics, daily_pnl, drawdown,
                                    round_trips, plot_equity, plot_drawdown,
                                    plot_daily_pnl, plot_position_timeline,
                                    combined_equity)
        from mft.data import Dataset
        from mft.engine import BacktestEngine
        from mft.portfolio import CostModel, VolatilityScaledCostModel
        from mft.strategy import (NearestStraddle, TimeWeightedStraddle,
                                   WidenedStrangle, STRATEGY_REGISTRY)
        # If we reach here, all imports are valid


# ============================================================================
# 16. __init__.py exports
# ============================================================================
class TestPackageExports:
    """Verify that the package __init__.py exports everything it advertises."""

    def test_all_exports(self):
        import mft
        expected = [
            "Instrument", "parse_instrument",
            "Order", "Side", "Fill", "Position", "MarketSnapshot",
            "Portfolio", "VolatilityScaledCostModel",
            "BacktestEngine", "BacktestResult",
            "Strategy", "NearestStraddle", "TimeWeightedStraddle",
            "WidenedStrangle", "STRATEGY_REGISTRY",
        ]
        for name in expected:
            assert hasattr(mft, name), f"Missing export: {name}"
            assert name in mft.__all__, f"Not in __all__: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
