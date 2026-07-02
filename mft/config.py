"""Configuration-driven strategy construction.

Load a YAML or JSON config file to instantiate a Strategy, CostModel, and
run parameters without touching code. This is the 'plug different strategies
in easily' feature the brief asks for.

Example YAML::

    strategy:
      name: nearest_straddle
      params:
        hysteresis: 5.0

    engine:
      lot_size: 1.0
      max_position: 1

    cost_model:
      type: static              # or "volatility_scaled"
      per_unit_slippage: 0.0
      fee_rate: 0.0

    data:
      root: allData
      underliers: [NIFTY, BANKNIFTY]

    output:
      dir: results
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .portfolio import CostModel, VolatilityScaledCostModel
from .strategy import STRATEGY_REGISTRY, Strategy


def _load_file(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML configs. Install with: pip install pyyaml"
            )
    return json.loads(text)


@dataclass
class RunConfig:
    """Parsed run configuration."""

    strategy: Strategy
    underliers: tuple[str, ...]
    data_root: str
    out_dir: str
    lot_size: float
    max_position: int
    cost_model: CostModel
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        raw = _load_file(path)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, cfg: dict) -> "RunConfig":
        # Strategy
        strat_cfg = cfg.get("strategy", {})
        strat_name = strat_cfg.get("name", "nearest_straddle")
        strat_params = strat_cfg.get("params", {})
        if strat_name not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy {strat_name!r}. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        strategy = STRATEGY_REGISTRY[strat_name](**strat_params)

        # Cost model
        cost_cfg = cfg.get("cost_model", {})
        cost_type = cost_cfg.get("type", "static")
        if cost_type == "volatility_scaled":
            cost_model = VolatilityScaledCostModel(
                base_slippage=cost_cfg.get("base_slippage", 0.5),
                vol_lookback=cost_cfg.get("vol_lookback", 60),
                vol_multiplier=cost_cfg.get("vol_multiplier", 0.1),
                fee_rate=cost_cfg.get("fee_rate", 0.0),
            )
        else:
            cost_model = CostModel(
                per_unit_slippage=cost_cfg.get("per_unit_slippage", 0.0),
                fee_rate=cost_cfg.get("fee_rate", 0.0),
            )

        # Engine
        engine_cfg = cfg.get("engine", {})
        data_cfg = cfg.get("data", {})
        out_cfg = cfg.get("output", {})

        return cls(
            strategy=strategy,
            underliers=tuple(data_cfg.get("underliers", ["NIFTY", "BANKNIFTY", "FINNIFTY"])),
            data_root=data_cfg.get("root", "allData"),
            out_dir=out_cfg.get("dir", "results"),
            lot_size=engine_cfg.get("lot_size", 1.0),
            max_position=engine_cfg.get("max_position", 1),
            cost_model=cost_model,
            raw=cfg,
        )
