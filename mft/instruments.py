from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

# UNDERLIER (letters) + YYMMDD expiry + STRIKE (digits) + CE/PE
_OPTION_RE = re.compile(r"^(?P<underlier>[A-Z]+?)(?P<expiry>\d{6})(?P<strike>\d+)(?P<opt>CE|PE)$")

# Continuous futures, e.g. NIFTY-I, BANKNIFTY-II
_FUTURE_RE = re.compile(r"^(?P<underlier>[A-Z]+)-(?P<series>I{1,3})$")


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    underlier: str
    expiry: date
    strike: int
    opt_type: str  # "CE" or "PE"

    @property
    def is_call(self) -> bool:
        return self.opt_type == "CE"


def _expiry_to_date(yymmdd: str) -> date:
    return date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))


def parse_instrument(name: str) -> Instrument:
    """Parse an option file/instrument name into its components.

    >>> parse_instrument("NIFTY22110314550PE").strike
    14550
    >>> parse_instrument("BANKNIFTY22112443200CE").underlier
    'BANKNIFTY'
    """
    stem = name[:-4] if name.endswith(".csv") else name
    m = _OPTION_RE.match(stem)
    if not m:
        raise ValueError(f"Not a valid option name: {name!r}")
    return Instrument(
        symbol=stem,
        underlier=m.group("underlier"),
        expiry=_expiry_to_date(m.group("expiry")),
        strike=int(m.group("strike")),
        opt_type=m.group("opt"),
    )


def is_option_name(name: str) -> bool:
    stem = name[:-4] if name.endswith(".csv") else name
    return _OPTION_RE.match(stem) is not None


def option_symbol(underlier: str, expiry: str, strike: int, opt_type: str) -> str:
    """Build the canonical instrument symbol from parts (expiry as YYMMDD)."""
    return f"{underlier}{expiry}{strike}{opt_type}"
