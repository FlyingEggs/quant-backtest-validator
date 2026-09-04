"""quant-backtest-validator — reference implementation of backtest honesty checks."""

from validator.core import (
    EXPANSION_HOURS,
    full_audit,
    lag_sensitivity,
    period_expansion,
    fill_timing_sensitivity,
    randomized_control,
    return_independence,
    save_report,
    to_jsonable,
)

__version__ = "1.1.0"
__all__ = [
    "EXPANSION_HOURS",
    "full_audit",
    "lag_sensitivity",
    "period_expansion",
    "fill_timing_sensitivity",
    "randomized_control",
    "return_independence",
    "save_report",
    "to_jsonable",
    "__version__",
]
