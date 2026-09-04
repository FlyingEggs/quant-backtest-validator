"""quant-backtest-validator — backtest-honesty engine + audit pipeline (V2).

Primitives (core) and the V2 orchestration layer:
  audit(strategy, df, data_spec, config) -> full report
"""

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
from validator.audit import audit, audit_text, ENGINE_VERSION
from validator.types import (
    Strategy, DataSpec, as_strategy, as_code_strategy, default_config,
)
from validator import data_integrity, execution, lookahead, statistics, robustness, costs

__version__ = "2.1.0"
__all__ = [
    # primitives
    "EXPANSION_HOURS", "full_audit", "lag_sensitivity", "period_expansion",
    "fill_timing_sensitivity", "randomized_control", "return_independence",
    # V2
    "audit", "audit_text", "ENGINE_VERSION",
    "Strategy", "DataSpec", "as_strategy", "as_code_strategy", "default_config",
    "data_integrity", "execution", "lookahead", "statistics", "robustness", "costs",
    # serialization
    "save_report", "to_jsonable",
    "__version__",
]
