"""quant-backtest-validator — backtest-honesty engine + audit pipeline (V3).

Primitives (core) + orchestration layer (audit) + V3 MTF temporal-availability engine.
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
from validator import (data_integrity, execution, lookahead, statistics,
                       robustness, costs, mtf, costengine, wf, surface)

__version__ = "3.5.0"
__all__ = [
    # primitives
    "EXPANSION_HOURS", "full_audit", "lag_sensitivity", "period_expansion",
    "fill_timing_sensitivity", "randomized_control", "return_independence",
    # V2 / V2.2
    "audit", "audit_text", "ENGINE_VERSION",
    "Strategy", "DataSpec", "as_strategy", "as_code_strategy", "default_config",
    "data_integrity", "execution", "lookahead", "statistics", "robustness", "costs",
    # V3 MTF / V3.1 timeline / V3.2 cost engine
    "mtf", "costengine", "wf", "surface",
    # serialization
    "save_report", "to_jsonable",
    "__version__",
]
