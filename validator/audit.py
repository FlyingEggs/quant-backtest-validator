"""audit() — the V2 one-call entry point.

    report = audit(strategy, df, data_spec=..., config=...)

Sections:
  Data Integrity  - frame sanity (always)
  Look-ahead      - lag/expansion (needs exposed signal column) else NOT VERIFIED
  Execution       - entry semantics + fill-timing perturbation (always)
  Statistics      - return independence / N_eff (needs per-trade rets) else NOT VERIFIED
  Robustness      - randomized control + chronological OOS + parameter sensitivity
  Costs           - NOT VERIFIED until a cost model is supplied
  MTF             - NOT VERIFIED (module on the roadmap)

Unimplemented capability is reported as NOT VERIFIED, never silently assumed clean.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from validator import data_integrity, execution, lookahead, statistics, robustness, costs
from validator import report as report_mod
from validator.types import DataSpec, Strategy, default_config

ENGINE_VERSION = "2.1.1"


def audit(strategy: Strategy, df: pd.DataFrame,
          data_spec: Optional[DataSpec] = None,
          config: Optional[Dict] = None) -> Dict:
    spec = data_spec if data_spec is not None else DataSpec()
    cfg = default_config()
    cfg.update(config or {})

    sections = {
        "Data Integrity": data_integrity.check(df, spec),
        "Look-ahead": lookahead.check(strategy, df, spec,
                                      cfg.get("expansion_confirmation")),
        "Execution": execution.check(strategy, df, spec, cfg),
        "Statistics": statistics.check(strategy, df, spec, cfg),
        "Robustness": robustness.check(strategy, df, spec, cfg),
        "Costs": costs.costs_check(cfg),
        "MTF": costs.mtf_check(cfg),
    }
    return report_mod.assemble_report(strategy.name, sections, cfg, ENGINE_VERSION)


def audit_text(strategy: Strategy, df: pd.DataFrame,
               data_spec: Optional[DataSpec] = None,
               config: Optional[Dict] = None) -> str:
    return report_mod.audit_report_text(audit(strategy, df, data_spec, config))
