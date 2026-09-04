"""audit() — the V2.2 one-call entry point with a declared scope.

    report = audit(strategy, df, data_spec=..., config=...)

Sections: Data Integrity · Look-ahead · Execution · Statistics · Robustness ·
Costs · MTF. `config['scope']` (optional) restricts the run to named sections;
the default scope is everything the engine knows about. PASS is only granted when
every section IN SCOPE is verified and clean — out-of-scope or not-implemented
capability makes the verdict INCOMPLETE, never a silent PASS.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from validator import (data_integrity, execution, lookahead, statistics,
                       robustness, costs)
from validator import report as report_mod
from validator.types import DataSpec, Strategy, default_config

ENGINE_VERSION = "2.2.0"

ALL_SECTIONS = ["Data Integrity", "Look-ahead", "Execution", "Statistics",
                "Robustness", "Costs", "MTF"]


def _build_sections(strategy, df, spec, cfg, scope: List[str]) -> Dict[str, Dict]:
    builders = {
        "Data Integrity": lambda: data_integrity.check(df, spec),
        "Look-ahead": lambda: lookahead.check(strategy, df, spec,
                                              cfg.get("expansion_confirmation")),
        "Execution": lambda: execution.check(strategy, df, spec, cfg),
        "Statistics": lambda: statistics.check(strategy, df, spec, cfg),
        "Robustness": lambda: robustness.check(strategy, df, spec, cfg),
        "Costs": lambda: costs.costs_check(cfg),
        "MTF": lambda: costs.mtf_check(cfg),
    }
    return {name: builders[name]() for name in ALL_SECTIONS if name in scope}


def audit(strategy: Strategy, df: pd.DataFrame,
          data_spec: Optional[DataSpec] = None,
          config: Optional[Dict] = None) -> Dict:
    spec = data_spec if data_spec is not None else DataSpec()
    cfg = default_config()
    cfg.update(config or {})
    scope = [s for s in ALL_SECTIONS if s in cfg.get("scope", ALL_SECTIONS)]

    sections = _build_sections(strategy, df, spec, cfg, scope)
    return report_mod.assemble_report(strategy.name, sections, cfg, ENGINE_VERSION, scope)


def audit_text(strategy: Strategy, df: pd.DataFrame,
               data_spec: Optional[DataSpec] = None,
               config: Optional[Dict] = None) -> str:
    return report_mod.audit_report_text(audit(strategy, df, data_spec, config))
