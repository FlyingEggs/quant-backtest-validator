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
                       robustness, costs, mtf)
from validator import manifest as manifest_mod
from validator import report as report_mod
from validator.types import DataSpec, Strategy, default_config

ENGINE_VERSION = "4.0.0"

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
        "Costs": lambda: costs.net_check(strategy, df, cfg, spec),
        "MTF": lambda: mtf.check(df, spec, cfg),
    }
    return {name: builders[name]() for name in ALL_SECTIONS if name in scope}


def audit(strategy: Strategy, df: pd.DataFrame,
          data_spec: Optional[DataSpec] = None,
          config: Optional[Dict] = None) -> Dict:
    spec = data_spec if data_spec is not None else DataSpec()
    cfg = default_config()
    cfg.update(config or {})
    scope = [s for s in ALL_SECTIONS if s in cfg.get("scope", ALL_SECTIONS)]
    if not scope:
        raise ValueError(f"config['scope'] matched no known sections: "
                         f"{cfg.get('scope')!r} (known: {ALL_SECTIONS})")

    sections = _build_sections(strategy, df, spec, cfg, scope)
    rep = report_mod.assemble_report(strategy, sections, cfg, ENGINE_VERSION,
                                     scope, df)
    # V3.9: the audit input manifest is the reproducible evidence anchor -
    # replayable fingerprint of code + data + spec + config, independent of the
    # run-time audit_id.
    rep["manifest"] = manifest_mod.build_manifest(strategy, df, spec, cfg,
                                                  ENGINE_VERSION, scope)
    rep["manifest_hash"] = rep["manifest"]["manifest_hash"]
    # V4.0: chain the RESULT to the manifest - tampering with the report's
    # verdict/findings/metrics is detected by evidence_hash even when inputs are
    # untouched (hash only detects change; L7 signing remains on the roadmap).
    ev = manifest_mod.evidence_hash_from_report(rep)
    rep["result_hash"] = ev["result_hash"]
    rep["evidence_hash"] = ev["evidence_hash"]
    return rep


def audit_text(strategy: Strategy, df: pd.DataFrame,
               data_spec: Optional[DataSpec] = None,
               config: Optional[Dict] = None) -> str:
    return report_mod.audit_report_text(audit(strategy, df, data_spec, config))
