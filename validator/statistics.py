"""Statistics section (V2) — return independence / N_eff (informational)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from validator import core
from validator.types import DataSpec, Strategy, run_metrics


def check(strategy: Strategy, df, spec: DataSpec, config: Dict) -> Dict:
    res = run_metrics(strategy, df)
    rets = res.get("rets")
    if rets is None or (isinstance(rets, (list, np.ndarray)) and len(rets) == 0):
        return {"status": "NOT VERIFIED", "issues": [
            {"code": "NO_RETS", "severity": "P4",
             "finding": "strategy returned no per-trade 'rets' - independence/N_eff "
                        "not assessable"}],
            "notes": ["return per-trade rets to enable N_eff"]}
    rep = core.return_independence(rets, verbose=False)
    issues, notes = [], []
    if rep["verdict"] == "AUTOCORRELATED":
        notes.append(f"N_eff={rep['n_eff']} < n={rep['n']} "
                     f"(inflation x{rep['inflation_factor']}); overlapping trades "
                     f"deflate significance")
    else:
        notes.append(f"N_eff={rep['n_eff']} / n={rep['n']}")
    return {"status": "PASS", "issues": issues,
            "notes": notes, "evidence": rep}
