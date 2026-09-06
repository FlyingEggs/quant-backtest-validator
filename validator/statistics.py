"""Statistics section (V2.1) — return independence / N_eff, graded not silent.

Rationale: autocorrelation does not make a strategy invalid - it makes its
*significance* claims weaker. So dependence is never a P0/P1 blocker at the overall
level, but a heavily dependent series can no longer get a silent PASS:

  N_eff / n >= 0.8   -> PASS (clean)
  0.5 .. 0.8         -> P3 note
  0.2 .. 0.5         -> P3 note (section PASS, significance flagged)
  < 0.2              -> P2 STAT_DEPENDENCE, section CONDITIONAL PASS

The issue + section status make the discount visible; the overall verdict is not
flipped by statistics alone (see module docstring).
"""

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

    issues, notes = [], []
    # ---- V4.1: rets must be the per-trade returns the strategy claims. If
    # trades=100 but len(rets)=50 (or rets are daily/aggregated bars), N_eff is
    # computed on the wrong sample - reported, never silently blessed. The
    # reference is the CLOSED ledger (trades_log) when present (a still-open
    # position is counted in `trades` but has no return yet), else the reported
    # trade count.
    n_rets = len(rets)
    ledger = res.get("trades_log")
    ref_n = len(ledger) if ledger is not None else int(res.get("trades", 0))
    if ref_n > 0 and n_rets != ref_n:
        src = "trades_log entries" if ledger is not None else "reported trades"
        issues.append({"code": "STAT_RETS_TRADE_MISMATCH", "severity": "P1",
                       "finding": f"strategy reports {ref_n} {src} but "
                                  f"len(rets)={n_rets} - rets are not one-per-trade; "
                                  f"N_eff below is computed on a mis-sized sample"})
        notes.append(f"return_unit assumed 'trade' but len(rets)={n_rets} != "
                     f"{src}={ref_n}; declare return_unit explicitly to verify")

    # ---- V4.2 red-team: rets must ALSO match the ledger's own per-trade returns.
    # Same length is not enough - a strategy can return len(rets)==trades while
    # rets values have nothing to do with the trades (e.g. 50% on a 10% move).
    if ledger is not None and n_rets == len(ledger):
        for i, (r, t) in enumerate(zip(rets, ledger)):
            ep, xp = t.get("entry_price"), t.get("exit_price")
            if ep is None or xp is None or float(ep) == 0.0:
                continue
            direction = 1.0 if t.get("side", "long") in ("long", "buy") else -1.0
            ledger_ret = (float(xp) - float(ep)) * direction / float(ep)
            if abs(float(r) - ledger_ret) > 1e-6:
                issues.append({"code": "STAT_RETS_LEDGER_MISMATCH",
                               "severity": "P1",
                               "finding": f"rets[{i}]={float(r):.6f} but the ledger "
                                          f"trade implies "
                                          f"{ledger_ret:+.6f} (entry {ep}, exit {xp}, "
                                          f"{t.get('side', 'long')}) - rets are not "
                                          f"the per-trade returns of this ledger"})
                break

    rep = core.return_independence(rets, verbose=False)
    if rep["verdict"] == "AUTOCORRELATED" and rep["n_eff"] is not None:
        r = rep["n_eff"] / rep["n"]
        notes.append(f"N_eff={rep['n_eff']} / n={rep['n']} (ratio {r:.2f}); overlapping "
                     f"trades deflate significance")
        if r < 0.2:
            issues.append({"code": "STAT_DEPENDENCE", "severity": "P2",
                           "finding": f"heavy return dependence: N_eff/n={r:.2f} - "
                           f"{rep['n']} trades behave like ~{rep['n_eff']:.0f}; any "
                           f"significance claim must be discounted"})
            status = "CONDITIONAL PASS"
        elif r < 0.8:
            issues.append({"code": "STAT_DEPENDENCE", "severity": "P3",
                           "finding": f"return dependence: N_eff/n={r:.2f} - significance "
                           f"claims need discounting"})
            status = "PASS"
        else:
            status = "PASS"
    else:
        status = "PASS"
        notes.append(f"N_eff={rep['n_eff']} / n={rep['n']}")
    if any(i["severity"] == "P1" for i in issues):
        status = "CONDITIONAL PASS"      # P1 findings must not read as a clean PASS
    return {"status": status, "issues": issues, "notes": notes, "evidence": rep}
