"""Report assembly & rendering (V2.2 — Audit Verdict Contract).

4-state verdict over the *declared audit scope*:

  FAIL               P0 finding present
  CONDITIONAL PASS   no P0, P1 present (manual confirmation needed)
  INCOMPLETE         no P0/P1, but some section is NOT VERIFIED
  PASS               no P0/P1 and every section in scope was verified

PASS therefore means "I checked the declared scope and it is clean" - never
"nothing is wrong anywhere". Statistical confidence is reported separately so a
PASS can never mask weak significance.
"""

from __future__ import annotations

import hashlib
import inspect
import secrets
import textwrap
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from validator.types import Strategy

WEIGHTS = {"P0": 40, "P1": 15, "P2": 5, "P3": 2, "P4": 0}
RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}

# Certification layers: what the engine can mechanically verify (max L4).
# L5 (adversarial suite as a service) / L6 (live parity) / L7 (signed immutable
# audits) are product-level and reported as NOT supported, never faked.
LAYERS = (
    ("L0", "STRUCTURAL", ["Data Integrity"]),
    ("L1", "TEMPORAL", ["Look-ahead", "MTF"]),
    ("L2", "EXECUTION", ["Execution"]),
    ("L3", "ECONOMIC", ["Costs"]),
    ("L4", "STATISTICAL", ["Statistics", "Robustness"]),
)


def _strategy_hash(strategy: Strategy) -> Optional[str]:
    """Source fingerprint of the strategy (name + description + run source)."""
    try:
        src = inspect.getsource(strategy.run)
    except (OSError, TypeError):
        src = None
    if src is None:
        return None                      # black-box / REPL / lambda run
    blob = f"{strategy.name}|{strategy.description}|{src}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _data_hash(df: pd.DataFrame) -> Optional[str]:
    """Full-frame fingerprint - EVERY column (OHLC, volume, signal, auxiliaries)
    plus the index - via the V3.9 canonical frame hash. Same OHLC with different
    volume or signal must NOT collide."""
    try:
        from validator.manifest import frame_hash
        return frame_hash(df)
    except Exception:
        return None


def certification_level(sections: Dict[str, Dict], scope: List[str]) -> Dict:
    """Highest CONTINUOUS layer whose every in-scope section is verified clean.

    A layer is certified only if all of its sections were audited (in scope),
    every one is PASS/VERIFIED, and no P0/P1 finding sits in the layer. A gap
    stops the climb - no skipping ahead to a higher layer.
    """
    level, reason = "NONE", None
    for lid, name, secs in LAYERS:
        missing = [s for s in secs if s not in scope]
        if missing:
            reason = f"{lid} {name}: section(s) {missing} out of scope"
            break
        layer_issues = [i for s in secs for i in sections.get(s, {}).get("issues", [])
                        if i.get("severity") in ("P0", "P1")]
        unclean = [s for s in secs if sections.get(s, {}).get("status")
                   not in ("PASS", "VERIFIED")]
        if unclean or layer_issues:
            reason = f"{lid} {name}: not clean " \
                     f"(status {unclean}, {len(layer_issues)} P0/P1)"
            break
        level = lid
    return {"level": level, "reason": reason,
            "max_supported_level": "L4",
            "signed": False,
            "signature_note": "immutable signed audits need a key infrastructure "
                              "(L7 on the product roadmap)"}


def assemble_report(strategy: Strategy, sections: Dict[str, Dict], config: Dict,
                    engine_version: str, scope: List[str],
                    df: Optional[pd.DataFrame] = None) -> Dict:
    issues: List[Dict] = []
    for sname, sec in sections.items():
        for i in sec.get("issues", []):
            item = dict(i)
            item["section"] = sname
            issues.append(item)
    issues_sorted = sorted(issues, key=lambda i: RANK.get(i.get("severity", "P4"), 4))

    p0 = any(i["severity"] == "P0" for i in issues)
    p1 = any(i["severity"] == "P1" for i in issues)
    verified_statuses = {"PASS", "CONDITIONAL PASS", "FAIL", "VERIFIED"}
    not_verified = [name for name, sec in sections.items()
                    if sec["status"] == "NOT VERIFIED"]
    declared = [name for name, sec in sections.items()
                if sec["status"] == "DECLARED"]
    uncovered = not_verified + declared

    # 4-state verdict contract (V2.2+): DECLARED is NOT verified - a section whose
    # capability was only declared (e.g. cost assumptions without a net audit) keeps
    # the audit INCOMPLETE, exactly like NOT VERIFIED.
    if p0:
        overall = "FAIL"
    elif p1:
        overall = "CONDITIONAL PASS"
    elif uncovered:
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    total = len(sections)
    verified_n = sum(1 for _, sec in sections.items()
                     if sec["status"] in verified_statuses)
    coverage_pct = round(100.0 * verified_n / total) if total else 0

    penalty = sum(WEIGHTS.get(i["severity"], 0) for i in issues
                  if i["severity"] in ("P0", "P1", "P2", "P3"))
    verified_score = max(0, 100 - penalty)
    blocking = {"P0": sum(1 for i in issues if i["severity"] == "P0"),
                "P1": sum(1 for i in issues if i["severity"] == "P1"),
                "P2": sum(1 for i in issues if i["severity"] == "P2")}

    # ---- statistical confidence: verdict != significance verdict --------------
    stat = sections.get("Statistics", {})
    stat_conf = {"significance_reliability": "NOT VERIFIED"}
    stat_ev = stat.get("evidence") or {}
    if stat_ev.get("n_eff") is not None and stat_ev.get("n"):
        ratio = stat_ev["n_eff"] / stat_ev["n"]
        stat_conf = {"significance_reliability": "DISCOUNTED" if ratio < 0.5
                     else "ADEQUATE",
                     "n_eff": stat_ev["n_eff"], "n": stat_ev["n"],
                     "ratio": round(ratio, 2)}
    elif stat.get("status") != "NOT VERIFIED":
        stat_conf = {"significance_reliability": "NOT ASSESSABLE"}

    if overall == "FAIL":
        recommendation = ("DO NOT DEPLOY - close all P0 findings (execution look-ahead / "
                          "unconfirmed expansion / entry semantics / broken data) and "
                          "re-audit before relying on reported performance")
    elif overall == "CONDITIONAL PASS":
        recommendation = ("CONDITIONAL - close the P1 items (manual construction/"
                          "execution-semantics review) before relying on reported "
                          "performance")
    elif overall == "INCOMPLETE":
        parts = []
        if not_verified:
            parts.append(f"NOT VERIFIED: {', '.join(not_verified)}")
        if declared:
            parts.append(f"DECLARED-but-not-verified: {', '.join(declared)}")
        recommendation = (f"No blocking findings in the VERIFIED scope, but the audit is "
                          f"INCOMPLETE (coverage {coverage_pct}%; "
                          f"{'; '.join(parts)}). PASS is only granted when every scope "
                          f"item is VERIFIED (DECLARED != VERIFIED) - complete the scope "
                          f"before treating this as a clean bill.")
    else:
        recommendation = ("PASS within the declared scope: every scope item was verified "
                          "and no P0/P1 finding was made. (Scope: " +
                          ", ".join(scope) + ".)")

    cert = certification_level(sections, scope)
    strat_hash = _strategy_hash(strategy)
    data_hash = _data_hash(df) if df is not None else None
    return {
        "engine_version": engine_version,
        "strategy": strategy.name,
        "scope": list(scope),
        "overall": overall,
        "audit_complete": overall == "PASS",
        "verified_score": verified_score,
        "reliability_score": verified_score,          # deprecated alias
        "coverage_pct": coverage_pct,
        "not_verified": not_verified,
        "declared": declared,
        "blocking": blocking,
        "statistical_confidence": stat_conf,
        "sections": {k: {kk: vv for kk, vv in v.items()} for k, v in sections.items()},
        "issues": issues_sorted,
        "recommendation": recommendation,
        "certification": {
            "audit_id": f"QBV-{datetime.now(timezone.utc):%Y%m%d}-"
                        f"{secrets.token_hex(4)}",
            "generated_at": datetime.now(timezone.utc)
                            .isoformat(timespec="seconds"),
            "strategy_hash": strat_hash,
            "strategy_hash_note": (None if strat_hash else
                                   "black-box run: source fingerprint unavailable"),
            "data_hash": data_hash,
            "data_hash_note": "full-frame fingerprint (every column + index)" if df is not None
                              else "no frame supplied",
            **cert,
        },
        "config_summary": {k: v for k, v in config.items()
                           if k in ("seed", "n_shuffles", "oos_frac",
                                    "expansion_confirmation")},
    }


def _interpretation(rep: Dict) -> str:
    """One human-readable line under the verdict: PASS is scoped, not a guarantee.

    PASS means "no cheating evidence in the dimensions we actually checked" - the
    report must say so at the top, because clients read the verdict first and the
    scope caveat second.
    """
    overall = rep["overall"]
    if overall == "FAIL":
        return ("A blocking defect was found in the checked scope - do not deploy "
                "on this evidence.")
    if overall == "CONDITIONAL PASS":
        return ("No hard defect, but P1 findings need manual confirmation before "
                "the reported performance is relied on.")
    if overall == "INCOMPLETE":
        return ("No defect in the verified scope, but key dimensions were not "
                "verified - missing evidence is not a clean bill.")
    return ("No evidence of cheating was found in the checked dimensions. Audit "
            "is not a guarantee of live performance - see Limitations.")


def audit_report_text(report: Dict) -> str:
    L = []
    L.append("=" * 60)
    L.append("QUANT BACKTEST VALIDATION REPORT")
    L.append(f"Strategy : {report['strategy']}")
    L.append(f"Engine   : {report['engine_version']}")
    L.append("=" * 60)
    L.append(f"Overall Verdict : {report['overall']}")
    interp = textwrap.wrap(_interpretation(report), width=62)
    L.append(f"Interpretation  : {interp[0]}" if interp else "Interpretation  :")
    for more in interp[1:]:
        L.append(f"                  {more}")
    cert = report.get("certification") or {}
    if cert.get("level") not in (None, "NONE"):
        L.append(f"Certified       : {cert['level']} of {cert['max_supported_level']} "
                 f"(continuous verified layers; L5-L7 = product roadmap)")
    else:
        stop = cert.get("reason") or "first layer not fully verified"
        L.append(f"Certified       : NO - {stop}")
    L.append(f"Verified Score  : {report['verified_score']}/100 "
             f"(over VERIFIED scope only)")
    L.append(f"Audit Coverage  : {report['coverage_pct']}%")
    b = report["blocking"]
    L.append(f"Blocking        : P0={b['P0']}  P1={b['P1']}  P2={b['P2']}")
    sc = report.get("statistical_confidence", {})
    if sc.get("significance_reliability") == "DISCOUNTED":
        L.append(f"Significance    : DISCOUNTED (N_eff {sc['n_eff']} / {sc['n']}, "
                 f"ratio {sc['ratio']}) - verdict != significance verdict")
    elif sc.get("significance_reliability") == "ADEQUATE":
        L.append(f"Significance    : ADEQUATE (N_eff {sc['n_eff']} / {sc['n']})")
    L.append("-" * 60)
    L.append("AUDIT SCOPE")
    for name, sec in report["sections"].items():
        mark = "△" if sec["status"] in ("NOT VERIFIED", "DECLARED") else "✓"
        L.append(f"  {mark} {name:<16} {sec['status']}")
    # V3.4 readability: surface / clustering rendered as their own lines
    ev = (report["sections"].get("Robustness") or {}).get("evidence") or {}
    if ev.get("surface"):
        sa = ev["surface"]
        L.append(f"Parameter Surface : {sa.get('verdict', 'n/a'):<8} best "
                 f"{sa.get('best_pnl')} plateau {sa.get('plateau_frac')} "
                 f"(isolated {sa.get('isolated_best')})")
    if ev.get("cluster"):
        ca = ev["cluster"]
        L.append(f"Trade Clustering  : {ca.get('verdict', 'n/a'):<8} "
                 f"{ca.get('raw_trades')} trades / {ca.get('active_days')} days")
    if report["issues"]:
        L.append("-" * 60)
        L.append("Findings (severity-ordered):")
        for i in report["issues"]:
            L.append(f"  [{i['severity']}] ({i['section']}) {i['code']}: {i['finding']}")
    L.append("-" * 60)
    L.append(f"Recommendation: {report['recommendation']}")
    L.append("=" * 60)
    return "\n".join(L)
