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

from typing import Dict, List

WEIGHTS = {"P0": 40, "P1": 15, "P2": 5, "P3": 2, "P4": 0}
RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def assemble_report(strategy_name: str, sections: Dict[str, Dict], config: Dict,
                    engine_version: str, scope: List[str]) -> Dict:
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

    return {
        "engine_version": engine_version,
        "strategy": strategy_name,
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
        "config_summary": {k: v for k, v in config.items()
                           if k in ("seed", "n_shuffles", "oos_frac",
                                    "expansion_confirmation")},
    }


def audit_report_text(report: Dict) -> str:
    L = []
    L.append("=" * 60)
    L.append("QUANT BACKTEST VALIDATION REPORT")
    L.append(f"Strategy : {report['strategy']}")
    L.append(f"Engine   : {report['engine_version']}")
    L.append("=" * 60)
    L.append(f"Overall Verdict : {report['overall']}")
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
