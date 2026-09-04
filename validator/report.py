"""Report assembly & rendering (V2.1).

Three-dimensional outcome, so an audit can never look "clean" merely because large
parts were NOT VERIFIED:

  * verified_score  - 100 - penalties, over the VERIFIED scope only
  * coverage_pct    - share of sections actually checked (NOT VERIFIED excluded)
  * blocking        - P0/P1/P2 counts
  * overall         - verdict on the verified scope (PASS/CONDITIONAL PASS/FAIL)
  * not_verified    - the un-checked remainder (INCOMPLETE when non-empty)

Unchecked != clean. The single headline number is gone.
"""

from __future__ import annotations

from typing import Dict, List

WEIGHTS = {"P0": 40, "P1": 15, "P2": 5, "P3": 2, "P4": 0}
RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def assemble_report(strategy_name: str, sections: Dict[str, Dict], config: Dict,
                    engine_version: str) -> Dict:
    issues: List[Dict] = []
    for sname, sec in sections.items():
        for i in sec.get("issues", []):
            item = dict(i)
            item["section"] = sname
            issues.append(item)
    # severity-ordered: P0 first, P4 last
    issues_sorted = sorted(issues, key=lambda i: RANK.get(i.get("severity", "P4"), 4))

    has_p0 = any(i["severity"] == "P0" for i in issues)
    has_p1 = any(i["severity"] == "P1" for i in issues)
    overall = "FAIL" if has_p0 else ("CONDITIONAL PASS" if has_p1 else "PASS")

    total = len(sections)
    not_verified = [name for name, sec in sections.items()
                    if sec["status"] == "NOT VERIFIED"]
    verified_n = total - len(not_verified)
    coverage_pct = round(100.0 * verified_n / total) if total else 0

    penalty = sum(WEIGHTS.get(i["severity"], 0) for i in issues
                  if i["severity"] in ("P0", "P1", "P2", "P3"))
    verified_score = max(0, 100 - penalty)
    blocking = {"P0": sum(1 for i in issues if i["severity"] == "P0"),
                "P1": sum(1 for i in issues if i["severity"] == "P1"),
                "P2": sum(1 for i in issues if i["severity"] == "P2")}

    incomplete = len(not_verified) > 0
    if has_p0:
        recommendation = ("DO NOT DEPLOY - close all P0 findings (execution look-ahead / "
                          "unconfirmed expansion / entry semantics / broken data) and "
                          "re-audit before relying on reported performance")
    elif has_p1:
        recommendation = ("CONDITIONAL - close the P1 items (manual construction/"
                          "execution-semantics review) before relying on reported "
                          "performance")
    elif incomplete:
        recommendation = (f"No blocking findings in the VERIFIED scope, but the audit is "
                          f"INCOMPLETE (coverage {coverage_pct}%; NOT VERIFIED: "
                          f"{', '.join(not_verified)}). Do not treat as full validation "
                          f"until those sections are checked.")
    else:
        recommendation = ("No blocking findings. Audit scope is complete for the modules "
                          "implemented in this engine version.")

    return {
        "engine_version": engine_version,
        "strategy": strategy_name,
        "overall": overall,
        "audit_complete": not incomplete,
        "verified_score": verified_score,
        "reliability_score": verified_score,          # deprecated alias, kept for compat
        "coverage_pct": coverage_pct,
        "not_verified": not_verified,
        "blocking": blocking,
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
    comp = "COMPLETE" if report["audit_complete"] else "INCOMPLETE"
    L.append(f"Overall Verdict : {report['overall']}   (audit {comp})")
    L.append(f"Verified Score  : {report['verified_score']}/100 "
             f"(over VERIFIED scope only)")
    L.append(f"Audit Coverage  : {report['coverage_pct']}%")
    b = report["blocking"]
    L.append(f"Blocking        : P0={b['P0']}  P1={b['P1']}  P2={b['P2']}")
    L.append("-" * 60)
    for name, sec in report["sections"].items():
        L.append(f"{name:<18} {sec['status']}")
    if report["not_verified"]:
        L.append("  NOT VERIFIED: " + ", ".join(report["not_verified"]))
    if report["issues"]:
        L.append("-" * 60)
        L.append("Findings (severity-ordered):")
        for i in report["issues"]:
            L.append(f"  [{i['severity']}] ({i['section']}) {i['code']}: {i['finding']}")
    L.append("-" * 60)
    L.append(f"Recommendation: {report['recommendation']}")
    L.append("=" * 60)
    return "\n".join(L)
