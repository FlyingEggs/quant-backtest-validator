"""Report assembly & rendering (V2)."""

from __future__ import annotations

from typing import Dict, List

WEIGHTS = {"P0": 40, "P1": 15, "P2": 5, "P3": 2, "P4": 0}


def assemble_report(strategy_name: str, sections: Dict[str, Dict], config: Dict,
                    engine_version: str) -> Dict:
    issues: List[Dict] = []
    for sname, sec in sections.items():
        for i in sec.get("issues", []):
            item = dict(i)
            item["section"] = sname
            issues.append(item)
    issues_sorted = sorted(issues, key=lambda i: WEIGHTS.get(i.get("severity", "P4"), 0))

    has_p0 = any(i["severity"] == "P0" for i in issues)
    has_p1 = any(i["severity"] == "P1" for i in issues)
    overall = "FAIL" if has_p0 else ("CONDITIONAL PASS" if has_p1 else "PASS")

    penalty = sum(WEIGHTS.get(i["severity"], 0) for i in issues
                  if i["severity"] in ("P0", "P1", "P2", "P3"))
    score = max(0, 100 - penalty)
    not_verified = [name for name, sec in sections.items()
                    if sec["status"] == "NOT VERIFIED"]

    if has_p0:
        recommendation = ("DO NOT DEPLOY - close all P0 findings (execution look-ahead / "
                          "unconfirmed expansion / entry semantics / broken data) and "
                          "re-audit before relying on reported performance")
    elif has_p1:
        recommendation = ("CONDITIONAL - close the P1 items (manual construction/"
                          "execution-semantics review) before relying on reported "
                          "performance")
    else:
        recommendation = ("No blocking findings in the implemented checks. Treat "
                          "NOT VERIFIED sections as open before deployment decisions.")

    return {
        "engine_version": engine_version,
        "strategy": strategy_name,
        "overall": overall,
        "reliability_score": score,
        "sections": {k: {kk: vv for kk, vv in v.items()} for k, v in sections.items()},
        "issues": issues_sorted,
        "not_verified": not_verified,
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
    L.append(f"Reliability     : {report['reliability_score']}/100")
    L.append("-" * 60)
    for name, sec in report["sections"].items():
        L.append(f"{name:<18} {sec['status']}")
    if report["not_verified"]:
        L.append("-" * 60)
        L.append("NOT VERIFIED: " + ", ".join(report["not_verified"]))
    if report["issues"]:
        L.append("-" * 60)
        L.append("Findings (severity-ordered):")
        for i in report["issues"]:
            L.append(f"  [{i['severity']}] ({i['section']}) {i['code']}: {i['finding']}")
    L.append("-" * 60)
    L.append(f"Recommendation: {report['recommendation']}")
    L.append("=" * 60)
    return "\n".join(L)
