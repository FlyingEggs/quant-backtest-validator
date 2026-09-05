"""Costs section (V3.2) — gate + net-PnL audit.

States:
  * no config['cost']                        -> NOT VERIFIED (gross PnL is not a claim)
  * config['cost'] + no per-trade trades_log -> DECLARED (assumptions recorded; the
    NET audit cannot run without per-trade fills -> NOT VERIFIED, never assumed)
  * config['cost'] + trades_log              -> VERIFIED (net engine ran; adverse fills,
    tick quantisation, per-side commission, spread/slippage/impact/financing)
"""

from __future__ import annotations

from typing import Dict

from validator import costengine
from validator.types import Strategy, run_metrics


def _gate(cost: Dict, trades_log) -> Dict:
    """Shared tail: static config gate + net-engine status mapping.

    Status rules:
      * static negative cost param        -> FAIL (COST_NEGATIVE P0)
      * net-engine invariant broken       -> FAIL (COST_ENGINE_INVARIANT P0)
      * configured sub-model NOT VERIFIED -> NOT VERIFIED (e.g. financing asked for
        but trades lack timestamps) - a VERIFIED stamp with a dead sub-model would
        leak state; overall audit then reports INCOMPLETE, never VERIFIED
      * legacy/unconsumed keys present    -> P2 note (config that does nothing must
        not look verified)
    """
    issues, notes = [], []
    issues += costengine.validate_cost_config(cost)
    legacy = costengine._legacy_unconsumed_keys(cost)
    if legacy:
        issues.append({"code": "COST_CONFIG_UNUSED", "severity": "P2",
                       "finding": f"cost config key(s) {legacy} are not consumed by "
                                  f"the V3.2 net engine (legacy flat-bps "
                                  f"fee_bps/slippage_bps?) - they do nothing; use "
                                  f"commission/spread/slippage sections instead"})
    if issues:
        return {"status": "FAIL", "issues": issues,
                "notes": ["cost config rejected before the net engine ran"]}
    if not trades_log:
        return {"status": "DECLARED",
                "issues": [{"code": "COST_DECLARED", "severity": "P3",
                            "finding": "cost assumptions declared but per-trade fills "
                                       "(trades_log) were not supplied - net PnL "
                                       "audit NOT VERIFIED"}],
                "notes": ["declared, not net-audited - return trades_log from run() "
                          "to enable the V3.2 net engine"]}
    net = costengine.net_audit(trades_log, cost)
    notes.append(f"net PnL {net['net_pnl']:,.2f} vs gross {net['gross_pnl']:,.2f} "
                 f"(cost drag {net['cost_drag_pct']}%)")
    notes.append("sub-models: " + ", ".join(f"{k}={v}" for k, v in
                                            net["sub_models"].items()))
    if net["verdict"] == "FAIL":
        return {"status": "FAIL", "issues": net["issues"], "notes": notes}
    if net.get("declared_missing"):
        notes.append("declared sub-models NOT VERIFIED: "
                     + ", ".join(net["declared_missing"])
                     + " - net audit incomplete, not a clean VERIFIED")
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_SUB_INCOMPLETE", "severity": "P3",
                            "finding": f"configured sub-model(s) "
                                       f"{net['declared_missing']} could not be "
                                       f"verified (missing timestamps/fields) - "
                                       f"overall cost is NOT VERIFIED, never "
                                       f"VERIFIED with a dead sub-model"}],
                "notes": notes, "evidence": {"net": net}}
    return {"status": "VERIFIED", "issues": [], "notes": notes,
            "evidence": {"net": net}}


def net_check(strategy: Strategy, df, config: Dict) -> Dict:
    """Audit-pipeline wrapper: pulls per-trade fills from the strategy run itself."""
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}
    res = run_metrics(strategy, df)
    return _gate(cost, res.get("trades_log"))


def costs_check(config: Dict) -> Dict:
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}
    return _gate(cost, cost.get("trades_log"))
