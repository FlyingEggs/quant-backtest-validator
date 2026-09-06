"""Costs section (V3.2) — gate + net-PnL audit.

States:
  * no config['cost']                        -> NOT VERIFIED (gross PnL is not a claim)
  * config['cost'] + no per-trade trades_log -> DECLARED (assumptions recorded; the
    NET audit cannot run without per-trade fills -> NOT VERIFIED, never assumed)
  * config['cost'] + trades_log              -> VERIFIED (net engine ran; adverse fills,
    tick quantisation, per-side commission, spread/slippage/impact/financing)
"""

from __future__ import annotations

from typing import Dict, Optional

from validator import costengine
from validator.types import Strategy, run_metrics


def _instrument_dict(spec) -> Dict:
    """Pull the declared instrument contract off DataSpec (empty = not declared)."""
    if spec is None:
        return {}
    out = {}
    for k in ("qty_step", "min_qty", "min_notional", "contract_size"):
        v = getattr(spec, k, None)
        if v:
            out[k] = v
    return out


def _gate(cost: Dict, trades_log, spec=None,
          reported_pnl: Optional[float] = None) -> Dict:
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
    net_cfg = dict(cost)
    inst = _instrument_dict(spec)
    if inst:
        net_cfg["instrument"] = inst      # V3.6: DataSpec instrument contract
    net = costengine.net_audit(trades_log, net_cfg)
    # ---- V4.1 ledger integrity: the strategy's REPORTED pnl must equal the
    # gross PnL implied by its own per-trade ledger. A strategy may claim any
    # headline number while returning a tiny/empty ledger - two disconnected
    # worlds - so the ledger is validated against the claim, P0.
    if reported_pnl is not None:
        ledger_gross = net["gross_pnl"]
        tol = max(1e-6, abs(ledger_gross) * 1e-6)
        if abs(reported_pnl - ledger_gross) > tol:
            return {"status": "FAIL",
                    "issues": [{"code": "TRADE_LEDGER_PNL_MISMATCH",
                                "severity": "P0",
                                "finding": f"strategy reports pnl {reported_pnl:,.2f} "
                                           f"but its trades_log implies a gross "
                                           f"ledger PnL of {ledger_gross:,.2f} - "
                                           f"reported performance is disconnected "
                                           f"from the trade ledger; the ledger is "
                                           f"authoritative"}],
                    "notes": ["ledger-integrity check failed: headline PnL != "
                              "Σ(trade gross)"]}
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
    # net ran without invariant breaks; surface engine-level findings
    # (V3.6 EXEC_* P1 etc.) - they must move the section, not vanish.
    if any(i["severity"] == "P1" for i in net["issues"]):
        return {"status": "CONDITIONAL PASS", "issues": net["issues"],
                "notes": notes, "evidence": {"net": net}}
    return {"status": "VERIFIED", "issues": net["issues"], "notes": notes,
            "evidence": {"net": net}}


def net_check(strategy: Strategy, df, config: Dict, spec=None) -> Dict:
    """Audit-pipeline wrapper: pulls per-trade fills from the strategy run itself."""
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}
    res = run_metrics(strategy, df)
    reported = float(res.get("pnl", 0.0)) if res.get("pnl") is not None else None
    return _gate(cost, res.get("trades_log"), spec, reported)


def costs_check(config: Dict) -> Dict:
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}
    return _gate(cost, cost.get("trades_log"))
