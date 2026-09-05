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


def net_check(strategy: Strategy, df, config: Dict) -> Dict:
    """Audit-pipeline wrapper: pulls per-trade fills from the strategy run itself."""
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}
    fee = float(cost.get("commission", {}).get("open_rate", 0.0))
    if fee < 0:
        return {"status": "FAIL",
                "issues": [{"code": "COST_NEGATIVE", "severity": "P0",
                            "finding": "cost config contains a negative commission rate"}],
                "notes": []}
    res = run_metrics(strategy, df)
    trades_log = res.get("trades_log")
    if not trades_log:
        return {"status": "DECLARED",
                "issues": [{"code": "COST_DECLARED", "severity": "P3",
                            "finding": "cost assumptions declared but the strategy did "
                                       "not return per-trade fills (trades_log) - net "
                                       "PnL audit NOT VERIFIED"}],
                "notes": ["declared, not net-audited - return trades_log from run() to "
                          "enable the V3.2 net engine"]}
    net = costengine.net_audit(trades_log, cost)
    return {"status": "VERIFIED", "issues": [],
            "notes": [f"net PnL {net['net_pnl']:,.2f} vs gross {net['gross_pnl']:,.2f} "
                      f"(cost drag {net['cost_drag_pct']}%)",
                      "sub-models: " + ", ".join(f"{k}={v}" for k, v in
                                                 net["sub_models"].items())],
            "evidence": {"net": net}}


def costs_check(config: Dict) -> Dict:
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}

    # validate the gate fields that exist regardless of trades_log
    fee = float(cost.get("commission", {}).get("open_rate", 0.0))
    if fee < 0:
        return {"status": "FAIL",
                "issues": [{"code": "COST_NEGATIVE", "severity": "P0",
                            "finding": "cost config contains a negative commission rate"}],
                "notes": []}

    trades_log = cost.get("trades_log")
    if not trades_log:
        return {"status": "DECLARED",
                "issues": [{"code": "COST_DECLARED", "severity": "P3",
                            "finding": "cost assumptions declared but per-trade fills "
                                       "(trades_log) were not supplied - net PnL audit "
                                       "NOT VERIFIED"}],
                "notes": ["declared, not net-audited - return trades_log from run() to "
                          "enable the V3.2 net engine"]}

    net = costengine.net_audit(trades_log, cost)
    return {"status": "VERIFIED", "issues": [],
            "notes": [f"net PnL {net['net_pnl']:,.2f} vs gross {net['gross_pnl']:,.2f} "
                      f"(cost drag {net['cost_drag_pct']}%)",
                      "sub-models: " + ", ".join(f"{k}={v}" for k, v in
                                                 net["sub_models"].items())],
            "evidence": {"net": net}}
