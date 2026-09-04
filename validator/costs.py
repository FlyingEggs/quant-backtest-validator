"""Costs & MTF sections (V2) — honest NOT VERIFIED until the modules exist."""

from __future__ import annotations

from typing import Dict


def costs_check(config: Dict) -> Dict:
    """A real cost model (fee schedule, funding history, slippage) is a separate
    module. Until a cost config is supplied this section is NOT VERIFIED - never
    silently assumed clean."""
    cost = config.get("cost")
    if cost is None:
        return {"status": "NOT VERIFIED",
                "issues": [{"code": "COST_MODEL", "severity": "P4",
                            "finding": "no cost model supplied (fee/funding/slippage) - "
                                       "reported PnL is gross; treat as NOT VERIFIED"}],
                "notes": ["supply config['cost'] with fee/slippage/funding to verify"]}
    fee = float(cost.get("fee_bps", 0.0))
    slip = float(cost.get("slippage_bps", 0.0))
    if fee < 0 or slip < 0:
        return {"status": "FAIL",
                "issues": [{"code": "COST_NEGATIVE", "severity": "P0",
                            "finding": "cost config contains negative fee/slippage"}],
                "notes": []}
    return {"status": "PASS",
            "issues": [{"code": "COST_MODEL", "severity": "P3",
                        "finding": f"cost assumptions declared: fee {fee} bps/side, "
                                   f"slippage {slip} bps - not independently verified"}],
            "notes": ["costs declared but not yet independently re-run; module roadmap"]}


def mtf_check(config: Dict) -> Dict:
    return {"status": "NOT VERIFIED",
            "issues": [{"code": "MTF_MODULE", "severity": "P4",
                        "finding": "multi-timeframe alignment module is on the roadmap - "
                                   "MTF leakage not assessed"}],
            "notes": ["provide aligned MTF frames for the V3 module"]}
