"""V3.2 — Realistic Cost & Net PnL engine.

Adversarial execution price model: every fill is adjusted AGAINST the trader, then
tick-quantised against the trader (BUY rounds up, SELL rounds down), so a cost model
can never accidentally improve fills.

Layers (each reported as a separate drag):
  tick_size      price cannot be a non-quantised value (BUY up / SELL down)
  spread         fixed | pct  (half-spread per fill, adverse)
  slippage       bps | pct | callable   (adverse only; separate from spread)
  market_impact  none | linear | sqrt | custom callable(price, qty, side) (interface-first)
  commission     open_rate != close_rate supported (amount/notional/per-contract/fixed)
  financing      funding_bps_per_day over held days (needs per-trade exit_ts)

Pure over per-trade rows (no cross-row / no future look-up) - Case-5 guarantee.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

# V3.2+ cost config top-level keys the net engine actually consumes. Anything else
# (e.g. legacy flat-bps fee_bps / slippage_bps) is reported COST_CONFIG_UNUSED -
# a config that silently does nothing must never look verified.
CONSUMED_COST_KEYS = ("commission", "spread", "slippage", "market_impact",
                      "financing", "tick_size")


def validate_cost_config(cost: Dict) -> List[Dict]:
    """Static gate: cost-type parameters must be >= 0. A negative spread/slippage/
    impact/commission rate would PAY the trader - a config error, never a model.

    Financing is deliberately EXEMPT: real funding rates go negative in some
    regimes (longs are paid to hold), so a negative funding parameter is market
    semantics, not a cheat. Callable modes cannot be judged here - the runtime
    invariant in net_audit() rejects a callable that returns a negative cost.
    """
    issues = []
    checks = (("commission", ("open_rate", "close_rate")),
              ("spread", ("value", "value_pct", "value_bps")),
              ("slippage", ("value", "value_pct", "value_bps")),
              ("market_impact", ("coeff",)))
    for seg_name, fields in checks:
        seg = cost.get(seg_name) or {}
        if seg.get("mode", "none") == "none" or seg.get("mode") == "callable":
            continue
        for f in fields:
            if f in seg and float(seg[f]) < 0.0:
                issues.append({"code": "COST_NEGATIVE", "severity": "P0",
                               "finding": f"cost config {seg_name}.{f} = "
                                          f"{seg[f]} < 0 - a negative cost pays the "
                                          f"trader; rejected, never VERIFIED"})
    return issues


def _legacy_unconsumed_keys(cost: Dict) -> List[str]:
    # "trades_log" is a costs_check wrapper key (fills passed beside the model),
    # not a cost parameter - never flagged.
    return [k for k in cost if k not in CONSUMED_COST_KEYS and k != "trades_log"]


def _side_action(side: str, is_entry: bool) -> str:
    """Map a position side to the market action at entry/exit."""
    if side in ("long", "buy"):
        return "buy" if is_entry else "sell"
    return "sell" if is_entry else "buy"


def _adverse_dir(action: str) -> int:
    return 1 if action == "buy" else -1      # buy pays up, sell receives down


def _quantize(price: float, tick: Optional[float], action: str) -> float:
    if not tick or tick <= 0:
        return price
    if action == "buy":
        return math.ceil(price / tick - 1e-12) * tick
    return math.floor(price / tick + 1e-12) * tick


def _shift(action: str, base: float, magnitude: float) -> float:
    """Adverse price shift: buy pays base+magnitude, sell receives base-magnitude."""
    return base + _adverse_dir(action) * magnitude


def _shift_amount(cfg: dict, key: str, price: float, qty: float) -> float:
    """Per-fill adverse shift in PRICE units from a named config, or 0."""
    if not cfg or cfg.get("mode", "none") == "none":
        return 0.0
    mode = cfg.get("mode")
    if mode == "bps":
        return price * float(cfg.get("value_bps", 0.0)) / 1e4
    if mode == "pct":
        return price * float(cfg.get("value_pct", 0.0))
    if mode == "fixed":
        return float(cfg.get("value", 0.0))
    if mode == "callable":
        return float(cfg["fn"](price, qty, "spread" if key == "spread" else "slippage"))
    raise ValueError(f"unknown {key} mode {mode!r}")


def _impact_shift(cfg: dict, price: float, qty: float, action: str) -> float:
    if not cfg or cfg.get("mode", "none") == "none":
        return 0.0
    mode = cfg.get("mode")
    coeff = float(cfg.get("coeff", 0.0))
    if mode == "fixed":
        return coeff
    if mode == "linear":
        return coeff * abs(qty)
    if mode == "sqrt":
        return coeff * math.sqrt(abs(qty))
    if mode == "callable":
        return float(cfg["fn"](price, qty, action))
    raise ValueError(f"unknown market_impact mode {mode!r}")


def _commission(cfg: dict, notional: float, qty: float, is_entry: bool) -> float:
    if not cfg or cfg.get("mode", "none") == "none":
        return 0.0
    rate = float(cfg.get("open_rate", 0.0)) if is_entry else float(cfg.get("close_rate", 0.0))
    mode = cfg.get("mode")
    if mode == "bps":                     # per-side notional rate (bps)
        return notional * rate / 1e4
    if mode == "notional":                # fraction of notional
        return notional * rate
    if mode == "per_contract":
        return abs(qty) * rate
    if mode == "fixed":
        return rate
    raise ValueError(f"unknown commission mode {mode!r}")


def net_audit(trades_log: List[Dict], cfg: Dict) -> Dict:
    """Apply the cost model over per-trade rows. Pure: row order cannot matter.

    Verdict contract: FAIL + COST_ENGINE_INVARIANT (P0) if any configured cost
    layer produced a NEGATIVE charge on any fill - the engine's central promise is
    "every fill is adjusted AGAINST the trader", and that is enforced here as a
    runtime invariant (callable modes bypass the static config gate). Financing is
    the only layer where negative is legal (real negative funding regimes).
    """
    from validator.execution import _to_seconds   # shared ns/ms/s normalisation
    if not trades_log:
        return {"verdict": "NOT VERIFIED", "reason": "no per-trade trades_log",
                "table": [], "sub_models": {}, "issues": []}
    t = cfg.get("tick_size")
    spr = cfg.get("spread") or {"mode": "none"}
    slp = cfg.get("slippage") or {"mode": "none"}
    imp = cfg.get("market_impact") or {"mode": "none"}
    com = cfg.get("commission") or {"mode": "none"}
    fin = cfg.get("financing")

    gross = comm = finc = 0.0
    drag_spread = drag_slip = drag_impact = drag_tick = 0.0
    fin_ok = True
    violations: List[str] = []
    configured = {
        "spread": spr.get("mode", "none") != "none",
        "slippage": slp.get("mode", "none") != "none",
        "market_impact": imp.get("mode", "none") != "none",
        "commission": com.get("mode", "none") != "none",
        "financing": fin is not None and fin.get("mode") != "none",
    }

    for tr in trades_log:
        side = tr.get("side", "long")
        qty = abs(float(tr.get("qty", tr.get("contracts", 1.0))))   # never improve via sign
        ep, xp = float(tr["entry_price"]), float(tr["exit_price"])
        direction = 1.0 if side in ("long", "buy") else -1.0
        gross += (xp - ep) * direction * qty

        e_act, x_act = _side_action(side, True), _side_action(side, False)

        # adverse shifts (price units), per fill
        spr_e = _shift_amount(spr, "spread", ep, qty) / 2.0   # half-spread per fill
        spr_x = _shift_amount(spr, "spread", xp, qty) / 2.0
        slp_e = _shift_amount(slp, "slippage", ep, qty)
        slp_x = _shift_amount(slp, "slippage", xp, qty)
        imp_e = _impact_shift(imp, ep, qty, e_act)
        imp_x = _impact_shift(imp, xp, qty, x_act)

        # base adverse price BEFORE tick
        b_e = _shift(e_act, ep, spr_e + slp_e + imp_e)
        b_x = _shift(x_act, xp, spr_x + slp_x + imp_x)
        # final executable price AFTER adversarial tick quantisation
        q_e = _quantize(b_e, t, e_act)
        q_x = _quantize(b_x, t, x_act)

        # ---- cost-engine invariant: configured layers may not pay the trader ----
        for label, amt in (("spread", spr_e + spr_x), ("slippage", slp_e + slp_x),
                           ("market_impact", imp_e + imp_x)):
            if configured[label] and amt < 0.0:
                violations.append(f"{label} charged {amt:.6g} (< 0) on one fill")
        comm_e = _commission(com, abs(ep) * qty, qty, True)
        comm_x = _commission(com, abs(xp) * qty, qty, False)
        if configured["commission"] and (comm_e < 0.0 or comm_x < 0.0):
            violations.append(f"commission charged {comm_e + comm_x:.6g} (< 0) "
                              f"across the round trip")
        comm += comm_e + comm_x

        drag_spread += (spr_e + spr_x) * qty
        drag_slip += (slp_e + slp_x) * qty
        drag_impact += (imp_e + imp_x) * qty
        # tick drag: quantisation never improves the trader (BUY ceil / SELL floor),
        # so the extra adverse cost is simply |quantised - base| per fill.
        drag_tick += abs(q_e - b_e) * qty + abs(q_x - b_x) * qty

        if configured["financing"] and fin is not None:
            if tr.get("exit_ts") is None or tr.get("entry_ts") is None:
                fin_ok = False
            else:
                secs = _to_seconds(tr["exit_ts"]) - _to_seconds(tr["entry_ts"])
                days = max(0.0, secs) / 86400.0
                notional = abs(ep) * qty
                finc += notional * float(fin.get("value_bps_per_day", 0.0)) / 1e4 * days

    total_cost = drag_tick + drag_spread + drag_slip + drag_impact + comm + finc
    net = gross - total_cost
    cost_drag = (total_cost / abs(gross) * 100.0) if abs(gross) > 1e-12 else 0.0
    table = [
        ("Gross PnL", gross, True),
        ("Tick size", -drag_tick, False),
        ("Spread", -drag_spread, False),
        ("Slippage", -drag_slip, False),
        ("Market Impact", -drag_impact, False),
        ("Commission", -comm, False),
        ("Financing", -finc, False),
        ("Net PnL", net, True),
    ]
    sub = {
        "commission": "PASS" if configured["commission"] else "NOT VERIFIED",
        "spread": "PASS" if configured["spread"] else "NOT VERIFIED",
        "slippage": "PASS" if configured["slippage"] else "NOT VERIFIED",
        "tick_size": "PASS" if t else "NOT VERIFIED",
        "market_impact": "PASS" if configured["market_impact"] else "NOT VERIFIED",
        "financing": ("PASS" if fin_ok and configured["financing"]
                      else "NOT VERIFIED"),
    }
    issues = []
    if violations:
        issues.append({"code": "COST_ENGINE_INVARIANT", "severity": "P0",
                       "finding": f"adverse-cost invariant violated on "
                                  f"{len(violations)} fill(s): {violations[0]} - a "
                                  f"cost model that PAYS the trader is rejected, "
                                  f"never VERIFIED"})
    verdict = "FAIL" if violations else "VERIFIED"
    declared_missing = [k for k in ("commission", "spread", "slippage",
                                    "market_impact", "financing")
                        if configured[k] and sub.get(k) != "PASS"]
    return {"verdict": verdict, "issues": issues,
            "declared_missing": declared_missing,
            "gross_pnl": gross, "net_pnl": net,
            "cost_drag_pct": round(cost_drag, 2), "total_cost": total_cost,
            "table": [[name, round(v, 4), is_hl] for name, v, is_hl in table],
            "sub_models": sub}


def format_performance_table(net: Dict) -> str:
    if not net.get("table"):
        return "(no net performance table)"
    L = ["PERFORMANCE AUDIT (cost layers, adverse fills)",
         f"{'':24}{'':>14}"]
    rows = []
    for name, val, hl in net["table"]:
        rows.append(f"{name:<24}{val:>+14,.4f}")
    rows.append(f"{'Cost Drag':<24}{net['cost_drag_pct']:>13.2f}%")
    return "\n".join(["PERFORMANCE AUDIT"] + rows +
                     [f"Sub-models: " + ", ".join(f"{k}={v}" for k, v in
                                                  net["sub_models"].items())])
