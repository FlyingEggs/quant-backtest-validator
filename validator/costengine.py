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

import numpy as np


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
    """Apply the cost model over per-trade rows. Pure: row order cannot matter."""
    if not trades_log:
        return {"verdict": "NOT VERIFIED", "reason": "no per-trade trades_log",
                "table": [], "sub_models": {}}
    t = cfg.get("tick_size")
    spr = cfg.get("spread") or {"mode": "none"}
    slp = cfg.get("slippage") or {"mode": "none"}
    imp = cfg.get("market_impact") or {"mode": "none"}
    com = cfg.get("commission") or {"mode": "none"}
    fin = cfg.get("financing")

    gross = comm = finc = 0.0
    drag_spread = drag_slip = drag_impact = drag_tick = 0.0
    fin_ok = True

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

        drag_spread += (spr_e + spr_x) * qty
        drag_slip += (slp_e + slp_x) * qty
        drag_impact += (imp_e + imp_x) * qty
        # tick drag: quantisation never improves the trader (BUY ceil / SELL floor),
        # so the extra adverse cost is simply |quantised - base| per fill.
        drag_tick += abs(q_e - b_e) * qty + abs(q_x - b_x) * qty

        comm += _commission(com, abs(ep) * qty, qty, True) + \
                _commission(com, abs(xp) * qty, qty, False)

        if fin and fin.get("mode") == "funding_bps_per_day":
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
        "commission": "PASS" if com.get("mode", "none") != "none" else "NOT VERIFIED",
        "spread": "PASS" if spr.get("mode", "none") != "none" else "NOT VERIFIED",
        "slippage": "PASS" if slp.get("mode", "none") != "none" else "NOT VERIFIED",
        "tick_size": "PASS" if t else "NOT VERIFIED",
        "market_impact": "PASS" if imp.get("mode", "none") != "none" else "NOT VERIFIED",
        "financing": ("PASS" if fin_ok and fin and fin.get("mode") != "none"
                      else "NOT VERIFIED"),
    }
    return {"verdict": "VERIFIED", "gross_pnl": gross, "net_pnl": net,
            "cost_drag_pct": round(cost_drag, 2), "total_cost": total_cost,
            "table": [[name, round(v, 4), is_hl] for name, v, is_hl in table],
            "sub_models": sub}


def _to_seconds(ts) -> float:
    import pandas as pd
    if isinstance(ts, np.datetime64):
        return float(ts.astype("datetime64[ns]").astype(np.int64)) / 1e9
    try:
        val = float(ts.value if isinstance(ts, pd.Timestamp) else ts)
    except AttributeError:
        val = float(ts)
    if val > 1e17:
        return val / 1e9
    if val > 1e13:
        return val / 1e3
    return val


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
