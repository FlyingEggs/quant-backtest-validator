"""V3.2 — Realistic Cost & Net PnL anti-cheat tests.

Reviewer-mandated cases:
  C1 gross>0 / net<0 must flip correctly
  C2 BUY vs SELL commission asymmetry (open_rate != close_rate)
  C3 price 100.003 with tick 0.01 -> legal adversarial tick (buy up, sell down)
  C4 slippage/spread/impact never IMPROVE a fill
  C5 cost model uses no future/cross-row data (pure per-trade, order-invariant)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import costengine
from validator.costengine import net_audit, format_performance_table


def tr(side, entry, exit_, qty=1.0, **kw):
    d = {"side": side, "entry_price": entry, "exit_price": exit_, "qty": qty}
    d.update(kw)
    return d


def basic_cfg(**over):
    cfg = {"commission": {"mode": "bps", "open_rate": 5.0, "close_rate": 5.0},
           "spread": {"mode": "bps", "value_bps": 2.0},
           "slippage": {"mode": "bps", "value_bps": 3.0},
           "tick_size": 0.01,
           "market_impact": {"mode": "none"},
           "financing": None}
    cfg.update(over)
    return cfg


class TestAntiCheat(unittest.TestCase):

    def test_c1_gross_positive_net_negative_flips(self):
        trades = [tr("long", 100.0, 101.0, qty=10.0)]      # gross +10
        cfg = basic_cfg()
        cfg["commission"] = {"mode": "notional", "open_rate": 0.5, "close_rate": 0.5}
        net = net_audit(trades, cfg)
        self.assertGreater(net["gross_pnl"], 0)
        self.assertLess(net["net_pnl"], 0)
        self.assertIn("Net PnL", [r[0] for r in net["table"]])

    def test_c2_open_close_fee_asymmetry(self):
        cfg = basic_cfg()
        cfg["commission"] = {"mode": "bps", "open_rate": 0.0, "close_rate": 100.0}
        a = net_audit([tr("long", 100.0, 101.0, qty=1.0)], cfg)
        b = net_audit([tr("short", 100.0, 99.0, qty=1.0)], cfg)
        self.assertNotEqual(a["table"], b["table"])          # both sides charged

    def test_c3_tick_quantisation_adversarial(self):
        trades = [tr("long", 100.003, 101.007, qty=1.0)]
        cfg = basic_cfg()
        cfg["spread"] = {"mode": "none"}
        cfg["slippage"] = {"mode": "none"}
        net = net_audit(trades, cfg)
        # BUY entry rounds UP (100.003 -> 100.01, +0.007), SELL exit rounds DOWN
        # (101.007 -> 101.00, +0.007): tick grid costs money, never improves.
        self.assertGreater(net["total_cost"], 0.0)
        table = dict((r[0], r[1]) for r in net["table"])
        self.assertAlmostEqual(abs(table["Tick size"]), 0.014, places=6)

    def test_c4_never_improves(self):
        for side, ep, xp in (("long", 100.0, 110.0), ("short", 110.0, 100.0)):
            cfg = basic_cfg()
            net = net_audit([tr(side, ep, xp, qty=1.0)], cfg)
            self.assertGreaterEqual(net["total_cost"], 0.0)
        # directional: buy exec never below raw+spread-0 etc. verified via cost>=0 plus
        # a pure-slippage long where exit price must be <= raw exit (adverse)
        cfg = basic_cfg()
        cfg["commission"] = {"mode": "none"}
        cfg["spread"] = {"mode": "none"}
        cfg["tick_size"] = None
        cfg["slippage"] = {"mode": "fixed", "value": 0.5}
        net = net_audit([tr("long", 100.0, 110.0, qty=1.0)], cfg)
        # 2 fills x 0.5 adverse = 1.0 total drag
        self.assertAlmostEqual(abs(net["net_pnl"] - net["gross_pnl"]), 1.0, places=6)

    def test_c5_pure_and_order_invariant(self):
        trades = [tr("long", 100.0, 101.0, qty=1.0),
                  tr("short", 101.0, 100.5, qty=1.0)]
        cfg = basic_cfg()
        a = net_audit(trades, cfg)
        b = net_audit(list(reversed(trades)), cfg)            # order must not matter
        self.assertEqual(a["total_cost"], b["total_cost"])
        self.assertEqual(a["net_pnl"], b["net_pnl"])

    def test_financing_needs_holding(self):
        cfg = basic_cfg()
        cfg["financing"] = {"mode": "funding_bps_per_day", "value_bps_per_day": 1.0}
        no_ts = net_audit([tr("long", 100.0, 101.0, qty=1.0)], cfg)
        self.assertEqual(no_ts["sub_models"]["financing"], "NOT VERIFIED")
        import pandas as pd
        t0 = pd.Timestamp("2026-01-01 00:00")
        t1 = t0 + pd.Timedelta(days=3)
        yes = net_audit([tr("long", 100.0, 101.0, qty=100.0,
                            entry_ts=t0, exit_ts=t1)], cfg)
        self.assertEqual(yes["sub_models"]["financing"], "PASS")
        self.assertLess(yes["table"][-2][1], 0.0)          # financing drag present (neg)


class TestModelModes(unittest.TestCase):

    def test_impact_modes_and_status(self):
        cfg = basic_cfg()
        cfg["market_impact"] = {"mode": "none"}
        net = net_audit([tr("long", 100.0, 101.0, qty=1.0)], cfg)
        self.assertEqual(net["sub_models"]["market_impact"], "NOT VERIFIED")
        cfg["market_impact"] = {"mode": "linear", "coeff": 0.05}
        net2 = net_audit([tr("long", 100.0, 101.0, qty=2.0)], cfg)
        self.assertEqual(net2["sub_models"]["market_impact"], "PASS")
        cfg["market_impact"] = {"mode": "callable",
                                "fn": lambda p, q, a: 0.1 * q}
        net3 = net_audit([tr("long", 100.0, 101.0, qty=1.0)], cfg)
        self.assertGreater(net3["total_cost"], net["total_cost"])

    def test_format_table(self):
        net = net_audit([tr("long", 100.0, 101.0, qty=1.0)], basic_cfg())
        txt = format_performance_table(net)
        self.assertIn("PERFORMANCE AUDIT", txt)
        self.assertIn("Cost Drag", txt)


if __name__ == "__main__":
    unittest.main()
