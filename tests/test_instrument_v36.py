"""V3.6 — Instrument / execution-realism contract (external-audit round).

A backtest whose fills cannot actually execute (non-lot qty, ghost fills below
min_qty, sub-min_notional) is not a realistic backtest. The instrument contract
lives on DataSpec (qty_step/min_qty/min_notional/contract_size) and is enforced
inside the net cost engine; nothing declared -> sub-check NOT VERIFIED, never an
assumed-clean PASS.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit
from validator import costengine, costs
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def tr(qty=1.0, ep=100.0, xp=110.0, volume=None, side="long"):
    t = {"side": side, "qty": qty, "entry_price": ep, "exit_price": xp,
         "entry_ts": pd.Timestamp("2026-01-01"),
         "exit_ts": pd.Timestamp("2026-01-02")}
    if volume is not None:
        t["volume"] = volume
    return t


def inst(qs=0.0, mq=0.0, mn=0.0, cs=1.0):
    out = {}
    if qs:
        out["qty_step"] = qs
    if mq:
        out["min_qty"] = mq
    if mn:
        out["min_notional"] = mn
    if cs != 1.0:
        out["contract_size"] = cs
    return out


class TestExecutabilityChecks(unittest.TestCase):

    def test_qty_not_lot_expressible(self):
        cfg = {"instrument": inst(qs=0.1)}
        rep = costengine.net_audit([tr(qty=1.237)], cfg)
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("EXEC_QTY_STEP", codes)
        self.assertEqual(rep["sub_models"]["execution"], "FAIL")

    def test_lot_exact_qty_clean(self):
        cfg = {"instrument": inst(qs=0.1)}
        rep = costengine.net_audit([tr(qty=1.2)], cfg)
        self.assertNotIn("EXEC_QTY_STEP", [i["code"] for i in rep["issues"]])
        self.assertEqual(rep["sub_models"]["execution"], "PASS")

    def test_ghost_fill_below_min_qty(self):
        cfg = {"instrument": inst(mq=1.0)}
        rep = costengine.net_audit([tr(qty=0.5)], cfg)
        self.assertIn("EXEC_MIN_QTY", [i["code"] for i in rep["issues"]])

    def test_sub_min_notional(self):
        cfg = {"instrument": inst(mn=500.0)}
        rep = costengine.net_audit([tr(qty=1.0, ep=100.0)], cfg)   # notional 100 < 500
        self.assertIn("EXEC_MIN_NOTIONAL", [i["code"] for i in rep["issues"]])

    def test_undeclared_instrument_not_verified_no_issues(self):
        rep = costengine.net_audit([tr(qty=1.237)], {})
        self.assertEqual(rep["sub_models"]["execution"], "NOT VERIFIED")
        self.assertEqual(rep["issues"], [])

    def test_future_contract_size_scales_commission_notional(self):
        # contract_size=10: notional for bps commission is qty*10*price
        cfg = {"instrument": inst(cs=10.0),
               "commission": {"mode": "bps", "open_rate": 100.0,
                              "close_rate": 100.0}}   # 1% of notional per side
        rep = costengine.net_audit([tr(qty=1.0, ep=100.0, xp=110.0)], cfg)
        # notional = 1*10*100 = 1000 entry, 1100 exit; 1% each = 10 + 11 = 21
        self.assertAlmostEqual(abs(rep["table"][5][1]), 21.0, places=6)


class TestVolumeAwareImpact(unittest.TestCase):

    def test_volume_linear_with_volume(self):
        cfg = {"market_impact": {"mode": "volume_linear", "coeff": 0.1}}
        rep = costengine.net_audit([tr(volume=1000.0)], cfg)
        self.assertEqual(rep["sub_models"]["market_impact"], "PASS")
        self.assertEqual(rep["declared_missing"], [])
        drag = dict((r[0], r[1]) for r in rep["table"])["Market Impact"]
        self.assertLess(drag, 0.0)                    # impact present, adverse

    def test_volume_linear_without_volume_not_verified(self):
        cfg = {"market_impact": {"mode": "volume_linear", "coeff": 0.1}}
        rep = costengine.net_audit([tr()], cfg)       # no volume key
        self.assertEqual(rep["sub_models"]["market_impact"], "NOT VERIFIED")
        self.assertIn("market_impact", rep["declared_missing"])


class TestInstrumentThroughAudit(unittest.TestCase):
    """DataSpec instrument reaches the Costs section via the audit pipeline."""

    def _run_audit(self, spec, cost=None):
        df = D.regime_trend_df(n=600)

        def run(frame, params):
            # ledger-consistent: 3 legs of qty 1.237 x gross 10 = 37.11
            return {"pnl": 37.11, "trades": 3,
                    "trades_log": [tr(qty=1.237) for _ in range(3)]}
        strat = Strategy(name="ghost", run=run, entry_semantics="next_open")
        cfg = {"scope": ["Costs"], "seed": 1,
               "cost": cost or {"commission": {"mode": "bps", "open_rate": 5.0,
                                               "close_rate": 5.0}}}
        return audit(strat, df, spec, cfg)

    def test_qty_step_violation_conditional(self):
        rep = self._run_audit(SPEC.__class__(bar_seconds=300, qty_step=0.1))
        self.assertEqual(rep["sections"]["Costs"]["status"], "CONDITIONAL PASS")
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")
        self.assertGreaterEqual(rep["blocking"]["P1"], 1)

    def test_no_instrument_stays_clean(self):
        rep = self._run_audit(SPEC)
        self.assertNotIn("EXEC_QTY_STEP", [i["code"] for i in rep["issues"]])
        codes = [i["code"] for i in rep["issues"]]
        self.assertNotIn("EXEC_QTY_STEP", codes)

    def test_clean_instrument_verified(self):
        spec = SPEC.__class__(bar_seconds=300, qty_step=0.1, min_qty=1.0)
        # qty=1.237 still fails step -> use exact-lot trades for the clean case
        df = D.regime_trend_df(n=600)

        def run(frame, params):
            # ledger-consistent: 3 legs of qty 1.2 x gross 10 = 36.0
            return {"pnl": 36.0, "trades": 3,
                    "trades_log": [tr(qty=1.2) for _ in range(3)]}
        strat = Strategy(name="lot", run=run, entry_semantics="next_open")
        rep = audit(strat, df, spec, {"scope": ["Costs"], "seed": 1,
                                      "cost": {"commission": {"mode": "bps",
                                                              "open_rate": 5.0,
                                                              "close_rate": 5.0}}})
        self.assertEqual(rep["sections"]["Costs"]["status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
