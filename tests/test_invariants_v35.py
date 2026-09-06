"""V3.5.0 — invariant & state-consistency guards (external audit round).

  * cost engine: a configured cost layer that PAYS the trader (negative spread /
    slippage / impact / commission rates, incl. callable returns) is rejected
    with COST_ENGINE_INVARIANT (P0) - never VERIFIED. Financing is the sole layer
    where negative is legal (real negative funding regimes pay longs).
  * cost status: VERIFIED with a dead sub-model is state leakage - configured but
    unverifiable sub-models (financing without timestamps) downgrade the section.
  * data integrity: +-inf OHLC is a hard P0 (DATA_NONFINITE), not a P1.
  * legacy flat-bps cost config (fee_bps/slippage_bps) is reported as unconsumed.
  * DataSpec bar-timestamp semantics are declared (OPEN default; CLOSE flagged).
  * MTF custom callable transforms are DECLARED (causality not mechanically
    verifiable), never blessed as verified PASS.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit, audit_text
from validator import costengine, costs, data_integrity as di, mtf
from validator.costs import costs_check
from examples import demo as D
from tests.test_engine_boundaries import hour_pair

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def tr(side="long", ep=100.0, xp=110.0, qty=1.0, ts=True):
    t = {"side": side, "entry_price": ep, "exit_price": xp, "qty": qty}
    if ts:
        t.update({"entry_ts": pd.Timestamp("2026-01-01"),
                  "exit_ts": pd.Timestamp("2026-01-02")})
    return t


class TestCostInvariants(unittest.TestCase):
    """The engine's central promise: every fill is adjusted AGAINST the trader.
    Any configured layer that pays the trader is a P0, not a model."""

    def _assert_rejected(self, cfg, code="COST_ENGINE_INVARIANT"):
        rep = costengine.net_audit([tr()], cfg)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertIn(code, [i["code"] for i in rep["issues"]])

    def test_negative_fixed_spread_rejected(self):
        self._assert_rejected({"spread": {"mode": "fixed", "value": -10.0}})

    def test_negative_callable_slippage_rejected(self):
        self._assert_rejected({"slippage": {"mode": "callable",
                                            "fn": lambda p, q, k: -50.0}})

    def test_negative_callable_impact_rejected(self):
        self._assert_rejected({"market_impact": {"mode": "callable",
                                                 "fn": lambda p, q, a: -50.0}})

    def test_negative_close_rate_rejected(self):
        # static gate catches both rates now (close_rate was unchecked before)
        rep = costs_check({"cost": {"commission": {"mode": "bps", "open_rate": 0.0,
                                                   "close_rate": -100.0}},
                           "trades_log": [tr()]})
        self.assertEqual(rep["status"], "FAIL")
        self.assertIn("COST_NEGATIVE", [i["code"] for i in rep["issues"]])

    def test_negative_open_rate_rejected(self):
        self._assert_rejected({"commission": {"mode": "bps", "open_rate": -5.0,
                                              "close_rate": 5.0}})

    def test_negative_financing_is_legal(self):
        # real funding regimes pay longs to hold - a negative parameter is market
        # semantics, so it must NOT trip the invariant.
        rep = costengine.net_audit([tr()], {"financing": {
            "mode": "funding_bps_per_day", "value_bps_per_day": -5.0}})
        self.assertEqual(rep["verdict"], "VERIFIED")
        self.assertLess(rep["total_cost"], 0.0)

    def test_legacy_flatbps_config_rejected_as_unused(self):
        rep = costs_check({"cost": {"fee_bps": 4.0, "slippage_bps": 2.0},
                           "trades_log": [tr()]})
        self.assertEqual(rep["status"], "FAIL")
        self.assertIn("COST_CONFIG_UNUSED", [i["code"] for i in rep["issues"]])

    def test_invariant_via_full_audit_fails_overall(self):
        df = D.regime_trend_df(n=600)
        strat = as_code_strategy("honest", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"scope": ["Costs"], "seed": 1,
                                      "cost": {"slippage": {"mode": "fixed",
                                                            "value": -10.0}}})
        self.assertEqual(rep["overall"], "FAIL")
        self.assertEqual(rep["sections"]["Costs"]["status"], "FAIL")
        self.assertGreaterEqual(rep["blocking"]["P0"], 1)

    def test_clean_cost_model_still_verified(self):
        rep = costs_check({"cost": {"commission": {"mode": "bps", "open_rate": 5.0,
                                                   "close_rate": 5.0},
                                    "slippage": {"mode": "bps", "value_bps": 2.0},
                                    "trades_log": [tr()]}})
        self.assertEqual(rep["status"], "VERIFIED")


class TestCostStatusConsistency(unittest.TestCase):
    """VERIFIED with a dead sub-model is state leakage."""

    def test_financing_without_timestamps_not_verified(self):
        rep = costs_check({"cost": {"financing": {"mode": "funding_bps_per_day",
                                                  "value_bps_per_day": 1.0},
                                    "trades_log": [tr(ts=False)]}})
        self.assertEqual(rep["status"], "NOT VERIFIED")
        self.assertIn("COST_SUB_INCOMPLETE", [i["code"] for i in rep["issues"]])
        # sub-verdict reflects it too
        self.assertEqual(rep["evidence"]["net"]["sub_models"]["financing"],
                         "NOT VERIFIED")

    def test_full_audit_reports_incomplete(self):
        df = D.regime_trend_df(n=600)

        def run(frame, params):
            # ledger-consistent: 5 legs x gross 10 = 50
            return {"pnl": 50.0, "trades": 5,
                    "trades_log": [tr(ts=False) for _ in range(5)]}
        strat = Strategy(name="no-ts", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"scope": ["Costs"], "seed": 1,
                                      "cost": {"financing": {
                                          "mode": "funding_bps_per_day",
                                          "value_bps_per_day": 1.0}}})
        self.assertEqual(rep["sections"]["Costs"]["status"], "NOT VERIFIED")
        self.assertEqual(rep["overall"], "INCOMPLETE")   # never a clean VERIFIED


class TestDataFiniteInvariant(unittest.TestCase):
    """+-inf prices poison every downstream layer; P0, never a silent PASS."""

    def _frame(self, value):
        return pd.DataFrame({"open": [value] * 5, "high": [value] * 5,
                             "low": [value] * 5, "close": [value] * 5},
                            index=pd.date_range("2026-01-01", periods=5,
                                                freq="300s"))

    def test_inf_ohlc_fails(self):
        rep = di.check(self._frame(float("inf")), None)
        self.assertEqual(rep["status"], "FAIL")
        self.assertIn("DATA_NONFINITE", [i["code"] for i in rep["issues"]])

    def test_neg_inf_ohlc_fails(self):
        rep = di.check(self._frame(float("-inf")), None)
        self.assertEqual(rep["status"], "FAIL")
        self.assertIn("DATA_NONFINITE", [i["code"] for i in rep["issues"]])

    def test_finite_frame_clean_of_nonfinite(self):
        df = self._frame(100.0)
        df["close"] = [101.0, 99.0, 102.0, 98.0, 100.0]
        rep = di.check(df, None)
        codes = [i["code"] for i in rep["issues"]]
        self.assertNotIn("DATA_NONFINITE", codes)

    def test_inf_frame_fails_full_audit(self):
        df = self._frame(float("inf"))

        def run(frame, params):
            return {"pnl": 1.0, "trades": 1}
        strat = Strategy(name="inf", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"scope": ["Data Integrity"]})
        self.assertEqual(rep["overall"], "FAIL")


class TestDataSpecTimestampSemantics(unittest.TestCase):
    def test_close_semantics_flagged(self):
        df = D.regime_trend_df(n=200)
        spec = DataSpec(bar_seconds=300, bar_timestamp_semantics="CLOSE")
        rep = di.check(df, spec)
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("DATA_TS_SEMANTICS", codes)

    def test_default_open_semantics_clean(self):
        df = D.regime_trend_df(n=200)
        rep = di.check(df, DataSpec(bar_seconds=300))
        self.assertNotIn("DATA_TS_SEMANTICS", [i["code"] for i in rep["issues"]])


class TestMtfCallableBoundary(unittest.TestCase):
    """A custom callable transform may hide future access - DECLARED, never a
    verified PASS; a detected leak on it is still FAIL (evidence over trust)."""

    def _check(self, transform):
        low, high = hour_pair(hours=6)
        spec = DataSpec(bar_seconds=300, timeframes={"h1": high})
        return mtf.check(low, spec, {"mtf": {"col": "sig_legal", "frame": "h1",
                                             "frame_seconds": 3600,
                                             "transform": transform}})

    def test_sign_diff_binding_stays_verified(self):
        rep = self._check("sign_diff")
        self.assertEqual(rep["status"], "PASS")

    def test_callable_transform_declared_not_verified(self):
        rep = self._check(lambda s: s)
        self.assertEqual(rep["status"], "NOT VERIFIED")
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("MTF_TRANSFORM_DECLARED", codes)

    def test_callable_with_leak_still_fails(self):
        # a callable that reproduces sign_diff semantics on the naive column: the
        # leak is still FAIL - leak evidence outweighs the DECLARED boundary.
        def sign_like(s):
            d = s.diff().to_numpy()
            return pd.Series(np.where(np.isnan(np.sign(d)), 0.0, np.sign(d)),
                             index=s.index)
        low, high = hour_pair(hours=6)
        spec = DataSpec(bar_seconds=300, timeframes={"h1": high})
        rep = mtf.check(low, spec, {"mtf": {"col": "sig_naive", "frame": "h1",
                                            "frame_seconds": 3600,
                                            "transform": sign_like}})
        self.assertEqual(rep["status"], "FAIL")    # leak evidence > trust boundary


if __name__ == "__main__":
    unittest.main()
