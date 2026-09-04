"""Adversarial suite — attack the Validator itself (FP/FN).

Goal: prove the engine cannot be fooled into a clean bill for cheating strategies,
and does not murder legitimate ones. Each scenario encodes an EXPECTED verdict.

Known capability gaps are marked `skip` with the reason stated - never faked as PASS.

Run:  python3 -m unittest tests.test_adversarial -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import audit, DataSpec, Strategy, as_code_strategy, randomized_control
from validator import statistics as stats_mod
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def pcode(rep, code):
    return [i for i in rep["issues"] if i["code"] == code]


def scoped_clean_cfg(**over):
    """Scope minus MTF (roadmap), costs declared: enables a genuine PASS test."""
    cfg = {"expansion_confirmation": "completed",
           "cost": {"fee_bps": 4.0, "slippage_bps": 2.0},
           "scope": ["Data Integrity", "Look-ahead", "Execution",
                     "Statistics", "Robustness", "Costs"]}
    cfg.update(over)
    return cfg


class TestNoFalseClean(unittest.TestCase):
    """FN: cheating strategies must never get a clean PASS."""

    def test_02_same_bar_cheat_fails(self):
        df, bt = D.same_bar_leak_df(), D.same_bar_bt
        strat = as_code_strategy("same-bar cheat", df, "sig", bt,
                                 entry_semantics="same_bar")
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 7})
        self.assertEqual(rep["overall"], "FAIL")
        self.assertTrue(pcode(rep, "EXECUTION_FILL"))
        self.assertTrue(pcode(rep, "ENTRY_SEMANTICS"))

    def test_03_future_signal_column_withholds_pass(self):
        """Signal column computed from the NEXT close: the engine cannot *prove* the
        leak mechanically, so the correct outcome is CONDITIONAL PASS (P1 evidence),
        never a clean PASS - confirming the leak is the code-review step."""
        rng = np.random.default_rng(4)
        n = 3000
        closes = 3000 * np.cumprod(1 + rng.normal(0.0003, 0.0012, n))
        df = D.frame(closes)
        fwd = pd.Series(closes).shift(-1)
        df["sig"] = np.where(fwd > closes, 1.0, -1.0)          # uses close[i+1]
        bt = D.next_open_hold(3)
        strat = as_code_strategy("future-col cheat", df, "sig", bt,
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 3})
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")    # NOT PASS
        self.assertTrue(pcode(rep, "LAG_DEPENDENCE"))

    def test_04_lowfreq_reuse_fails_unconfirmed(self):
        df = D.daily_signal_df()
        strat = as_code_strategy("day-col reuse", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"expansion_confirmation": None, "seed": 7})
        self.assertEqual(rep["overall"], "FAIL")
        self.assertTrue(pcode(rep, "PERIOD_EXPANSION"))

    def test_06_costs_never_auto_pass(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("c", df, "sig", D.next_open_hold(5))
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 1})
        self.assertEqual(rep["sections"]["Costs"]["status"], "NOT VERIFIED")
        rep2 = audit(strat, df, SPEC, {"expansion_confirmation": "completed",
                                       "cost": {"fee_bps": 1, "slippage_bps": 1},
                                       "seed": 1})
        self.assertEqual(rep2["sections"]["Costs"]["status"], "DECLARED")

    def test_07_oos_overfit_conditional(self):
        rets = np.concatenate([np.full(700, 0.01), np.full(300, -0.02)])
        closes = np.cumsum(rets)
        o = np.empty(len(rets)); o[0] = closes[0]; o[1:] = closes[:-1]
        df = pd.DataFrame({"open": o, "high": np.maximum(o, closes),
                           "low": np.minimum(o, closes), "close": closes},
                          index=pd.date_range("2026-01-01", periods=len(rets),
                                              freq="300s"))

        def run(frame, params):
            return {"pnl": float(frame["close"].iloc[-1] - frame["close"].iloc[0]),
                    "trades": 1}
        strat = Strategy(name="is-fit", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"oos_frac": 0.3, "seed": 1})
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")
        self.assertTrue(pcode(rep, "OOS_INSTABILITY"))

    def test_08_parameter_cliff_1d_detected(self):
        df = D.regime_trend_df()

        def run(frame, params):
            k = params["k"]
            return {"pnl": -1000.0 if k >= 9 else 10.0, "trades": 100}
        strat = Strategy(name="cliffy", run=run, default_params={"k": 1},
                         param_grid={"k": [1, 3, 5, 10]}, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 1})
        self.assertTrue(pcode(rep, "PARAM_CLIFF"))

    def test_09_autocorrelated_returns_not_silent(self):
        """Heavily overlapping returns: Statistics section must NOT be a silent PASS.
        Within a complete scope the verdict stays PASS - dependence discounts
        significance, it does not flip the verdict."""
        rets = D.ar1_rets(n=400, rho=0.8)

        def run(frame, params):
            return {"pnl": float(np.sum(rets)), "trades": 400, "rets": rets}
        strat = Strategy(name="overlapper", run=run, entry_semantics="next_open")
        df = D.regime_trend_df()
        rep = audit(strat, df, SPEC, {"seed": 1})
        self.assertEqual(rep["sections"]["Statistics"]["status"], "CONDITIONAL PASS")
        self.assertTrue(pcode(rep, "STAT_DEPENDENCE"))
        self.assertEqual(rep["overall"], "INCOMPLETE")       # full default scope
        # policy check: within a complete scope, dependence does not flip the verdict
        no_mtf = {"expansion_confirmation": "completed",
                  "cost": {"fee_bps": 4.0, "slippage_bps": 2.0},
                  "scope": ["Data Integrity", "Execution", "Statistics",
                            "Robustness", "Costs"],        # black box: Look-ahead excluded
                  "seed": 1}
        rep2 = audit(strat, df, SPEC, no_mtf)
        self.assertEqual(rep2["overall"], "PASS")
        self.assertEqual(rep2["sections"]["Statistics"]["status"], "CONDITIONAL PASS")
        self.assertEqual(rep2["statistical_confidence"]["significance_reliability"],
                         "DISCOUNTED")


class TestNoFalseFail(unittest.TestCase):
    """FP: legitimate strategies must never be auto-FAILED."""

    def test_01_honest_trend_passes(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("honest", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        # full default scope: MTF roadmap -> INCOMPLETE, but zero blocking findings
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 11})
        self.assertEqual(rep["overall"], "INCOMPLETE")
        self.assertEqual(rep["blocking"]["P0"], 0)     # P2 = STAT_DEPENDENCE (expected)
        self.assertEqual(rep["blocking"]["P1"], 0)
        self.assertEqual(rep["sections"]["Look-ahead"]["status"], "PASS")
        self.assertEqual(rep["sections"]["Execution"]["status"], "PASS")
        # declared scope minus the roadmap module -> genuine PASS
        rep2 = audit(strat, df, SPEC, scoped_clean_cfg(seed=11))
        self.assertEqual(rep2["overall"], "PASS")
        self.assertTrue(rep2["audit_complete"])

    def test_10_short_horizon_legitimate_is_conditional_not_fail(self):
        df = D.markov_short_df()
        strat = as_code_strategy("short-horizon", df, "sig", D.markov_bt,
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 5})
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")
        self.assertNotEqual(rep["overall"], "FAIL")

    def test_11_overlapping_legit_not_failed(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("overlap-hold5", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, scoped_clean_cfg(seed=11))
        self.assertEqual(rep["overall"], "PASS")                     # not failed
        self.assertEqual(rep["sections"]["Statistics"]["status"], "CONDITIONAL PASS")
        self.assertEqual(rep["blocking"]["P0"], 0)

    def test_12_regime_strategy_no_false_fail(self):
        """A slow regime state machine (long in up-regime, flat/short in down) with
        confirmed per-bar semantics must pass, not be murdered by RC/expansion."""
        df = D.regime_trend_df()
        strat = as_code_strategy("regime-switch", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, scoped_clean_cfg(seed=11))
        self.assertEqual(rep["overall"], "PASS")


class TestStatisticsTiers(unittest.TestCase):

    def test_tiers(self):
        from validator.types import Strategy, run_metrics
        df = D.regime_trend_df()

        def strat_with(rets):
            def run(frame, params):
                return {"pnl": 1.0, "trades": len(rets), "rets": rets}
            return Strategy(name="t", run=run, entry_semantics="next_open")

        heavy = stats_mod.check(strat_with(D.ar1_rets(n=400, rho=0.8)), df, SPEC, {})
        self.assertEqual(heavy["status"], "CONDITIONAL PASS")
        self.assertTrue(any(i["code"] == "STAT_DEPENDENCE" for i in heavy["issues"]))
        clean = stats_mod.check(
            strat_with(np.random.default_rng(7).normal(0, 1, 400)), df, SPEC, {})
        self.assertEqual(clean["status"], "PASS")
        self.assertEqual(clean["issues"], [])


# Known capability gaps - stated, never faked -----------------------------------
class TestKnownGaps(unittest.TestCase):
    """Scenarios the engine does NOT claim to detect yet. Skipped with the reason,
    so the suite documents capability boundaries instead of lying about them."""

    def test_05_survivorship_bias(self):
        self.skipTest("survivorship needs a full coin universe + listing history; the "
                      "Strategy/DataSpec contract carries a single frame. Data-level gap "
                      "- a survivorship-biased panel would audit clean today (reported, "
                      "not hidden).")

    def test_08_parameter_island_2d(self):
        self.skipTest("2D parameter-surface island detection is V2.2 (PARAM_ISLAND); "
                      "today only adjacent 1D cliffs (PARAM_CLIFF) are detected.")

    def test_04_mtf_temporal_leak(self):
        self.skipTest("true MTF temporal-availability engine is V3 (5m->1H->4h->Daily "
                      "close/availability timestamps). The period-expansion gate catches "
                      "low-frequency reuse but not all MTF leaks.")

    def test_03_future_signal_proof(self):
        self.skipTest("lag collapse is P1 evidence; *proving* the leak is a code-review "
                      "step (signal construction/timestamps), not a mechanical verdict.")


if __name__ == "__main__":
    unittest.main()
