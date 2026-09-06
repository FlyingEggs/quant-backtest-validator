"""V3.7 — Parameter-provenance contract (hidden-refit attack round).

A strategy may declare fit_is(df) -> IS-learned params plus accepts_frozen (run()
treats injected parameters as authoritative). The audit then machine-verifies the
OOS parameters really come from the frozen IS fit: OOS output under frozen-vs-
adversarial injection must DIFFER. Identical output with trades = the strategy
decides internally = hidden refit = P0 PARAM_PROVENANCE.

Un-declared contract -> provenance NOT VERIFIED (reported, never assumed) and no
verdict impact (existing black-box strategies must not degrade).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import Strategy
from validator.wf import parameter_freeze_audit, _param_hash
from examples import demo as D

DF = D.regime_trend_df(n=1200)


def run_uses_params(frame, params):
    """Honest contract strategy: pnl follows the injected parameter."""
    return {"pnl": 100.0 * float(params["k"]), "trades": 10}


def honest_strategy():
    return Strategy(name="honest", run=run_uses_params,
                    default_params={"k": 1.0},
                    param_grid={"k": [0.5, 1.0, 2.0]},
                    entry_semantics="next_open",
                    fit_is=lambda df: {"k": 2.0},
                    accepts_frozen=True)


class TestProvenanceHonest(unittest.TestCase):

    def test_frozen_is_params_verified_pass(self):
        rep = parameter_freeze_audit(honest_strategy(), DF, {})
        self.assertEqual(rep["provenance"], "PASS")
        self.assertEqual(rep["frozen_hash"], _param_hash({"k": 2.0}))
        codes = [i["code"] for i in rep["issues"]]
        self.assertNotIn("PARAM_PROVENANCE", codes)


class TestProvenanceAttacks(unittest.TestCase):
    """Hidden refit must be caught, not reported."""

    def test_ignores_injected_params_fails(self):
        # accepts the contract but never looks at params: identical under both
        # injections -> internal decision-making -> P0
        def run_ignore(frame, params):
            return {"pnl": 42.0, "trades": 10}
        strat = Strategy(name="cheat", run=run_ignore,
                         default_params={"k": 1.0},
                         entry_semantics="next_open",
                         fit_is=lambda df: {"k": 2.0},
                         accepts_frozen=True)
        rep = parameter_freeze_audit(strat, DF, {})
        self.assertEqual(rep["provenance"], "FAIL")
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("PARAM_PROVENANCE", codes)
        self.assertEqual(next(i["severity"] for i in rep["issues"]
                             if i["code"] == "PARAM_PROVENANCE"), "P0")

    def test_internal_refit_on_full_frame_fails(self):
        # fit_is declares IS params, but run() re-fits on whatever frame it is
        # given (here: uses the frame's own data to decide) -> same output under
        # any injected params because the internal fit wins -> P0.
        def run_refits(frame, params):
            # hidden refit: params injected are ignored; output driven by frame
            return {"pnl": float(len(frame)) / 100.0, "trades": 10}
        strat = Strategy(name="refit", run=run_refits,
                         default_params={"k": 1.0},
                         entry_semantics="next_open",
                         fit_is=lambda df: {"k": 2.0},
                         accepts_frozen=True)
        rep = parameter_freeze_audit(strat, DF, {})
        self.assertEqual(rep["provenance"], "FAIL")
        self.assertIn("PARAM_PROVENANCE", [i["code"] for i in rep["issues"]])


class TestProvenanceUndeclared(unittest.TestCase):
    """No contract -> NOT VERIFIED, never a fabricated PASS or FAIL."""

    def test_default_strategy_not_verified(self):
        strat = Strategy(name="blackbox", run=run_uses_params,
                         entry_semantics="next_open")
        rep = parameter_freeze_audit(strat, DF, {})
        self.assertEqual(rep["provenance"], "NOT VERIFIED")
        self.assertIsNone(rep["frozen_hash"])
        self.assertNotIn("PARAM_PROVENANCE", [i["code"] for i in rep["issues"]])

    def test_fit_is_without_accepts_frozen_not_verified(self):
        # declaring only one half of the contract is not enough for verification
        strat = Strategy(name="half", run=run_uses_params,
                         default_params={"k": 1.0},
                         entry_semantics="next_open",
                         fit_is=lambda df: {"k": 2.0})
        rep = parameter_freeze_audit(strat, DF, {})
        self.assertEqual(rep["provenance"], "NOT VERIFIED")

    def test_small_sample_skipped_not_verified(self):
        strat = honest_strategy()
        rep = parameter_freeze_audit(strat, D.regime_trend_df(n=120), {})
        self.assertEqual(rep["provenance"], "NOT VERIFIED")


if __name__ == "__main__":
    unittest.main()
