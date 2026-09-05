"""V3.4.2 — the report carries a human-readable interpretation under the verdict.

PASS means "no cheating evidence in the dimensions we checked", never "this will
work live". The machine report must say so at the top, because clients read the
verdict first and the scope caveat second (this is the anti-''PASS = all clear''
misreading guard).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit, audit_text
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def honest(strategy, df, config):
    return audit_text(strategy, df, SPEC, config)


def honest_strategy():
    df = D.regime_trend_df()
    return as_code_strategy("trend", df, "sig", D.next_open_hold(5),
                            entry_semantics="next_open"), df


class TestInterpretationLabel(unittest.TestCase):

    def test_pass_says_scoped_not_guarantee(self):
        strat, df = honest_strategy()
        cfg = {"scope": ["Data Integrity", "Execution", "Statistics"], "seed": 11}
        txt = audit_text(strat, df, SPEC, cfg)
        self.assertIn("Overall Verdict : PASS", txt)
        self.assertIn("No evidence of cheating was found", txt)
        self.assertIn("not a guarantee of live performance", txt)

    def test_incomplete_says_missing_evidence(self):
        strat, df = honest_strategy()
        # Costs in scope but no cost model -> NOT VERIFIED -> INCOMPLETE.
        # (Look-ahead excluded: the slow trend signal legitimately trips the
        # expansion P0 without an explicit confirmation, which is a different case.)
        cfg = {"scope": ["Data Integrity", "Execution", "Statistics", "Costs"],
               "seed": 11}
        txt = audit_text(strat, df, SPEC, cfg)
        self.assertIn("Overall Verdict : INCOMPLETE", txt)
        self.assertIn("missing evidence is not a clean bill", txt)

    def test_conditional_says_manual_confirmation(self):
        def run(frame, params):
            x, y = float(params["x"]), float(params["y"])
            pnl = 1000.0 if (x == 0.0 and y == 0.0) else 10.0
            return {"pnl": pnl, "trades": 100}
        strat = Strategy(name="island", run=run, default_params={"x": 0, "y": 0},
                         param_grid={"x": [-2, -1, 0, 1, 2],
                                     "y": [-2, -1, 0, 1, 2]},
                         entry_semantics="next_open")
        cfg = {"scope": ["Robustness"], "seed": 1,
               "surface": {"x": "x", "y": "y", "x_values": [-2, -1, 0, 1, 2],
                           "y_values": [-2, -1, 0, 1, 2]}}
        txt = audit_text(strat, D.regime_trend_df(), SPEC, cfg)
        self.assertIn("Overall Verdict : CONDITIONAL PASS", txt)
        self.assertIn("P1 findings need manual confirmation", txt)

    def test_fail_says_blocking_defect(self):
        df = D.same_bar_leak_df()
        strat = as_code_strategy("leaky", df, "sig", D.same_bar_bt,
                                 entry_semantics="same_bar")
        txt = audit_text(strat, df, SPEC,
                         {"scope": ["Execution"], "seed": 7,
                          "expansion_confirmation": "completed"})
        self.assertIn("Overall Verdict : FAIL", txt)
        self.assertIn("blocking defect was found", txt)


if __name__ == "__main__":
    unittest.main()
