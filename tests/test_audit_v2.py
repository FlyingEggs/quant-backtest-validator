"""V2 audit-pipeline unit tests: overall verdicts, NOT VERIFIED honesty, OOS,
parameter cliffs, reliability score, report rendering & JSON."""

import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import (audit, audit_text, DataSpec, Strategy, as_code_strategy,
                       save_report, to_jsonable)
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def _df_from_ret(rets):
    closes = np.cumsum(rets)
    o = np.empty(len(rets))
    o[0] = closes[0]
    o[1:] = closes[:-1]
    idx = pd.date_range("2026-01-01", periods=len(rets), freq="300s")
    return pd.DataFrame({"open": o, "high": np.maximum(o, closes),
                         "low": np.minimum(o, closes), "close": closes}, index=idx)


class TestAuditOverall(unittest.TestCase):

    def test_honest_code_strategy_passes(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("honest", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC,
                    {"expansion_confirmation": "completed",
                     "cost": {"fee_bps": 4, "slippage_bps": 1}, "seed": 11})
        self.assertEqual(rep["overall"], "PASS")
        # two P3 hygiene items (expansion-confirmed note, declared-but-unverified costs)
        # each cost 2 pts off a perfect 100; no P0/P1
        self.assertGreaterEqual(rep["reliability_score"], 90)
        self.assertLess(rep["reliability_score"], 100)
        # Costs supplied -> verified; MTF is on the roadmap -> reported NOT VERIFIED
        self.assertNotIn("Costs", rep["not_verified"])
        self.assertIn("MTF", rep["not_verified"])
        self.assertIn("reliability_score", rep)
        self.assertIn("recommendation", rep)

    def test_costs_unverified_when_not_supplied(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("honest", df, "sig", D.next_open_hold(5))
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 11})
        self.assertIn("Costs", rep["not_verified"])
        self.assertEqual(rep["sections"]["Costs"]["status"], "NOT VERIFIED")

    def test_same_bar_declared_fails_with_p0(self):
        df = D.same_bar_leak_df()
        strat = as_code_strategy("leaky", df, "sig", D.same_bar_bt,
                                 entry_semantics="same_bar")
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 7})
        self.assertEqual(rep["overall"], "FAIL")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("ENTRY_SEMANTICS", codes)
        self.assertIn("EXECUTION_FILL", codes)
        self.assertLess(rep["reliability_score"], 100)
        self.assertIn("DO NOT DEPLOY", audit_text(strat, df, SPEC,
                                                  {"expansion_confirmation": "completed"}))

    def test_blackbox_strategy_mechanism_not_verified(self):
        """A strategy that does not expose its signal column: lag/expansion/RC honest."""
        df = D.regime_trend_df()

        def run(frame, params):
            sig = frame["sig"].fillna(0).to_numpy()
            o = frame["open"].to_numpy()
            n = len(frame)
            pnl = trades = 0.0
            for i in range(n):
                if sig[i] > 0 and i + 5 < n:
                    pnl += o[i + 5] - o[i + 1]
                    trades += 1
            return {"pnl": pnl, "trades": int(trades)}

        strat = Strategy(name="blackbox", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 11})
        self.assertEqual(rep["overall"], "PASS")            # no fabricated P0
        self.assertIn("Look-ahead", rep["not_verified"])    # needs signal column
        notes = "\n".join(rep["sections"]["Robustness"]["notes"])
        self.assertIn("NOT VERIFIED", notes)                 # RC needs signal column


class TestOOSAndParam(unittest.TestCase):

    def test_oos_instability_conditional(self):
        """Price rises in-sample then falls out-of-sample -> OOS instability (P1)."""
        rets = np.concatenate([np.full(700, 0.01), np.full(300, -0.02)])
        df = _df_from_ret(rets)

        def run(frame, params):
            # long & hold the whole frame: IS pnl > 0, OOS pnl < 0
            return {"pnl": float(frame["close"].iloc[-1] - frame["close"].iloc[0]),
                    "trades": 1}

        strat = Strategy(name="hold", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"oos_frac": 0.3, "seed": 1})
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("OOS_INSTABILITY", codes)

    def test_param_cliff_detected(self):
        df = D.regime_trend_df()

        def run(frame, params):
            k = params["k"]
            return {"pnl": -1000.0 if k >= 9 else 10.0, "trades": 100}

        strat = Strategy(name="cliffy", run=run, default_params={"k": 1},
                         param_grid={"k": [1, 3, 5, 10]}, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 1})
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("PARAM_CLIFF", codes)
        # P2 does not change the verdict
        self.assertEqual(rep["overall"], "PASS")


class TestReportShape(unittest.TestCase):

    def test_sections_and_score_bounds(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("honest", df, "sig", D.next_open_hold(5))
        rep = audit(strat, df, SPEC, {"expansion_confirmation": "completed", "seed": 3})
        for name in ("Data Integrity", "Look-ahead", "Execution", "Statistics",
                     "Robustness", "Costs", "MTF"):
            self.assertIn(name, rep["sections"])
        self.assertGreaterEqual(rep["reliability_score"], 0)
        self.assertLessEqual(rep["reliability_score"], 100)
        json.dumps(to_jsonable(rep))

    def test_data_integrity_catches_bad_frame(self):
        bad = pd.DataFrame({"open": [1.0, -5.0], "close": [1.0, 2.0],
                            "high": [2.0, 3.0], "low": [0.5, 0.5]})
        bad.index = pd.to_datetime(["2026-01-01", "2026-01-01"])   # dup + nonpositive
        from validator.data_integrity import check as dcheck
        sec = dcheck(bad)
        self.assertEqual(sec["status"], "FAIL")
        codes = {i["code"] for i in sec["issues"]}
        self.assertIn("DATA_NONPOS", codes)
        self.assertIn("DATA_DUP", codes)


if __name__ == "__main__":
    unittest.main()
