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
                     "cost": {"fee_bps": 4.0, "slippage_bps": 2.0},
                     "seed": 11})
        # 4-state contract: MTF is on the roadmap -> NOT VERIFIED -> INCOMPLETE,
        # even though every implemented section is clean.
        self.assertEqual(rep["overall"], "INCOMPLETE")
        self.assertEqual(rep["blocking"]["P0"], 0)      # P2 may come from STAT_DEPENDENCE
        self.assertEqual(rep["blocking"]["P1"], 0)      # (honest overlapping rets)
        self.assertGreaterEqual(rep["verified_score"], 90)
        self.assertLess(rep["verified_score"], 100)       # two P3 hygiene items
        self.assertEqual(rep["reliability_score"], rep["verified_score"])
        self.assertEqual(rep["sections"]["Costs"]["status"], "DECLARED")
        self.assertIn("MTF", rep["not_verified"])
        self.assertNotIn("Costs", rep["not_verified"])
        self.assertFalse(rep["audit_complete"])
        self.assertIn("recommendation", rep)

    def test_pass_only_when_scope_complete(self):
        """PASS is reserved for a fully verified declared scope (V2.2 contract)."""
        df = D.regime_trend_df()
        strat = as_code_strategy("honest", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        cfg = {"expansion_confirmation": "completed",
               "cost": {"fee_bps": 4.0, "slippage_bps": 2.0}, "seed": 11,
               "scope": ["Data Integrity", "Look-ahead", "Execution",
                         "Statistics", "Robustness", "Costs"]}     # MTF out of scope
        rep = audit(strat, df, SPEC, cfg)
        self.assertEqual(rep["overall"], "PASS")
        self.assertTrue(rep["audit_complete"])
        self.assertNotIn("MTF", rep["not_verified"])
        self.assertIn("scope", rep)

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
        self.assertEqual(rep["issues"][0]["severity"], "P0")   # severity-ordered: P0 first
        self.assertLess(rep["verified_score"], 100)
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
        self.assertEqual(rep["overall"], "INCOMPLETE")       # no fabricated FAIL/PASS
        self.assertIn("Look-ahead", rep["not_verified"])    # needs signal column
        self.assertIn("Statistics", rep["not_verified"])    # no per-trade rets
        self.assertLess(rep["coverage_pct"], 100)
        self.assertFalse(rep["audit_complete"])
        notes = "\n".join(rep["sections"]["Robustness"]["notes"])
        self.assertIn("NOT VERIFIED", notes)                 # RC needs signal column

    def test_no_fake_clean_bill_for_unverified_scope(self):
        """ADVERSARIAL: a black box with no rets/signal/cost must NOT look clean.
        verified_score can be 100 over what WAS checked, but coverage < 100%,
        audit INCOMPLETE, and the recommendation says so."""
        df = D.regime_trend_df()

        def run(frame, params):
            return {"pnl": 0.0, "trades": 0}                # nothing assessable

        strat = Strategy(name="blackbox-empty", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 1})
        self.assertEqual(rep["verified_score"], 100)         # nothing found...
        self.assertLess(rep["coverage_pct"], 100)            # ...but almost nothing checked
        self.assertFalse(rep["audit_complete"])
        self.assertGreaterEqual(len(rep["not_verified"]), 3)
        self.assertIn("INCOMPLETE", audit_text(strat, df, SPEC, {"seed": 1}))

    def test_costs_three_states(self):
        from validator.costs import costs_check
        self.assertEqual(costs_check({})["status"], "NOT VERIFIED")
        declared = costs_check({"cost": {"commission": {"mode": "bps",
                                                        "open_rate": 5.0,
                                                        "close_rate": 5.0}}})
        self.assertEqual(declared["status"], "DECLARED")   # no per-trade fills
        log = [{"side": "long", "entry_price": 100.0, "exit_price": 101.0, "qty": 1.0}]
        verified = costs_check({"cost": {"commission": {"mode": "bps",
                                                        "open_rate": 5.0,
                                                        "close_rate": 5.0},
                                         "trades_log": log}})
        self.assertEqual(verified["status"], "VERIFIED")
        bad = costs_check({"cost": {"commission": {"mode": "bps",
                                                   "open_rate": -1.0,
                                                   "close_rate": 5.0}}})
        self.assertEqual(bad["status"], "FAIL")

    def test_oos_warmup_contract(self):
        """supports_from_bar=True -> OOS uses warm-up context; otherwise cold slice
        is flagged."""
        df = D.regime_trend_df()
        from validator.robustness import check as rcheck

        def run(frame, params):
            sig = frame["sig"].fillna(0).to_numpy()
            o = frame["open"].to_numpy()
            n = len(frame)
            from_bar = int(params.get("_from_bar", 0))
            pnl = trades = 0.0
            for i in range(n):
                if sig[i] > 0 and i >= from_bar and i + 5 < n:
                    pnl += o[i + 5] - o[i + 1]
                    trades += 1
            return {"pnl": pnl, "trades": int(trades)}

        warm = Strategy(name="w", run=run, entry_semantics="next_open",
                        supports_from_bar=True)
        cold = Strategy(name="c", run=run, entry_semantics="next_open",
                        supports_from_bar=False)
        nw = "\n".join(rcheck(warm, df, SPEC, {"oos_frac": 0.3, "seed": 1})["notes"])
        nc = "\n".join(rcheck(cold, df, SPEC, {"oos_frac": 0.3, "seed": 1})["notes"])
        self.assertIn("warm-up context", nw)
        self.assertIn("cold slice", nc)


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
        # no P0/P1 -> INCOMPLETE (MTF/Costs not verified), never silent PASS
        self.assertEqual(rep["overall"], "INCOMPLETE")
        self.assertEqual(rep["blocking"]["P0"], 0)


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
