"""Boundary round 3 — DECLARED != VERIFIED (coverage), degenerate RC null,
parameter-sign sweep edges, and the legal intraday-holding FP fix."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit
from validator.robustness import check as rcheck
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")

SCOPE_NO_MTF_NO_LA = ["Data Integrity", "Execution", "Statistics",
                      "Robustness", "Costs"]
COST = {"commission": {"mode": "bps", "open_rate": 5.0, "close_rate": 5.0}}


def plain_run(df, params, with_log=False):
    n = len(df)
    rets = np.random.default_rng(3).normal(0, 1, 200).tolist()
    res = {"pnl": 10.0, "trades": 200, "rets": rets}
    if with_log:
        # ledger consistent with the headline: 200 legs, +0.05 gross each = pnl 10
        ts = df.index.to_numpy() if isinstance(df.index, pd.DatetimeIndex) else None
        res["trades_log"] = [{"side": "long", "qty": 1.0,
                              "signal_ts": ts[0], "entry_ts": ts[min(i + 1, n - 1)],
                              "exit_ts": ts[min(i + 2, n - 1)],
                              "entry_price": 100.0, "exit_price": 100.05}
                             for i in range(200)]
    return res


class TestDeclaredIsNotVerified(unittest.TestCase):
    """DECLARED cost assumptions must keep the audit INCOMPLETE (PASS needs VERIFIED)."""

    def _audit(self, with_log):
        df = D.regime_trend_df()
        strat = Strategy(name="s", run=lambda f, p: plain_run(f, p, with_log),
                         entry_semantics="next_open")
        cfg = {"scope": SCOPE_NO_MTF_NO_LA, "cost": dict(COST), "seed": 1}
        return audit(strat, df, SPEC, cfg)

    def test_declared_without_logs_is_incomplete(self):
        rep = self._audit(with_log=False)
        self.assertEqual(rep["sections"]["Costs"]["status"], "DECLARED")
        self.assertEqual(rep["overall"], "INCOMPLETE")
        self.assertIn("Costs", rep["declared"])
        self.assertLess(rep["coverage_pct"], 100)

    def test_verified_with_logs_can_pass(self):
        rep = self._audit(with_log=True)
        self.assertEqual(rep["sections"]["Costs"]["status"], "VERIFIED")
        self.assertEqual(rep["overall"], "PASS")
        self.assertEqual(rep["coverage_pct"], 100)
        self.assertEqual(rep["declared"], [])


class TestDegenerateRC(unittest.TestCase):
    """A shuffled null that mostly produces no trades is weak evidence - must be said."""

    def test_degenerate_null_noted(self):
        df = D.regime_trend_df().copy()
        sig = np.full(len(df), -1.0)
        sig[500:510] = 1.0                       # one short +1 run: ~8 real pairs
        df["sig"] = sig

        def bt(frame):
            s = frame["sig"].fillna(0).to_numpy()
            o = frame["open"].to_numpy()
            n = len(frame)
            pnl = trades = 0.0
            for i in range(2, n):
                if s[i] == s[i - 1] == 1.0 and i + 5 < n:   # needs a PAIR (rare after shuffle)
                    pnl += o[i + 5] - o[i + 1]
                    trades += 1
            return {"pnl": pnl, "trades": int(trades)}
        strat = Strategy(name="run-dependent", run=lambda f, p: bt(f),
                         entry_semantics="next_open")
        strat.signal_col = "sig"
        strat.bt_mechanism = bt
        sec = rcheck(strat, df, SPEC, {"seed": 7, "n_shuffles": 200})
        notes = " ".join(sec["notes"])
        self.assertIn("DEGENERATE", notes)


class TestParamSignEdges(unittest.TestCase):

    def _sec(self, pnls):
        df = D.regime_trend_df()

        def run(frame, params):
            return {"pnl": float(pnls[int(float(params["k"]))]), "trades": 100}
        strat = Strategy(name="p", run=run, default_params={"k": 0},
                         param_grid={"k": [str(i) for i in range(len(pnls))]},
                         entry_semantics="next_open")
        return rcheck(strat, df, SPEC, {"seed": 1})

    def test_negative_monotone_not_flagged(self):
        sec = self._sec([-10.0, -11.0, -12.0, -13.0])
        codes = {i["code"] for i in sec["issues"]}
        self.assertNotIn("PARAM_CLIFF", codes)
        self.assertNotIn("PARAM_OSCILLATION", codes)

    def test_single_spike_flagged(self):
        sec = self._sec([0.0, 0.0, 10.0, 0.0])
        codes = {i["code"] for i in sec["issues"]}
        self.assertIn("PARAM_CLIFF", codes)


class TestLegalIntradayHoldNotKilled(unittest.TestCase):
    """A LEGAL next-open strategy that holds open->close of the entry bar has ~0%
    retention under the fill shift. That is a real short-horizon edge, not a
    same-bar cheat - the perturbation must stay P1 evidence, not auto-FAIL."""

    def test_not_fail_without_timeline(self):
        df = D.same_bar_leak_df()

        def run(frame, params):
            s = frame["sig"].fillna(0).to_numpy()
            o = frame["open"].to_numpy()
            c = frame["close"].to_numpy()
            n = len(frame)
            pnl = trades = 0.0
            for i in range(1, n - 1):
                if s[i] == 1.0:                       # enter NEXT open, exit same-bar close
                    pnl += (c[i + 1] - o[i + 1]) * 1.0
                    trades += 1
            return {"pnl": pnl, "trades": int(trades)}
        strat = Strategy(name="intraday-hold", run=run, entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"scope": ["Execution"], "seed": 1})
        codes = {i["code"] for i in rep["issues"]}
        self.assertNotEqual(rep["overall"], "FAIL")    # legal strategy not murdered
        self.assertIn("EXECUTION_FILL", codes)
        self.assertEqual([i["severity"] for i in rep["issues"]
                          if i["code"] == "EXECUTION_FILL"], ["P1"])


if __name__ == "__main__":
    unittest.main()
