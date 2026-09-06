"""V4.2 — Red-team attack audit (hostile-strategy developer perspective).

Each attack was actually run against the engine BEFORE the fix; the ones that got
through were fixed, the ones that were caught are locked here as regressions, and
known heuristic boundaries are asserted to stay honest (never a silent clean bill):

CAUGHT (fixed this round):
  * forged ledger with legal timeline but fill prices far outside the frame's bar
    range (entry 100 / exit 140 vs bars ~3160) was PASS -> EXEC_PRICE_UNREACHABLE
  * rets one-per-trade but VALUES unrelated to the ledger (50% on a 10% move) ->
    STAT_RETS_LEDGER_MISMATCH
  * a cost callable that raises crashed the whole audit -> section FAIL, no crash

BOUNDARY (heuristic, must not fake clean):
  * a shattered long-period signal still trips PERIOD_EXPANSION (sparse 0-runs);
    only an explicit expansion_confirmation lets it through - a declared choice.
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")
COST = {"commission": {"mode": "bps", "open_rate": 5.0, "close_rate": 5.0}}
DF = D.regime_trend_df(n=2000)


def _audit(run, scope, cost=True, cfg=None):
    strat = Strategy(name="red", run=run, entry_semantics="next_open")
    c = {"scope": scope, "seed": 1}
    if cost:
        c["cost"] = COST
    c.update(cfg or {})
    return audit(strat, DF, SPEC, c)


class TestPriceReachability(unittest.TestCase):

    def _liar(self, ep=100.0, xp=140.0):
        def run(frame, params):
            ts = frame.index
            return {"pnl": xp - ep, "trades": 1,
                    "trades_log": [{"side": "long", "qty": 1.0,
                                    "entry_price": ep, "exit_price": xp,
                                    "entry_ts": ts[10], "exit_ts": ts[11],
                                    "signal_ts": ts[9]}]}
        return run

    def test_unreachable_fill_prices_caught(self):
        rep = _audit(self._liar(), ["Execution", "Costs"])
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("EXEC_PRICE_UNREACHABLE", codes)
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")

    def test_consistent_prices_no_false_positive(self):
        # honest next_open_hold fills ARE inside the bars
        hdf = D.regime_trend_df(n=600)
        strat = as_code_strategy("h", hdf, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, hdf, SPEC, {"scope": ["Execution", "Costs"],
                                       "cost": COST, "seed": 1})
        self.assertNotIn("EXEC_PRICE_UNREACHABLE",
                         [i["code"] for i in rep["issues"]])


class TestRetsLedgerCrosscheck(unittest.TestCase):

    def _fake_rets(self, ret_values):
        def run(frame, params):
            ts = frame.index
            logs = [{"side": "long", "qty": 1.0, "entry_price": 100.0,
                     "exit_price": 110.0, "entry_ts": ts[10], "exit_ts": ts[11],
                     "signal_ts": ts[9]},
                    {"side": "long", "qty": 1.0, "entry_price": 100.0,
                     "exit_price": 110.0, "entry_ts": ts[20], "exit_ts": ts[21],
                     "signal_ts": ts[19]}]
            return {"pnl": 20.0, "trades": 2, "rets": list(ret_values),
                    "trades_log": logs}
        return run

    def test_fake_return_values_caught(self):
        rep = _audit(self._fake_rets([0.5, 0.5]), ["Statistics", "Costs"])
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("STAT_RETS_LEDGER_MISMATCH", codes)   # 50% vs ledger 10%
        self.assertEqual(rep["overall"], "CONDITIONAL PASS")

    def test_honest_returns_clean(self):
        hdf = D.regime_trend_df(n=600)
        strat = as_code_strategy("h", hdf, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, hdf, SPEC, {"scope": ["Statistics", "Costs"],
                                       "cost": COST, "seed": 1})
        self.assertNotIn("STAT_RETS_LEDGER_MISMATCH",
                         [i["code"] for i in rep["issues"]])


class TestNoCrashOnHostileConfig(unittest.TestCase):

    def test_raising_cost_callable_fails_section_not_crash(self):
        def boom(p, q, k):
            raise ValueError("boom")
        tr = {"side": "long", "qty": 1.0, "entry_price": 100.0,
              "exit_price": 110.0, "entry_ts": pd.Timestamp("2026-01-01"),
              "exit_ts": pd.Timestamp("2026-01-02")}

        def run(frame, params):
            return {"pnl": 10.0, "trades": 1, "trades_log": [tr]}
        rep = _audit(run, ["Costs"], cost=False,
                     cfg={"cost": {"slippage": {"mode": "callable", "fn": boom}}})
        self.assertEqual(rep["overall"], "FAIL")
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("SECTION_ERROR", codes)      # failed loudly, no crash


class TestShatteredExpansionBoundary(unittest.TestCase):
    """Sparse/short-run signals still trip PERIOD_EXPANSION (0-runs are long);
    only an explicit confirmation lets them through. The heuristic never fakes."""

    def _shattered(self):
        import numpy as np
        n = 2000
        rng = np.random.default_rng(2)
        ret = rng.normal(0.0002, 0.0015, n)
        closes = 3000 * np.cumprod(1 + ret)
        f = D.frame(closes)
        sig = np.zeros(n)
        for a, b in [(100, 120), (300, 318), (700, 718), (1200, 1220), (1600, 1615)]:
            sig[a:b] = 1.0
        f["sig"] = sig
        return f

    def _bt(self, fr):
        import numpy as np
        sg = fr["sig"].fillna(0).to_numpy()
        o = fr["open"].to_numpy()
        pnl, tr = 0.0, 0
        for i in range(1, len(fr) - 1):
            if sg[i] == 1.0:
                pnl += (o[i + 1] - o[i])
                tr += 1
        return {"pnl": pnl, "trades": tr}

    def test_unconfirmed_shattered_signal_not_clean(self):
        f = self._shattered()
        strat = as_code_strategy("shat", f, "sig", self._bt,
                                 entry_semantics="next_open")
        rep = audit(strat, f, SPEC,
                    {"scope": ["Data Integrity", "Look-ahead", "Execution"],
                     "seed": 1})
        self.assertEqual(rep["overall"], "FAIL")     # never a silent PASS
        self.assertIn("PERIOD_EXPANSION", [i["code"] for i in rep["issues"]])

    def test_explicit_confirmation_is_the_escape_hatch(self):
        f = self._shattered()
        strat = as_code_strategy("shat", f, "sig", self._bt,
                                 entry_semantics="next_open")
        rep = audit(strat, f, SPEC,
                    {"scope": ["Data Integrity", "Look-ahead", "Execution"],
                     "seed": 1, "expansion_confirmation": "completed"})
        # declared choice: the signal's low-frequency provenance is confirmed
        self.assertNotIn("PERIOD_EXPANSION", [i["code"] for i in rep["issues"]])


if __name__ == "__main__":
    unittest.main()
