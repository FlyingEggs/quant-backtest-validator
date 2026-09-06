"""V4.1 — integrity-chain round (external audit): ledger/PnL consistency, WF
frozen-parameter main path, real MTF CLOSE semantics, rets/data temporal
contracts. Each was reproduced as a real gap before the fix.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit
from validator import data_integrity as di
from validator import statistics as stats_mod
from validator.mtf import temporal_availability
from validator.wf import walk_forward_audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def ledger_trade(k=1.0, ep=100.0, ret=1.0):
    return {"side": "long", "qty": k, "entry_price": ep, "exit_price": ep + ret,
            "entry_ts": pd.Timestamp("2026-01-01"),
            "exit_ts": pd.Timestamp("2026-01-02")}


class TestLedgerPnLConsistency(unittest.TestCase):
    """Headline pnl must equal the gross pnl implied by the strategy's own
    trades_log - the ledger is authoritative, the claim is checked (P0)."""

    def _audit(self, pnl, trades_log, cost=True):
        def run(frame, params):
            return {"pnl": pnl, "trades": len(trades_log),
                    "trades_log": trades_log}
        strat = Strategy(name="s", run=run, entry_semantics="next_open")
        cfg = {"scope": ["Costs"], "seed": 1}
        if cost:
            cfg["cost"] = {"commission": {"mode": "bps", "open_rate": 5.0,
                                          "close_rate": 5.0}}
        return audit(strat, D.regime_trend_df(n=300), SPEC, cfg)

    def test_claimed_pnl_disconnected_from_ledger_fails(self):
        rep = self._audit(999999.0, [ledger_trade()])   # ledger gross = 1.0
        sec = rep["sections"]["Costs"]
        self.assertEqual(sec["status"], "FAIL")
        self.assertIn("TRADE_LEDGER_PNL_MISMATCH",
                      [i["code"] for i in sec["issues"]])
        self.assertEqual(rep["overall"], "FAIL")

    def test_consistent_pnl_verified(self):
        rep = self._audit(1.0, [ledger_trade(ret=1.0)])
        sec = rep["sections"]["Costs"]
        self.assertEqual(sec["status"], "VERIFIED")
        self.assertNotIn("TRADE_LEDGER_PNL_MISMATCH",
                         [i["code"] for i in sec["issues"]])

    def test_consistent_multileg_pnl_verified(self):
        # long + short legs net to zero-ish ledger
        legs = [ledger_trade(ret=5.0),
                {"side": "short", "qty": 1.0, "entry_price": 105.0,
                 "exit_price": 100.0,
                 "entry_ts": pd.Timestamp("2026-01-02"),
                 "exit_ts": pd.Timestamp("2026-01-03")}]
        rep = self._audit(5.0 + 5.0, legs)
        self.assertEqual(rep["sections"]["Costs"]["status"], "VERIFIED")


class TestWalkForwardFrozenPath(unittest.TestCase):
    """WF main loop runs IS -> fit_is() -> frozen params -> OOS when the
    provenance contract is declared; per-window source is reported."""

    def _strat(self, with_contract=True):
        def run(frame, params):
            k = float(params.get("k", 1.0))
            n = int(k)
            ts = frame.index
            i0 = max(0, int(len(frame) * 0.5))
            i1 = min(len(frame) - 1, i0 + 1)
            log = [{"side": "long", "qty": 1.0, "entry_price": 100.0,
                    "exit_price": 110.0,
                    "entry_ts": ts[i0], "exit_ts": ts[i1]}
                   for _ in range(n)]
            return {"pnl": 10.0 * n, "trades": n, "trades_log": log}
        kw = dict(entry_semantics="next_open")
        if with_contract:
            kw.update(fit_is=lambda df: {"k": 7.0}, accepts_frozen=True)
        return Strategy(name="wf", run=run, default_params={"k": 1.0},
                        param_grid={"k": [1.0, 7.0]}, **kw)

    def _wf(self, strat):
        return walk_forward_audit(strat, D.regime_trend_df(n=1500),
                                  {"oos": {"n_windows": 2, "oos_bars": 200,
                                           "min_is_bars": 300}}, SPEC)

    def test_frozen_contract_used_in_windows(self):
        rep = self._wf(self._strat(with_contract=True))
        self.assertTrue(rep["windows"])
        for w in rep["windows"]:
            self.assertEqual(w["params_source"], "frozen_fit")
            self.assertIsNotNone(w["frozen_params_hash"])
            self.assertEqual(w["oos_pnl"], 70.0)   # k=7 * 10, 7 trades

    def test_no_contract_reports_defaults(self):
        rep = self._wf(self._strat(with_contract=False))
        for w in rep["windows"]:
            self.assertEqual(w["params_source"], "defaults")
            self.assertIsNone(w["frozen_params_hash"])
            self.assertEqual(w["oos_pnl"], 10.0)   # k=1


class TestMtfCloseSemantics(unittest.TestCase):
    """CLOSE-indexed frames must be computed under CLOSE semantics, not warned."""

    def _pair(self):
        rng = np.random.default_rng(3)
        hours = 6
        hi_idx = pd.date_range("2026-01-01 01:00", periods=hours, freq="3600s")
        hc = 100.0 + np.cumsum(rng.normal(0, 0.5, hours))
        high = pd.DataFrame({"close": hc}, index=hi_idx)
        n = hours * 12
        low_idx = pd.date_range("2026-01-01 00:05", periods=n, freq="300s")
        low = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                            "close": 100.0}, index=low_idx)
        sh = np.sign(np.diff(hc, prepend=hc[0]))
        hidx = np.arange(n) // 12
        pos = np.arange(n) % 12
        low["sig_legal"] = np.where(pos < 11, sh[np.maximum(hidx - 1, 0)], sh[hidx])
        low["sig_naive"] = sh[hidx]
        return low, high

    def test_close_semantics_naive_leak_detected(self):
        low, high = self._pair()
        rep = temporal_availability(low, "sig_naive", high, 3600, 300,
                                    transform="sign_diff", semantics="CLOSE")
        self.assertEqual(rep["verdict"], "FAIL")          # current-hour use = leak
        self.assertGreaterEqual(rep["leak_frac"], 0.9)

    def test_close_semantics_legal_passes(self):
        # high closes at 00:00/01:00; every low bar closes AFTER 01:00 and carries
        # the last CLOSED hour's sign (+1): under CLOSE semantics nothing is
        # forming -> PASS. Under OPEN semantics the same frame reads the 01:00
        # high as still forming at 01:05+ -> NOT PASS. The contrast proves the
        # semantics changes the computation, not just the warning.
        high = pd.DataFrame({"close": [100.0, 110.0]},
                            index=pd.DatetimeIndex(["2026-01-01 00:00",
                                                    "2026-01-01 01:00"]))
        low_idx = pd.date_range("2026-01-01 01:05", periods=11, freq="300s")
        low = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                            "close": 100.0, "sig_legal": 1.0}, index=low_idx)
        close_rep = temporal_availability(low, "sig_legal", high, 3600, 300,
                                          transform="sign_diff", semantics="CLOSE")
        self.assertEqual(close_rep["verdict"], "PASS")
        open_rep = temporal_availability(low, "sig_legal", high, 3600, 300,
                                         transform="sign_diff", semantics="OPEN")
        self.assertNotEqual(open_rep["verdict"], "PASS")

    def test_open_semantics_default_unchanged(self):
        low, high = self._pair()
        a = temporal_availability(low, "sig_legal", high, 3600, 300,
                                  transform="sign_diff")          # default OPEN
        b = temporal_availability(low, "sig_legal", high, 3600, 300,
                                  transform="sign_diff", semantics="OPEN")
        self.assertEqual(a["verdict"], b["verdict"])

    def test_mtf_check_honours_spec_semantics(self):
        from validator import mtf
        high = pd.DataFrame({"close": [100.0, 110.0]},
                            index=pd.DatetimeIndex(["2026-01-01 00:00",
                                                    "2026-01-01 01:00"]))
        low_idx = pd.date_range("2026-01-01 01:05", periods=11, freq="300s")
        low = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                            "close": 100.0, "sig_legal": 1.0}, index=low_idx)
        spec = DataSpec(bar_seconds=300, bar_timestamp_semantics="CLOSE",
                        timeframes={"h1": high})
        rep = mtf.check(low, spec, {"mtf": {"col": "sig_legal", "frame": "h1",
                                            "frame_seconds": 3600,
                                            "transform": "sign_diff"}})
        self.assertEqual(rep["status"], "PASS")           # computed, not warned


class TestReturnsTemporalContract(unittest.TestCase):

    def test_rets_len_mismatch_trades_reported(self):
        def run(frame, params):
            return {"pnl": 1.0, "trades": 100, "rets": [0.01] * 50}
        strat = Strategy(name="bad", run=run, entry_semantics="next_open")
        rep = stats_mod.check(strat, D.regime_trend_df(n=200), SPEC, {})
        self.assertEqual(rep["status"], "CONDITIONAL PASS")
        self.assertIn("STAT_RETS_TRADE_MISMATCH",
                      [i["code"] for i in rep["issues"]])

    def test_rets_one_per_trade_clean(self):
        def run(frame, params):
            return {"pnl": 1.0, "trades": 50, "rets": [0.01] * 50}
        strat = Strategy(name="ok", run=run, entry_semantics="next_open")
        rep = stats_mod.check(strat, D.regime_trend_df(n=200), SPEC, {})
        self.assertNotIn("STAT_RETS_TRADE_MISMATCH",
                         [i["code"] for i in rep["issues"]])


class TestDataUniformity(unittest.TestCase):

    def test_gap_frame_reported(self):
        idx = [pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(seconds=300 * i)
               for i in (0, 1, 2, 3, 4, 11)]           # 6-bar hole after 4
        df = pd.DataFrame({"open": [1.0] * 6, "high": [1.0] * 6,
                           "low": [1.0] * 6, "close": [1.0] * 6}, index=idx)
        rep = di.check(df, DataSpec(bar_seconds=300))
        self.assertIn("DATA_NONUNIFORM", [i["code"] for i in rep["issues"]])

    def test_uniform_frame_clean(self):
        df = D.regime_trend_df(n=300)
        rep = di.check(df, SPEC)
        self.assertNotIn("DATA_NONUNIFORM", [i["code"] for i in rep["issues"]])

    def test_no_spec_no_uniformity_check(self):
        idx = [pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(seconds=300 * i)
               for i in (0, 1, 2, 3, 4, 11)]
        df = pd.DataFrame({"open": [1.0] * 6, "high": [1.0] * 6,
                           "low": [1.0] * 6, "close": [1.0] * 6}, index=idx)
        rep = di.check(df, None)
        self.assertNotIn("DATA_NONUNIFORM", [i["code"] for i in rep["issues"]])


if __name__ == "__main__":
    unittest.main()
