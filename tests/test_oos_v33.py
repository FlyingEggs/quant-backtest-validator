"""V3.3 — OOS / Walk-Forward contract tests (boundary policy, parameter freeze,
contamination P0, WF consistency)."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit
from validator import wf
from validator.wf import (POLICIES, filter_trades, parameter_freeze_audit,
                          walk_forward_audit)
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def bars_from_ret(rets, start="2026-01-01 00:00", seconds=300):
    closes = 3000 * np.exp(0.001 * np.cumsum(rets))
    o = np.empty(len(closes)); o[0] = closes[0]; o[1:] = closes[:-1]
    idx = pd.date_range(start, periods=len(closes), freq=f"{seconds}s")
    return pd.DataFrame({"open": o, "high": np.maximum(o, closes),
                         "low": np.minimum(o, closes), "close": closes}, index=idx)


def honest_param_strategy(hold: int = 5):
    """next-open long-when-up strategy honouring params (hold bars) + trades_log."""

    def run(df, params):
        h = int(params.get("hold", hold))
        o = df["open"].to_numpy()
        c = df["close"].to_numpy()
        ts = df.index.to_numpy()
        n = len(df)
        trades, pnl = [], 0.0
        for i in range(1, n - h - 1):
            if c[i] > c[i - 1] and i >= int(params.get("_from_bar", 0)):
                e_ts = ts[i] + np.timedelta64(300, "s")
                x_ts = ts[i] + np.timedelta64(300 * (h + 1), "s")
                trades.append({"side": "long", "qty": 1.0,
                               "entry_ts": e_ts, "exit_ts": x_ts,
                               "signal_ts": ts[i],
                               "entry_price": float(o[i + 1]),
                               "exit_price": float(o[i + h + 1])})
        for t in trades:
            pnl += (t["exit_price"] - t["entry_price"]) * 1.0
        return {"pnl": pnl, "trades": int(len(trades)), "trades_log": trades}

    return Strategy(name="honest-wf", run=run, default_params={"hold": 5},
                    param_grid={"hold": [2, 5, 10]}, entry_semantics="next_open",
                    supports_from_bar=True)


def cheat_refit_strategy():
    """P0 contamination: ignores frozen params and 'fits' on whatever df it sees."""

    def run(df, params):
        o = df["open"].to_numpy()
        c = df["close"].to_numpy()
        ts = df.index.to_numpy()
        n = len(df)
        trades, pnl = [], 0.0
        for i in range(1, n - 6):
            if c[i] > c[i - 1]:
                trades.append({"side": "long", "qty": 1.0,
                               "entry_ts": ts[i] + np.timedelta64(300, "s"),
                               "exit_ts": ts[i] + np.timedelta64(1800, "s"),
                               "signal_ts": ts[i],
                               "entry_price": float(o[i + 1]),
                               "exit_price": float(o[i + 6])})
        for t in trades:
            pnl += (t["exit_price"] - t["entry_price"]) * 1.0
        return {"pnl": pnl, "trades": int(len(trades)), "trades_log": trades}

    return Strategy(name="cheat-refit", run=run, default_params={"hold": 5},
                    param_grid={"hold": [2, 5, 10]}, entry_semantics="next_open",
                    supports_from_bar=True)


class TestBoundaryPolicy(unittest.TestCase):

    def _cut_trades(self):
        t0 = pd.Timestamp("2026-01-02 00:00")
        t1 = pd.Timestamp("2026-01-03 00:00")
        inside = {"side": "long", "entry_ts": t0 + pd.Timedelta(hours=1),
                  "exit_ts": t0 + pd.Timedelta(hours=2), "qty": 1.0,
                  "entry_price": 100.0, "exit_price": 101.0}
        straddle = {"side": "long", "entry_ts": t0 - pd.Timedelta(minutes=1),
                    "exit_ts": t1 + pd.Timedelta(minutes=1), "qty": 1.0,
                    "entry_price": 100.0, "exit_price": 101.0}
        return [inside, straddle]

    def test_policy_semantics(self):
        lo = pd.Timestamp("2026-01-02 00:00").timestamp()
        hi = pd.Timestamp("2026-01-03 00:00").timestamp()
        for pol in POLICIES:
            f = filter_trades(self._cut_trades(), lo, hi, pol)
            self.assertEqual(f["policy"], pol)
        e = filter_trades(self._cut_trades(), lo, hi, "ENTRY_IN_WINDOW")
        self.assertEqual(len(e["kept"]), 1)          # straddle entry is outside
        f = filter_trades(self._cut_trades(), lo, hi, "EXIT_IN_WINDOW")
        self.assertEqual(len(f["kept"]), 1)          # straddle exit is outside
        full = filter_trades(self._cut_trades(), lo, hi, "FULL_TRADE_IN_WINDOW")
        self.assertEqual(len(full["kept"]), 1)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            filter_trades([], 0, 1, "BOGUS")


class TestParameterFreeze(unittest.TestCase):

    def test_honest_params_pass(self):
        df = D.regime_trend_df()
        pf = parameter_freeze_audit(honest_param_strategy(), df, {})
        self.assertEqual(pf["determinism"], "PASS")
        self.assertEqual(pf["refit_probe"], "PASS")   # different hold => different pnl

    def test_contamination_caught(self):
        """P0: declared tunable params, but OOS output identical across extremes."""
        df = D.regime_trend_df()
        pf = parameter_freeze_audit(cheat_refit_strategy(), df, {})
        self.assertEqual(pf["refit_probe"], "FAIL")
        codes = {i["code"] for i in pf["issues"]}
        self.assertIn("PARAM_FREEZE", codes)
        self.assertTrue(any(i["severity"] == "P0" for i in pf["issues"]))


class TestWalkForward(unittest.TestCase):

    def test_wf_contract_shape(self):
        n = 1600
        rng = np.random.default_rng(5)
        rets = np.where(np.arange(n) % 800 < 400, 0.001, -0.0002) + \
            rng.normal(0, 0.0015, n)
        df = bars_from_ret(rets)
        strat = honest_param_strategy()
        cfg = {"oos": {"policy": "FULL_TRADE_IN_WINDOW", "n_windows": 3,
                       "oos_bars": 200, "min_is_bars": 800,
                       "min_oos_trades": 3}}
        rep = walk_forward_audit(strat, df, cfg, SPEC)
        self.assertEqual(rep["policy"], "FULL_TRADE_IN_WINDOW")
        self.assertEqual(len(rep["windows"]), 3)
        for r in rep["windows"]:
            self.assertIn(r["status"], ("PASS", "FAIL", "INSUFFICIENT"))
        self.assertIn("positive_window_pct", rep)
        self.assertIn("expectancy_consistency_pct", rep)
        self.assertIn("trade_adequacy_pct", rep)

    def test_wf_in_audit_robustness(self):
        from validator.robustness import check as rcheck
        df = D.regime_trend_df()
        strat = honest_param_strategy()
        cfg = {"oos": {"policy": "ENTRY_IN_WINDOW", "n_windows": 2,
                       "oos_bars": 400, "min_is_bars": 1200,
                       "min_oos_trades": 2}, "seed": 1}
        sec = rcheck(strat, df, SPEC, cfg)
        self.assertIn("evidence", sec)
        self.assertEqual(sec["evidence"]["wf"]["policy"], "ENTRY_IN_WINDOW")
        self.assertIn("parameter freeze", " ".join(sec["notes"]))

    def test_contamination_fails_robustness_section(self):
        df = D.regime_trend_df()
        strat = cheat_refit_strategy()
        cfg = {"oos": {"policy": "FULL_TRADE_IN_WINDOW", "n_windows": 1,
                       "oos_bars": 300, "min_is_bars": 1500,
                       "min_oos_trades": 2}}
        sec = wf_check_robustness(strat, df, cfg)
        self.assertEqual(sec["status"], "FAIL")
        codes = {i["code"] for i in sec["issues"]}
        self.assertIn("PARAM_FREEZE", codes)


def wf_check_robustness(strategy, df, cfg):
    from validator.robustness import check as rcheck
    return rcheck(strategy, df, SPEC, cfg)


if __name__ == "__main__":
    unittest.main()
