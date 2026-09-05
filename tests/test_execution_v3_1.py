"""V3.1 — Execution / Information Boundary tests.

The classic cheat the df-level perturbation test cannot see: "decide at 09:35 close,
fill at 09:35 close" (no latency). It is only provable with per-trade timestamps.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, audit
from validator.execution import timeline_audit


def bars(n: int = 60, start: str = "2026-01-01 00:00", seconds: int = 300):
    idx = pd.date_range(start, periods=n, freq=f"{seconds}s")
    closes = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, n))
    o = np.empty(n); o[0] = closes[0]; o[1:] = closes[:-1]
    return pd.DataFrame({"open": o, "high": np.maximum(o, closes),
                         "low": np.minimum(o, closes), "close": closes}, index=idx)


class TestTimelineAudit(unittest.TestCase):

    def test_fill_at_signal_time_fails(self):
        trades = [{"signal_ts": 1000, "entry_ts": 1000},
                  {"signal_ts": 1300, "entry_ts": 1600}]
        rep = timeline_audit(trades)
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertEqual(len(rep["violations"]), 1)
        self.assertEqual(rep["violations"][0]["trade"], 0)

    def test_fill_after_signal_passes(self):
        trades = [{"signal_ts": 1000, "entry_ts": 1300},
                  {"signal_ts": 1300, "entry_ts": 1600}]
        rep = timeline_audit(trades)
        self.assertEqual(rep["verdict"], "PASS")

    def test_missing_timestamps_not_verified(self):
        rep = timeline_audit([{"entry_ts": 1}, {"signal_ts": 1, "entry_ts": 2}])
        self.assertEqual(rep["verdict"], "NOT VERIFIED")

    def test_datetime64_and_min_latency(self):
        a = np.datetime64("2026-01-01T00:00:00")
        rep = timeline_audit([{"signal_ts": a, "entry_ts": a + np.timedelta64(1, "s")}],
                             min_latency_s=0.0)
        self.assertEqual(rep["verdict"], "PASS")
        rep2 = timeline_audit([{"signal_ts": a, "entry_ts": a + np.timedelta64(1, "s")}],
                              min_latency_s=2.0)
        self.assertEqual(rep2["verdict"], "FAIL")


class TestTimelineInAudit(unittest.TestCase):

    def _df(self):
        return bars()

    def test_same_instant_close_fill_fails(self):
        """Close-fill cheat: signal at bar close AND fill at that same close. The
        perturbation test alone would miss it; the timeline catches it as P0."""
        df = self._df()
        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()

        def run(frame, params):
            n = len(frame)
            trades = []
            pnl = 0.0
            o = frame["open"].to_numpy()
            c = frame["close"].to_numpy()
            ts_close = frame.index.to_numpy()
            for i in range(1, n - 1):
                sig = 1.0 if c[i] > c[i - 1] else -1.0      # knowable at close i
                entry_ts = ts_close[i]                       # fill AT the same close
                entry_px = c[i]
                exit_ts = ts_close[i] + np.timedelta64(0, "s")  # not used by audit
                exit_px = o[i + 1]
                trades.append({"signal_ts": ts_close[i], "entry_ts": entry_ts,
                               "entry_price": entry_px, "exit_price": exit_px})
                pnl += (exit_px - entry_px) * sig
            return {"pnl": pnl, "trades": int(len(trades)),
                    "trades_log": trades}

        # NOTE: timeline reads run()'s optional 'trades_log' (list of per-trade dicts);
        # 'trades' stays an int count for the rest of the pipeline.
        def run_log(frame, params):
            res = run(frame, params)
            return {"pnl": res["pnl"], "trades": int(len(res["trades_log"])),
                    "trades_log": res["trades_log"]}

        strat = Strategy(name="close-fill", run=run_log, entry_semantics="next_open")
        rep = audit(strat, df, DataSpec(bar_seconds=300),
                    {"scope": ["Execution"], "seed": 1})
        self.assertEqual(rep["sections"]["Execution"]["status"], "FAIL")
        codes = {i["code"] for i in rep["issues"]}
        self.assertIn("EXECUTION_TIMELINE", codes)
        self.assertEqual(rep["overall"], "FAIL")

    def test_next_open_fill_passes(self):
        df = self._df()

        def run_log(frame, params):
            o = frame["open"].to_numpy()
            c = frame["close"].to_numpy()
            ts = frame.index.to_numpy()
            n = len(frame)
            trades, pnl = [], 0.0
            for i in range(1, n - 1):
                sig = 1.0 if c[i] > c[i - 1] else -1.0
                entry_ts = ts[i] + np.timedelta64(300, "s")   # next open (legal)
                pnl += (o[i + 1] - o[i + 1]) * sig            # zero, fine
                trades.append({"signal_ts": ts[i], "entry_ts": entry_ts})
            return {"pnl": pnl, "trades": int(len(trades)), "trades_log": trades}

        strat = Strategy(name="next-open", run=run_log, entry_semantics="next_open")
        rep = audit(strat, df, DataSpec(bar_seconds=300),
                    {"scope": ["Execution"], "seed": 1})
        codes = {i["code"] for i in rep["issues"]}
        self.assertNotIn("EXECUTION_TIMELINE", codes)
        self.assertIn("timeline=PASS", " ".join(rep["sections"]["Execution"]["notes"]))

    def test_no_timeline_not_verified_note(self):
        df = self._df()

        def run(frame, params):
            return {"pnl": 1.0, "trades": 10}                 # count only
        strat = Strategy(name="no-log", run=run, entry_semantics="next_open")
        rep = audit(strat, df, DataSpec(bar_seconds=300),
                    {"scope": ["Execution"], "seed": 1})
        notes = " ".join(rep["sections"]["Execution"]["notes"])
        self.assertIn("NOT VERIFIED", notes)


if __name__ == "__main__":
    unittest.main()
