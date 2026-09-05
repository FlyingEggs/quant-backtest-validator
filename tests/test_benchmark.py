"""Benchmark — audit() wall time on realistic frame sizes (RC = 200 shuffles).

Deliberately NOT part of the default suite: timings are machine-dependent and the
minute-scale case is slow with the pure-Python strategy loop. Gate with:

    RUN_BENCHMARK=1 python3 -m unittest tests.test_benchmark -v      # daily + 5-min
    RUN_BENCHMARK=1 BENCH_BIG=1 python3 -m unittest tests.test_benchmark -v

prints per-case wall time and asserts a generous ceiling (regression tripwire,
not a performance promise). The README Performance table carries the measured
numbers.
"""

import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import audit, DataSpec, as_code_strategy
from examples import demo as D

SKIP = not os.environ.get("RUN_BENCHMARK")


def _frame(n: int, bar_seconds: int) -> pd.DataFrame:
    """Deterministic OHLCV-ish frame of `n` bars at `bar_seconds` freq with a `sig`
    column that actually flips (so the strategy trades)."""
    rng = np.random.default_rng(11)
    ret = rng.normal(0.0001, 0.0015, n)
    closes = 3000.0 * np.cumprod(1 + ret)
    df = D.frame(closes, bar_seconds)
    e20 = pd.Series(closes).ewm(span=20, adjust=False).mean().shift(1)
    e60 = pd.Series(closes).ewm(span=60, adjust=False).mean().shift(1)
    df["sig"] = np.where(e20 > e60, 1.0, -1.0)
    return df


def _timed_audit(n: int, bar_seconds: int, label: str, ceiling_s: float,
                 n_shuffles: int = 200) -> float:
    df = _frame(n, bar_seconds)
    bt = D.next_open_hold(5)
    strat = as_code_strategy("bench", df, "sig", bt, entry_semantics="next_open")
    spec = DataSpec(bar_seconds=bar_seconds, source="synthetic")
    t0 = time.perf_counter()
    audit(strat, df, spec, {"n_shuffles": n_shuffles, "seed": 11})
    dt = time.perf_counter() - t0
    print(f"[bench] {label:<16} n={n:>7}  dt={dt:7.1f}s  "
          f"shuffles={n_shuffles}", flush=True)
    assert dt < ceiling_s, f"{label} took {dt:.1f}s > ceiling {ceiling_s:.0f}s"
    return dt


@unittest.skipIf(SKIP, "set RUN_BENCHMARK=1 to measure audit() wall time")
class TestAuditBenchmark(unittest.TestCase):

    def test_daily_scale(self):
        # ~10y and ~5y of daily bars
        _timed_audit(2520, 86400, "10y daily", 90.0)
        _timed_audit(1260, 86400, "5y daily", 60.0)

    def test_fivemin_scale(self):
        # ~1y of 5-minute bars (~105k rows), full RC=200
        _timed_audit(105120, 300, "1y 5-min", 600.0)

    @unittest.skipIf(not os.environ.get("BENCH_BIG"),
                     "set BENCH_BIG=1 for the slow minute-scale case")
    def test_minute_scale_scaled_shuffles(self):
        # ~1y of minute bars (525k rows): pure-Python strategy loop makes a full
        # RC=200 pass prohibitively slow, so this case is measured at 50 shuffles
        # and the RC cost scales ~linearly with n_shuffles (reported, not hidden).
        _timed_audit(525600, 60, "1y min(50)", 900.0, n_shuffles=50)


if __name__ == "__main__":
    unittest.main()
