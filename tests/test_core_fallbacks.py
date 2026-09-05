"""V3.4.2 — _chi2_sf fallback/cache contract (control flow was restructured during
the mypy pass; this locks the new semantics).

Under test:
  * scipy importable  -> exact scipy.stats.chi2.sf, imported ONCE and cached
  * scipy unavailable -> Wilson-Hilferty normal approximation (numeric, in (0,1))
  * a failed import is PERMANENT (no per-call retry) - the old None-means-retry
    behaviour was replaced by an explicit TRIED flag
"""

import builtins
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import core

try:
    from scipy import stats as _sp   # type: ignore[import-untyped]
    _EXACT = _sp.chi2.sf
    HAVE_SCIPY = True
except Exception:
    _EXACT = None
    HAVE_SCIPY = False


def _wilson_hilferty(x: float, df: int) -> float:
    """Independent re-implementation of the fallback, for cross-checking."""
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class TestChi2Fallback(unittest.TestCase):

    def setUp(self):
        core._SCIPY_CHI2_TRIED = False
        core._SCIPY_CHI2_SF = None

    def _blocking_import(self):
        """Import hook that raises ImportError for scipy only; counts attempts."""
        real = builtins.__import__
        state = {"calls": 0}

        def fake(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                state["calls"] += 1
                raise ImportError("scipy blocked for test")
            return real(name, *args, **kwargs)
        return fake, state

    def test_fallback_numeric_when_scipy_unavailable(self):
        fake, _ = self._blocking_import()
        with mock.patch("builtins.__import__", side_effect=fake):
            for x, df in [(3.84, 1), (21.0, 10), (124.0, 100)]:
                v = core._chi2_sf(x, df)
                self.assertTrue(np.isfinite(v))
                self.assertTrue(0.0 < v < 1.0)
                # fallback must match the independent Wilson-Hilferty form exactly
                self.assertAlmostEqual(v, _wilson_hilferty(x, df), places=12)

    def test_fallback_approximation_quality_when_scipy_absent(self):
        # With scipy present in the test env we can quantify the approximation gap:
        # tight at df>=10 (validator-grade), loose at df=1 (known WH weakness).
        if not HAVE_SCIPY:
            self.skipTest("scipy not installed - cannot cross-check approximation")
        fake, _ = self._blocking_import()
        with mock.patch("builtins.__import__", side_effect=fake):
            self.assertAlmostEqual(core._chi2_sf(21.0, 10), _EXACT(21.0, 10), delta=5e-3)
            self.assertAlmostEqual(core._chi2_sf(124.0, 100), _EXACT(124.0, 100), delta=1e-4)
            self.assertTrue(0.0 < core._chi2_sf(3.84, 1) < 1.0)  # no range assertion

    def test_failed_import_is_permanent_no_retry(self):
        fake, state = self._blocking_import()
        with mock.patch("builtins.__import__", side_effect=fake):
            v1 = core._chi2_sf(10.0, 5)
            v2 = core._chi2_sf(20.0, 5)      # second call: must NOT re-import
            self.assertEqual(state["calls"], 1, "TRIED flag must make a failed "
                             "scipy import permanent")
            self.assertAlmostEqual(v1, _wilson_hilferty(10.0, 5), places=12)
            self.assertAlmostEqual(v2, _wilson_hilferty(20.0, 5), places=12)

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_exact_when_scipy_available(self):
        self.assertEqual(core._chi2_sf(3.84, 1), _EXACT(3.84, 1))
        self.assertEqual(core._chi2_sf(21.0, 10), _EXACT(21.0, 10))
        self.assertEqual(core._chi2_sf(124.0, 100), _EXACT(124.0, 100))

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_success_is_cached_imported_once(self):
        real = builtins.__import__
        state = {"calls": 0}

        def counting(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                state["calls"] += 1
            return real(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=counting):
            a = core._chi2_sf(10.0, 5)
            b = core._chi2_sf(10.0, 5)
            self.assertEqual(state["calls"], 1, "successful scipy resolve is cached")
            self.assertEqual(a, b)
            self.assertEqual(a, _EXACT(10.0, 5))


if __name__ == "__main__":
    unittest.main()
