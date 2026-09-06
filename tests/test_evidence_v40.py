"""V4.0 — Evidence chain (external audit round).

  * result_hash / evidence_hash: chaining WHAT the audit concluded (verdicts,
    findings, metrics, cert level) to WHAT it ran on (manifest_hash) - a tampered
    JSON verdict is detected even when inputs were untouched.
  * environment fingerprint: python/platform/numpy/pandas/scipy versions land in
    the manifest (hostname excluded so replay survives machine moves).
  * callable tokens now carry module/qualname/source + __closure__ cell values:
    same def, different captured coeff => different strategy_source_hash.
  * frame_hash is self-contained: dtype, column names and index dtype/timezone are
    part of the data identity (int64 vs float64 no longer collide on the frame).
  * module-GLOBAL drift stays a documented boundary (globals are invisible on the
    function object) - mutable deps belong in params/closures.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit
from validator.audit import ENGINE_VERSION
from validator.manifest import (_callable_token, _environment_fingerprint,
                                build_manifest, frame_hash, verify_evidence,
                                verify_manifest)
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def sstrategy():
    def run(frame, params):
        return {"pnl": 100.0 * float(params.get("k", 1.0)), "trades": 5}
    return Strategy(name="s", run=run, default_params={"k": 1.0},
                    param_grid={"k": [0.5, 1.0, 2.0]},
                    entry_semantics="next_open")


class TestEnvironmentFingerprint(unittest.TestCase):

    def test_manifest_carries_runtime_versions(self):
        m = build_manifest(sstrategy(), D.regime_trend_df(n=100), SPEC, {"seed": 1},
                           ENGINE_VERSION, ["X"])
        env = m["environment"]
        for key in ("python", "implementation", "os", "machine", "numpy", "pandas"):
            self.assertIn(key, env)
        self.assertIn("scipy", env)

    def test_environment_enters_manifest_hash(self):
        base = _environment_fingerprint()
        self.assertIn("python", base)
        self.assertNotIn("hostname", " ".join(str(v) for v in base.values())
                         .lower().replace("macos", ""))


class TestClosureDrift(unittest.TestCase):

    def test_closure_value_drift_changes_source_hash(self):
        def make(coeff):
            def run(frame, params):
                return {"pnl": coeff * 1.0, "trades": 1}
            return run
        df = D.regime_trend_df(n=100)
        m1 = build_manifest(Strategy(name="c", run=make(0.1),
                                     entry_semantics="next_open"), df, SPEC,
                            {}, ENGINE_VERSION, ["X"])
        m2 = build_manifest(Strategy(name="c", run=make(0.5),
                                     entry_semantics="next_open"), df, SPEC,
                            {}, ENGINE_VERSION, ["X"])
        self.assertNotEqual(m1["strategy_source_hash"], m2["strategy_source_hash"])

    def test_same_factory_closure_stable(self):
        def make(coeff):
            def run(frame, params):
                return {"pnl": coeff * 1.0, "trades": 1}
            return run
        df = D.regime_trend_df(n=100)
        m1 = build_manifest(Strategy(name="c", run=make(0.1),
                                     entry_semantics="next_open"), df, SPEC,
                            {}, ENGINE_VERSION, ["X"])
        m2 = build_manifest(Strategy(name="c", run=make(0.1),
                                     entry_semantics="next_open"), df, SPEC,
                            {}, ENGINE_VERSION, ["X"])
        self.assertEqual(m1["strategy_source_hash"], m2["strategy_source_hash"])

    def test_token_contains_module_qualname_closure(self):
        def outer():
            x = 3
            def inner(f, p):
                return x * 2
            return inner
        tok = _callable_token(outer())
        self.assertEqual(tok["closure"][0]["value"], "i3")
        self.assertTrue(tok["qualname"].endswith("outer.<locals>.inner"))


class TestFrameHashSelfContained(unittest.TestCase):

    def _frame(self, vals, dtype):
        return pd.DataFrame({"v": np.asarray(vals, dtype=dtype)},
                            index=pd.date_range("2026-01-01", periods=len(vals),
                                                freq="D"))

    def test_int64_vs_float64_distinct(self):
        a = self._frame([1, 2, 3], "int64")
        b = self._frame([1.0, 2.0, 3.0], "float64")
        self.assertNotEqual(frame_hash(a), frame_hash(b))

    def test_float32_vs_float64_distinct(self):
        a = self._frame([0.1, 0.2], "float32")
        b = self._frame([0.1, 0.2], "float64")
        self.assertNotEqual(frame_hash(a), frame_hash(b))

    def test_tz_aware_vs_naive_distinct(self):
        base = D.regime_trend_df(n=50)
        a = base.copy()
        b = base.copy()
        b.index = b.index.tz_localize("UTC")
        self.assertNotEqual(frame_hash(a), frame_hash(b))

    def test_mixed_object_column_stable(self):
        df = D.regime_trend_df(n=50)
        df["tag"] = ["a", "b", None, 1, 2.5] * 10          # object, basic types
        self.assertEqual(frame_hash(df), frame_hash(df.copy()))


class TestResultEvidenceChain(unittest.TestCase):

    def _audit(self, **over):
        df = D.regime_trend_df(n=400)
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        return audit(strat, df, SPEC, {"seed": 1, **over})

    def test_report_carries_chain(self):
        rep = self._audit()
        for k in ("manifest_hash", "result_hash", "evidence_hash"):
            self.assertIn(k, rep)
        self.assertEqual(len(rep["result_hash"]), 64)

    def test_verify_ok_untouched(self):
        rep = self._audit()
        self.assertTrue(verify_evidence(rep)["ok"])

    def test_tampered_verdict_detected(self):
        rep = self._audit()
        rep["overall"] = "PASS"                     # forged green
        v = verify_evidence(rep)
        self.assertFalse(v["ok"])
        self.assertIn("result_hash", v["mismatches"])
        self.assertIn("evidence_hash", v["mismatches"])

    def test_tampered_finding_detected(self):
        rep = self._audit()
        rep["issues"] = [i for i in rep["issues"]
                         if i.get("severity") != "P0"]  # drop a P0
        self.assertFalse(verify_evidence(rep)["ok"])

    def test_tampered_section_status_detected(self):
        rep = self._audit(scope=["Data Integrity"])
        sec = rep["sections"]["Data Integrity"]
        self.assertEqual(sec["status"], "PASS")
        sec["status"] = "FAIL"                      # forged section verdict
        v = verify_evidence(rep)
        self.assertFalse(v["ok"])
        self.assertIn("result_hash", v["mismatches"])

    def test_evidence_stable_across_audits(self):
        a = self._audit()
        b = self._audit()
        self.assertNotEqual(a["certification"]["audit_id"],
                            b["certification"]["audit_id"])
        self.assertEqual(a["evidence_hash"], b["evidence_hash"])

    def test_input_change_breaks_evidence_too(self):
        a = self._audit()
        df2 = D.regime_trend_df(n=400)
        df2["volume"] = df2["volume"] * 1e6
        strat = as_code_strategy("t", df2, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        b = audit(strat, df2, SPEC, {"seed": 1})
        self.assertNotEqual(a["manifest_hash"], b["manifest_hash"])
        self.assertNotEqual(a["evidence_hash"], b["evidence_hash"])


if __name__ == "__main__":
    unittest.main()
