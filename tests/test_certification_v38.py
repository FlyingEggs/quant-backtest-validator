"""V3.8 — Certification contract (audit_id / hashes / continuous L0-L4 layers).

The report gains reproduction anchors (audit_id, generated_at, strategy source
fingerprint, full-frame data hash) and a certification level: the highest
CONTINUOUS layer whose every section was audited and verified clean. A missing or
unclean layer stops the climb - no skipping to a higher layer. L5-L7 are declared
product-roadmap, never faked. strategy_hash degrades to None (noted) for runs
whose source is unavailable (REPL/lambda).
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit, audit_text
from validator.report import certification_level, _strategy_hash, _data_hash
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")


def all_clean_sections(**over):
    base = {"Data Integrity": {"status": "PASS", "issues": []},
            "Look-ahead": {"status": "PASS", "issues": []},
            "MTF": {"status": "PASS", "issues": []},
            "Execution": {"status": "PASS", "issues": []},
            "Costs": {"status": "VERIFIED", "issues": []},
            "Statistics": {"status": "PASS", "issues": []},
            "Robustness": {"status": "PASS", "issues": []}}
    base.update(over)
    return base


class TestCertificationLevel(unittest.TestCase):

    def test_all_clean_reaches_l4(self):
        sect = all_clean_sections()
        rep = certification_level(sect, list(sect))
        self.assertEqual(rep["level"], "L4")
        self.assertEqual(rep["max_supported_level"], "L4")
        self.assertFalse(rep["signed"])            # L7 honest

    def test_out_of_scope_layer_stops(self):
        sect = all_clean_sections()
        scope = [s for s in sect if s != "MTF"]    # TEMPORAL incomplete
        rep = certification_level(sect, scope)
        self.assertEqual(rep["level"], "L0")
        self.assertIn("out of scope", rep["reason"])

    def test_p1_in_layer_stops_below_it(self):
        sect = all_clean_sections(
            Robustness={"status": "CONDITIONAL PASS",
                        "issues": [{"code": "X", "severity": "P1"}]})
        rep = certification_level(sect, list(sect))
        self.assertEqual(rep["level"], "L3")       # L4 STATISTICAL not clean

    def test_not_verified_section_stops(self):
        sect = all_clean_sections(MTF={"status": "NOT VERIFIED", "issues": []})
        rep = certification_level(sect, list(sect))
        self.assertEqual(rep["level"], "L0")       # L1 TEMPORAL not verified


class TestReportAnchors(unittest.TestCase):

    def _audit(self, scope):
        df = D.regime_trend_df()
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        return audit(strat, df, SPEC, {"scope": scope, "seed": 1})

    def test_audit_id_unique_and_present(self):
        a = self._audit(["Data Integrity"])
        b = self._audit(["Data Integrity"])
        self.assertNotEqual(a["certification"]["audit_id"],
                            b["certification"]["audit_id"])
        self.assertTrue(a["certification"]["audit_id"].startswith("QBV-"))
        self.assertIn("generated_at", a["certification"])

    def test_hashes_stable_for_same_inputs(self):
        a = self._audit(["Data Integrity"])
        b = self._audit(["Data Integrity"])
        self.assertEqual(a["certification"]["strategy_hash"],
                         b["certification"]["strategy_hash"])
        self.assertEqual(a["certification"]["data_hash"],
                         b["certification"]["data_hash"])

    def test_data_hash_sensitive_to_frame(self):
        df = D.regime_trend_df()
        df2 = df.copy()
        df2.loc[df2.index[0], "close"] = df2["close"].iloc[0] * 1.0001
        self.assertNotEqual(_data_hash(df), _data_hash(df2))

    def test_dynamic_run_hash_none_noted(self):
        ns = {}
        exec("def _dyn(frame, params):\n"
             "    return {'pnl': 1.0, 'trades': 1}", ns)   # no source file to read
        strat = Strategy(name="d", run=ns["_dyn"], entry_semantics="next_open")
        self.assertIsNone(_strategy_hash(strat))

    def test_text_report_renders_certified_line(self):
        txt = audit_text(*self._honest(), SPEC,
                         {"scope": ["Data Integrity"], "seed": 1})
        self.assertIn("Certified", txt)

    def _honest(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        return strat, df

    def test_scope_gap_reported_in_level(self):
        rep = self._audit(["Data Integrity", "Execution"])
        self.assertEqual(rep["certification"]["level"], "L0")
        self.assertIn("Look-ahead", rep["certification"]["reason"])


if __name__ == "__main__":
    unittest.main()
