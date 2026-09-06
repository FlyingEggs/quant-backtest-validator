"""Post-Epic edge sweep (V3.6-V3.8) — code-auditor round on the newly added
contracts: instrument executability edges, provenance probe boundaries, and
certification layer/anchor edges. Each case targets a path the Epic tests did
not pin down.
"""

import json
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit, audit_text
from validator import costengine
from validator.report import certification_level
from validator.wf import parameter_freeze_audit
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")
DF = D.regime_trend_df(n=1200)


def tr(qty=1.0, ep=100.0, xp=110.0, volume=None, ts=True):
    t = {"side": "long", "qty": qty, "entry_price": ep, "exit_price": xp}
    if ts:
        t.update({"entry_ts": pd.Timestamp("2026-01-01"),
                  "exit_ts": pd.Timestamp("2026-01-02")})
    if volume is not None:
        t["volume"] = volume
    return t


def inst(qs=0.0, mq=0.0, mn=0.0, cs=1.0):
    out = {}
    if qs:
        out["qty_step"] = qs
    if mq:
        out["min_qty"] = mq
    if mn:
        out["min_notional"] = mn
    if cs != 1.0:
        out["contract_size"] = cs
    return out


class TestInstrumentEdges(unittest.TestCase):

    def test_multi_violation_one_issue_per_rule(self):
        cfg = {"instrument": inst(qs=0.1, mq=1.0, mn=500.0)}
        # qty 0.5: not a 0.1-lot multiple? 0.5 IS 5 lots; violates min_qty and
        # notional only. Pick qty=1.237: breaks step, >= min_qty, notional<500.
        rep = costengine.net_audit([tr(qty=1.237)], cfg)
        codes = [i["code"] for i in rep["issues"]]
        self.assertIn("EXEC_QTY_STEP", codes)
        self.assertIn("EXEC_MIN_NOTIONAL", codes)
        self.assertNotIn("EXEC_MIN_QTY", codes)      # 1.237 >= 1.0
        self.assertEqual(len(codes), 2, "exactly one issue per violated rule")

    def test_boundary_equality_not_flagged(self):
        # qty == min_qty (not <), notional == min_notional (not <), qty on-lot
        cfg = {"instrument": inst(qs=1.0, mq=1.0, mn=100.0)}
        rep = costengine.net_audit([tr(qty=1.0, ep=100.0)], cfg)
        self.assertEqual([i["code"] for i in rep["issues"]], [])
        self.assertEqual(rep["sub_models"]["execution"], "PASS")

    def test_negative_qty_not_waved_through(self):
        # net engine abs()es qty, but |−1.237| still breaks the 0.1 lot
        cfg = {"instrument": inst(qs=0.1)}
        rep = costengine.net_audit([tr(qty=-1.237)], cfg)
        self.assertIn("EXEC_QTY_STEP", [i["code"] for i in rep["issues"]])

    def test_contracts_field_alias_checked(self):
        cfg = {"instrument": inst(qs=0.5)}
        t = tr(qty=None)
        del t["qty"]
        t["contracts"] = 1.237                       # alias path
        rep = costengine.net_audit([t], cfg)
        self.assertIn("EXEC_QTY_STEP", [i["code"] for i in rep["issues"]])

    def test_volume_mixed_presence_partial_drag(self):
        cfg = {"market_impact": {"mode": "volume_linear", "coeff": 0.1}}
        log = [tr(qty=1.0, volume=100.0), tr(qty=1.0)]   # second has no volume
        rep = costengine.net_audit(log, cfg)
        self.assertEqual(rep["sub_models"]["market_impact"], "NOT VERIFIED")
        self.assertIn("market_impact", rep["declared_missing"])
        drag = dict((r[0], r[1]) for r in rep["table"])["Market Impact"]
        self.assertLess(drag, 0.0)                   # first fill's drag still counted

    def test_contract_size_meets_notional_floor(self):
        # qty*cs*price: with cs=10 notional 1000 >= floor 1000 -> clean;
        # without cs the same qty would violate -> the cs scaling is respected.
        cfg10 = {"instrument": inst(mn=1000.0, cs=10.0)}
        rep = costengine.net_audit([tr(qty=1.0, ep=100.0)], cfg10)
        self.assertNotIn("EXEC_MIN_NOTIONAL", [i["code"] for i in rep["issues"]])
        cfg1 = {"instrument": inst(mn=1000.0)}
        rep1 = costengine.net_audit([tr(qty=1.0, ep=100.0)], cfg1)
        self.assertIn("EXEC_MIN_NOTIONAL", [i["code"] for i in rep1["issues"]])


class TestProvenanceEdges(unittest.TestCase):

    def _honest(self, supports_from_bar=False):
        def run(frame, params):
            return {"pnl": 100.0 * float(params["k"]), "trades": 10}
        return Strategy(name="h", run=run, default_params={"k": 1.0},
                        entry_semantics="next_open",
                        supports_from_bar=supports_from_bar,
                        fit_is=lambda df: {"k": 2.0},
                        accepts_frozen=True)

    def test_supports_from_bar_path_passes(self):
        rep = parameter_freeze_audit(self._honest(supports_from_bar=True), DF, {})
        self.assertEqual(rep["provenance"], "PASS")   # _from_bar injection path

    def test_fit_is_raising_is_reported_not_crash(self):
        def bad_fit(df):
            raise RuntimeError("fit exploded")
        strat = Strategy(name="b", run=lambda f, p: {"pnl": 1.0, "trades": 10},
                         entry_semantics="next_open",
                         fit_is=bad_fit, accepts_frozen=True)
        rep = parameter_freeze_audit(strat, DF, {})
        self.assertEqual(rep["provenance"], "NOT VERIFIED")
        self.assertIn("PROVENANCE_FIT_ERROR", [i["code"] for i in rep["issues"]])

    def test_zero_trades_not_verified_not_pass(self):
        # identical zero-trade output under both injections proves nothing about
        # whether the strategy honours the frozen params - PASS would be fake.
        def run(frame, params):
            return {"pnl": 0.0, "trades": 0}
        strat = Strategy(name="z", run=run, default_params={"k": 1.0},
                         entry_semantics="next_open",
                         fit_is=lambda df: {"k": 2.0}, accepts_frozen=True)
        rep = parameter_freeze_audit(strat, DF, {})
        self.assertEqual(rep["provenance"], "NOT VERIFIED")
        self.assertNotIn("PARAM_PROVENANCE", [i["code"] for i in rep["issues"]])


def _clean_sections(**over):
    base = {"Data Integrity": {"status": "PASS", "issues": []},
            "Look-ahead": {"status": "PASS", "issues": []},
            "MTF": {"status": "PASS", "issues": []},
            "Execution": {"status": "PASS", "issues": []},
            "Costs": {"status": "VERIFIED", "issues": []},
            "Statistics": {"status": "PASS", "issues": []},
            "Robustness": {"status": "PASS", "issues": []}}
    base.update(over)
    return base


class TestCertificationEdges(unittest.TestCase):

    def test_failed_section_stops_layer(self):
        sect = _clean_sections(Execution={"status": "FAIL",
                                          "issues": [{"code": "X",
                                                      "severity": "P0"}]})
        rep = certification_level(sect, list(sect))
        self.assertEqual(rep["level"], "L1")          # L2 EXECUTION unclean

    def test_no_l0_in_scope_level_none(self):
        sect = _clean_sections()
        rep = certification_level(sect, ["Execution", "Costs"])
        self.assertEqual(rep["level"], "NONE")
        self.assertIn("L0", rep["reason"])

    def test_report_json_serializable_with_certification(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"scope": ["Data Integrity"], "seed": 1})
        blob = json.dumps(rep)                        # certification block must not break it
        self.assertIn("QBV-", blob)
        self.assertIn("certification", blob)

    def test_text_certified_no_branch(self):
        df = D.regime_trend_df()
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        txt = audit_text(strat, df, SPEC, {"scope": ["Costs"], "seed": 1,
                                           "cost": {"commission": {"mode": "bps",
                                                                   "open_rate": 4.0,
                                                                   "close_rate": 4.0}}})
        self.assertIn("Certified       : NO", txt)    # L0 out of scope

    def test_strategy_hash_changes_with_description(self):
        def run(frame, params):
            return {"pnl": 1.0, "trades": 1}
        a = Strategy(name="s", run=run, description="v1", entry_semantics="next_open")
        b = Strategy(name="s", run=run, description="v2", entry_semantics="next_open")
        from validator.report import _strategy_hash
        self.assertNotEqual(_strategy_hash(a), _strategy_hash(b))


if __name__ == "__main__":
    unittest.main()
