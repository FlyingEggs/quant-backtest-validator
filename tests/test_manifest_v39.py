"""V3.9 — Audit Input Manifest & replay anchors.

Regression targets from the external audit round:
  * same OHLC, different volume/signal  -> data_hash must DIFFER (was identical)
  * same run() source, different default_params/param_grid/contract fields ->
    strategy_contract_hash must differ while strategy_source_hash stays equal
  * canonical serialisation: dict order / float spelling / nan / Timestamp must
    not destabilise hashes
  * replay: same code+data+spec+config => identical manifest; any single drift
    shows up as the corresponding fingerprint mismatch
"""

import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import DataSpec, Strategy, as_code_strategy, audit
from validator.audit import ENGINE_VERSION
from validator.manifest import (_fmt, build_manifest, verify_manifest,
                                frame_hash, schema_hash, canonical_bytes)
from examples import demo as D

SPEC = DataSpec(bar_seconds=300, source="synthetic")
CFG = {"seed": 1, "n_shuffles": 200}
SCOPE = ["Data Integrity", "Execution"]


def sstrategy(lookback=5):
    def run(frame, params):
        return {"pnl": 100.0 * float(params.get("lookback", 1.0)), "trades": 5}
    return Strategy(name="s", run=run, default_params={"lookback": lookback},
                    param_grid={"lookback": [3, 5, 8]},
                    entry_semantics="next_open")


class TestCanonicalStability(unittest.TestCase):

    def test_dict_order_irrelevant(self):
        self.assertEqual(_fmt({"a": 1, "b": 2}), _fmt({"b": 2, "a": 1}))
        self.assertEqual(_fmt({"x": [1, {"k": 2}]}), _fmt({"x": [1, {"k": 2}]}))

    def test_float_spelling_irrelevant(self):
        self.assertEqual(_fmt(100.0), _fmt(1e2))
        self.assertEqual(_fmt(0.1), _fmt(0.10))

    def test_nan_inf_stable(self):
        a = _fmt(float("nan"))
        b = _fmt(float("inf"))
        c = _fmt(float("-inf"))
        self.assertEqual(a, _fmt(float("nan")))
        self.assertNotEqual(b, c)

    def test_pandas_timestamp_stable(self):
        t = pd.Timestamp("2026-09-05 10:30:00")
        self.assertEqual(_fmt(t), _fmt(pd.Timestamp("2026-09-05 10:30:00")))
        self.assertEqual(_fmt(np.datetime64("2026-09-05 10:30:00")),
                         _fmt(pd.Timestamp("2026-09-05 10:30:00")))

    def test_numpy_scalars_match_python(self):
        self.assertEqual(_fmt(np.float64(1.5)), _fmt(1.5))
        self.assertEqual(_fmt(np.int64(7)), _fmt(7))

    def test_frame_hash_nan_deterministic(self):
        df = D.regime_trend_df(n=200)
        df.loc[df.index[3], "close"] = np.nan
        self.assertEqual(frame_hash(df), frame_hash(df.copy()))


class TestHashCoverage(unittest.TestCase):
    """The two external-audit claims, now regressions."""

    def test_data_hash_covers_volume_and_signal(self):
        a = D.regime_trend_df(n=800)
        b = a.copy()
        b["volume"] = b["volume"] * 1e6
        b["sig"] = -b["sig"]
        self.assertNotEqual(frame_hash(a), frame_hash(b))

    def test_data_hash_covers_single_price_tick(self):
        a = D.regime_trend_df(n=800)
        b = a.copy()
        b.loc[b.index[0], "close"] = b["close"].iloc[0] * 1.0001
        self.assertNotEqual(frame_hash(a), frame_hash(b))

    def test_contract_hash_covers_defaults_and_grid(self):
        m5 = build_manifest(sstrategy(5), D.regime_trend_df(n=300), SPEC, CFG,
                            ENGINE_VERSION, SCOPE)
        m50 = build_manifest(sstrategy(50), D.regime_trend_df(n=300), SPEC, CFG,
                             ENGINE_VERSION, SCOPE)
        self.assertNotEqual(m5["strategy_contract_hash"],
                            m50["strategy_contract_hash"])
        self.assertEqual(m5["strategy_source_hash"], m50["strategy_source_hash"])

    def test_contract_hash_covers_description(self):
        def run(f, p):
            return {"pnl": 1.0, "trades": 1}
        a = Strategy(name="s", run=run, description="v1", entry_semantics="next_open")
        b = Strategy(name="s", run=run, description="v2", entry_semantics="next_open")
        self.assertNotEqual(build_manifest(a, D.regime_trend_df(n=200), SPEC,
                                           CFG, ENGINE_VERSION, SCOPE)
                            ["strategy_contract_hash"],
                            build_manifest(b, D.regime_trend_df(n=200), SPEC,
                                           CFG, ENGINE_VERSION, SCOPE)
                            ["strategy_contract_hash"])

    def test_schema_hash_distinguishes_columns(self):
        a = D.regime_trend_df(n=200)
        b = a.copy()
        b = b.rename(columns={"volume": "not_volume"})
        self.assertNotEqual(schema_hash(a), schema_hash(b))

    def test_dataspec_hash_covers_instrument_and_frames(self):
        base = DataSpec(bar_seconds=300)
        with_qs = DataSpec(bar_seconds=300, qty_step=0.1)
        self.assertNotEqual(build_manifest(sstrategy(), D.regime_trend_df(n=200),
                                           base, CFG, ENGINE_VERSION, SCOPE)
                            ["dataspec_hash"],
                            build_manifest(sstrategy(), D.regime_trend_df(n=200),
                                           with_qs, CFG, ENGINE_VERSION, SCOPE)
                            ["dataspec_hash"])
        # MTF frame content drift must move dataspec_hash
        h1 = D.regime_trend_df(n=100)
        spec1 = DataSpec(bar_seconds=300, timeframes={"h1": h1})
        h2 = h1.copy()
        h2["close"] = h2["close"] * 1.0001
        spec2 = DataSpec(bar_seconds=300, timeframes={"h1": h2})
        self.assertNotEqual(build_manifest(sstrategy(), D.regime_trend_df(n=200),
                                           spec1, CFG, ENGINE_VERSION, SCOPE)
                            ["dataspec_hash"],
                            build_manifest(sstrategy(), D.regime_trend_df(n=200),
                                           spec2, CFG, ENGINE_VERSION, SCOPE)
                            ["dataspec_hash"])

    def test_config_hash_covers_seed_and_cost(self):
        m1 = build_manifest(sstrategy(), D.regime_trend_df(n=200), SPEC,
                            {"seed": 1}, ENGINE_VERSION, SCOPE)
        m2 = build_manifest(sstrategy(), D.regime_trend_df(n=200), SPEC,
                            {"seed": 2}, ENGINE_VERSION, SCOPE)
        m3 = build_manifest(sstrategy(), D.regime_trend_df(n=200), SPEC,
                            {"seed": 1, "cost": {"commission": {"mode": "bps",
                                                                "open_rate": 5.0}}},
                            ENGINE_VERSION, SCOPE)
        self.assertNotEqual(m1["config_hash"], m2["config_hash"])
        self.assertNotEqual(m1["config_hash"], m3["config_hash"])


class TestReplayVerification(unittest.TestCase):

    def _base(self):
        return build_manifest(sstrategy(), D.regime_trend_df(n=300), SPEC, CFG,
                              ENGINE_VERSION, SCOPE)

    def test_replay_ok_same_inputs(self):
        m = self._base()
        v = verify_manifest(m, sstrategy(), D.regime_trend_df(n=300), SPEC, CFG,
                            ENGINE_VERSION, SCOPE)
        self.assertTrue(v["ok"])
        self.assertEqual(v["mismatches"], [])

    def test_volume_drift_reported(self):
        m = self._base()
        df2 = D.regime_trend_df(n=300)
        df2["volume"] = df2["volume"] * 1e6
        v = verify_manifest(m, sstrategy(), df2, SPEC, CFG, ENGINE_VERSION, SCOPE)
        self.assertIn("data_hash", v["mismatches"])

    def test_params_drift_reported(self):
        m = self._base()
        v = verify_manifest(m, sstrategy(lookback=50), D.regime_trend_df(n=300),
                            SPEC, CFG, ENGINE_VERSION, SCOPE)
        self.assertIn("strategy_contract_hash", v["mismatches"])

    def test_seed_drift_reported(self):
        m = self._base()
        v = verify_manifest(m, sstrategy(), D.regime_trend_df(n=300), SPEC,
                            {"seed": 99}, ENGINE_VERSION, SCOPE)
        self.assertIn("random_seed", v["mismatches"])
        self.assertIn("config_hash", v["mismatches"])


class TestAuditIntegration(unittest.TestCase):

    def test_report_carries_manifest(self):
        df = D.regime_trend_df(n=400)
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 1})
        self.assertIn("manifest", rep)
        self.assertIn("manifest_hash", rep)
        self.assertEqual(rep["manifest_hash"], rep["manifest"]["manifest_hash"])
        json.dumps(rep)                       # json-safe

    def test_manifest_hash_replayable_across_audits(self):
        df = D.regime_trend_df(n=400)
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        a = audit(strat, df, SPEC, {"seed": 1})
        b = audit(strat, df, SPEC, {"seed": 1})
        self.assertNotEqual(a["certification"]["audit_id"],
                            b["certification"]["audit_id"])
        self.assertEqual(a["manifest_hash"], b["manifest_hash"])

    def test_strategy_and_data_hash_fields_still_present(self):
        df = D.regime_trend_df(n=400)
        strat = as_code_strategy("t", df, "sig", D.next_open_hold(5),
                                 entry_semantics="next_open")
        rep = audit(strat, df, SPEC, {"seed": 1})
        rep2 = audit(strat, df, SPEC, {"seed": 1})
        cert = rep["certification"]
        # certification.strategy_hash = identity blob (name+desc+source), stable
        self.assertEqual(cert["strategy_hash"], rep2["certification"]["strategy_hash"])
        self.assertEqual(len(cert["strategy_hash"]), 64)
        # data_hash now uses the manifest full-frame fingerprint
        self.assertEqual(cert["data_hash"], rep["manifest"]["data_hash"])


if __name__ == "__main__":
    unittest.main()
