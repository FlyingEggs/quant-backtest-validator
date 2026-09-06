"""V3.9 — Audit Input Manifest & replay anchors.

The old certification anchors hashed only strategy source + OHLC. A reproducible
audit evidence chain must fingerprint EVERYTHING that can change the result:

    engine_version
    scope
    random_seed
    strategy_source_hash        run() source (source-unavailable -> token)
    strategy_contract_hash      name/description/entry_semantics/supports_from_bar/
                                accepts_frozen/signal_col/default_params/param_grid/
                                fit_is & bt_mechanism sources
    data_schema_hash            column names + dtypes + index type
    data_hash                   FULL-frame fingerprint (every column, not just OHLC)
    dataspec_hash               DataSpec fields + timeframes (each frame hashed)
    config_hash                 the audit config (cost/mtf/surface/seed/...)
    cost_hash                   canonical of config['cost'] (readable breakdown)
    manifest_hash               sha256 over the canonicalised manifest

Canonical serialisation makes hashes stable across dict ordering, float
spellings, numpy scalars and pandas timestamps. `verify_manifest` recomputes the
fingerprints from (strategy, df, spec, cfg, engine_version) and reports which
fields drifted - the replay check.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from validator.types import DataSpec, Strategy


# ---------------------------------------------------------------------------
# canonical serialisation (order-independent, type-explicit, lossless-ish)
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    """Deterministic token for any input. float uses float.hex() (exact, stable,
    nan/inf spelled out); dicts sort keys; pandas timestamps go ISO."""
    if v is None:
        return "n"
    if v is True or v is False:
        return "b1" if v else "b0"
    if isinstance(v, int):
        return f"i{v}"
    if isinstance(v, float):
        return f"f{float.hex(v)}"
    if isinstance(v, str):
        return f"s{len(v)}:{v}"
    if isinstance(v, bytes):
        return f"x{len(v)}:{v.hex()}"
    if isinstance(v, (datetime, date)):
        return f"d{v.isoformat()}"
    if isinstance(v, np.datetime64):
        return f"d{pd.Timestamp(v).isoformat()}"
    if isinstance(v, pd.Timestamp):
        return f"d{v.isoformat()}"
    if isinstance(v, np.ndarray):
        return "a[" + ",".join(str(x) for x in v.shape) + "]{" + \
               "|".join(_fmt(x) for x in v.reshape(-1)) + "}"
    if isinstance(v, np.generic):
        return _fmt(v.item())
    if isinstance(v, dict):
        body = "|".join(f"{_fmt(str(k))}={_fmt(val)}"
                        for k, val in sorted(v.items(), key=lambda kv: str(kv[0])))
        return "d{" + body + "}"
    if isinstance(v, (list, tuple)):
        return "l[" + "|".join(_fmt(x) for x in v) + "]"
    if callable(v):
        return "c" + _fmt(_callable_token(v))
    # unknown objects: repr() is NOT guaranteed stable (default __repr__ embeds
    # memory addresses) - keep the type name + repr, but note in the docs that an
    # object column whose repr is address-based makes its frame_hash unstable by
    # design (replay then fails honestly instead of silently passing).
    return f"r{type(v).__module__}.{type(v).__name__}:{v!r}"


def _callable_token(fn: Callable, _seen: Optional[set] = None) -> Dict:
    """Stable fingerprint of a callable's ENVIRONMENT, not just its source:

      module + qualname + source-hash + __closure__ cell values (canonicalised).

    Source alone cannot detect closure/global drift (same def, different captured
    coeff). Closure cells ARE captured here (deterministic, no pickle). Module-level
    globals a function reads are NOT visible on the function object - that drift is a
    documented boundary: keep mutable dependencies in params/closures, or treat a
    module-global change as a code change. Recursion is guarded by an id set.
    """
    if _seen is None:
        _seen = set()
    fid = id(fn)
    if fid in _seen:
        return {"recursive": True,
                "qualname": getattr(fn, "__qualname__", type(fn).__name__)}
    _seen = _seen | {fid}
    try:
        src = inspect.getsource(fn)
        source = hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        source = "UNAVAILABLE"
    closure: List[Dict] = []
    for cell in (getattr(fn, "__closure__", None) or ()):
        val = cell.cell_contents
        if callable(val):
            closure.append(_callable_token(val, _seen))
        else:
            closure.append({"value": _fmt(val)})
    return {"module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", type(fn).__name__),
            "source": source,
            "closure": closure}


def _environment_fingerprint() -> Dict:
    """Runtime/package versions. Platform excludes hostname so replay survives
    moving between machines with the same OS/kernel class."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import scipy  # type: ignore[import-untyped]  # noqa: PLC0415 - optional
        env["scipy"] = scipy.__version__
    except Exception:
        env["scipy"] = None
    return env


def canonical_bytes(obj: Any) -> bytes:
    return _fmt(obj).encode("utf-8")


def _h(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


# ---------------------------------------------------------------------------
# full-frame hash (every column: OHLC, volume, signal, auxiliaries)
# ---------------------------------------------------------------------------

def frame_hash(df: pd.DataFrame) -> str:
    """Self-contained full-frame identity: column names, per-column DTYPE, index
    dtype/timezone and every value - a single hash that already distinguishes
    int64 vs float64, float32 vs float64 and tz-aware vs naive indexes."""
    parts = []
    idx = df.index
    parts.append(f"idxdt:{idx.dtype}".encode())     # datetime64[ns, UTC] vs [ns] differ
    if isinstance(idx, pd.DatetimeIndex):
        parts.append(np.ascontiguousarray(idx.asi8).tobytes())  # type: ignore[attr-defined]
    else:
        parts.append(_fmt(list(idx)).encode())
    for c in df.columns:
        s = df[c]
        parts.append(_fmt(str(c)).encode())
        parts.append(f"dt:{s.dtype}".encode())
        if pd.api.types.is_datetime64_any_dtype(s):
            parts.append(np.ascontiguousarray(
                s.to_numpy().astype("datetime64[ns]").astype(np.int64)).tobytes())
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            parts.append(np.ascontiguousarray(s.to_numpy()).tobytes())
        else:
            # str / category / object: canonical tokens; custom objects whose
            # repr embeds addresses make the hash unstable BY DESIGN (honest
            # replay failure -> normalise such columns to str/numeric first)
            parts.append(_fmt(list(s)).encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def schema_hash(df: pd.DataFrame) -> str:
    cols = [(str(c), str(df[c].dtype)) for c in df.columns]
    idx = type(df.index).__name__
    return _h({"columns": cols, "index": idx, "n": int(len(df))})


# ---------------------------------------------------------------------------
# per-input fingerprints
# ---------------------------------------------------------------------------

def strategy_source_hash(strategy: Strategy) -> str:
    # full environment token of run(): module + qualname + source + closure values
    return _h(_callable_token(strategy.run))


def strategy_contract_hash(strategy: Strategy) -> str:
    fit_src = _callable_token(strategy.fit_is) if strategy.fit_is else None
    bt_src = _callable_token(strategy.bt_mechanism) if strategy.bt_mechanism else None
    return _h({
        "name": strategy.name,
        "description": strategy.description,
        "entry_semantics": strategy.entry_semantics,
        "supports_from_bar": strategy.supports_from_bar,
        "accepts_frozen": strategy.accepts_frozen,
        "signal_col": strategy.signal_col,
        "default_params": strategy.default_params or {},
        "param_grid": strategy.param_grid or {},
        "fit_is_env": fit_src,
        "bt_mechanism_env": bt_src,
    })


def dataspec_hash(spec: DataSpec) -> str:
    if spec is None:
        return _h(None)
    frames = {}
    for name, fr in (spec.timeframes or {}).items():
        frames[name] = {"schema": schema_hash(fr), "data": frame_hash(fr)}
    return _h({
        "bar_seconds": spec.bar_seconds,
        "source": spec.source,
        "description": spec.description,
        "bar_timestamp_semantics": spec.bar_timestamp_semantics,
        "qty_step": spec.qty_step,
        "min_qty": spec.min_qty,
        "min_notional": spec.min_notional,
        "contract_size": spec.contract_size,
        "timeframes": frames,
    })


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def build_manifest(strategy: Strategy, df: pd.DataFrame, spec: DataSpec,
                   cfg: Dict, engine_version: str, scope: List[str]) -> Dict:
    manifest = {
        "engine_version": engine_version,
        "scope": list(scope),
        "random_seed": cfg.get("seed"),
        "environment": _environment_fingerprint(),
        "strategy_source_hash": strategy_source_hash(strategy),
        "strategy_contract_hash": strategy_contract_hash(strategy),
        "data_schema_hash": schema_hash(df),
        "data_hash": frame_hash(df),
        "dataspec_hash": dataspec_hash(spec),
        "config_hash": _h(_fmt(cfg)),          # full canonical cfg (callables by env)
        "cost_hash": _h(_fmt(cfg.get("cost"))),
    }
    manifest["manifest_hash"] = _h(manifest)
    return manifest


# ---------------------------------------------------------------------------
# result / evidence hashes (anti-tamper on the OUTPUT side)
# ---------------------------------------------------------------------------

def _result_dict(report: Dict) -> Dict:
    """Canonical projection of the audit RESULT: verdicts, findings, key metrics
    and certification level. Everything a report consumer actually reads."""
    sections = report.get("sections") or {}
    stat = sections.get("Statistics") or {}
    stat_ev = stat.get("evidence") or {}
    costs = sections.get("Costs") or {}
    net = (costs.get("evidence") or {}).get("net") or {}
    return {
        "overall": report.get("overall"),
        "verified_score": report.get("verified_score"),
        "coverage_pct": report.get("coverage_pct"),
        "blocking": report.get("blocking"),
        "certification_level": (report.get("certification") or {}).get("level"),
        "section_statuses": {k: (v.get("status") if isinstance(v, dict) else v)
                             for k, v in sorted(sections.items())},
        "findings": sorted((i.get("severity"), i.get("code"))
                           for i in (report.get("issues") or [])),
        "n_eff": stat_ev.get("n_eff"),
        "n_trades": stat_ev.get("n"),
        "net_pnl": net.get("net_pnl"),
        "gross_pnl": net.get("gross_pnl"),
    }


def result_hash(report: Dict) -> str:
    return _h(_result_dict(report))


def evidence_hash_from_report(report: Dict) -> Dict:
    """result_hash (what the audit concluded) chained to manifest_hash (what it
    ran on). Detects a tampered verdict even when inputs were not touched."""
    mh = report.get("manifest_hash")
    rh = result_hash(report)
    if not mh:
        return {"result_hash": rh, "evidence_hash": rh}
    return {"result_hash": rh,
            "evidence_hash": _h({"manifest_hash": mh, "result_hash": rh})}


def verify_evidence(report: Dict) -> Dict:
    recomputed = evidence_hash_from_report(report)
    mismatches = []
    for key in ("result_hash", "evidence_hash"):
        stored = report.get(key)
        if stored is not None and stored != recomputed[key]:
            mismatches.append(key)
    return {"ok": not mismatches, "mismatches": mismatches,
            "recomputed": recomputed}


def verify_manifest(manifest: Dict, strategy: Strategy, df: pd.DataFrame,
                    spec: DataSpec, cfg: Dict, engine_version: str,
                    scope: List[str]) -> Dict:
    """Replay check: recompute every fingerprint and report drifts."""
    recomputed = build_manifest(strategy, df, spec, cfg, engine_version, scope)
    mismatches = []
    for key, stored in manifest.items():
        if key == "manifest_hash":
            continue
        if stored != recomputed.get(key):
            mismatches.append(key)
    stored_mh = manifest.get("manifest_hash")
    recomputed_mh = recomputed["manifest_hash"]
    if stored_mh is not None and stored_mh != recomputed_mh:
        mismatches.append("manifest_hash")
    return {"ok": not mismatches, "mismatches": mismatches,
            "stored": manifest, "recomputed": recomputed}
