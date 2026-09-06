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
        return "c" + _callable_token(v)
    return f"r{type(v).__name__}:{v!r}"


def _callable_token(fn: Callable) -> str:
    """Stable fingerprint of a callable's SOURCE. Unavailable (exec/REPL/lambda
    without file) degrades to an honest qualname token - never the memory address."""
    try:
        src = inspect.getsource(fn)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        qn = getattr(fn, "__qualname__", type(fn).__name__)
        return f"UNAVAILABLE:{qn}"


def canonical_bytes(obj: Any) -> bytes:
    return _fmt(obj).encode("utf-8")


def _h(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


# ---------------------------------------------------------------------------
# full-frame hash (every column: OHLC, volume, signal, auxiliaries)
# ---------------------------------------------------------------------------

def frame_hash(df: pd.DataFrame) -> str:
    parts = []
    if isinstance(df.index, pd.DatetimeIndex):
        parts.append(b"idx=dt64")
        parts.append(np.ascontiguousarray(df.index.asi8).tobytes())  # type: ignore[attr-defined]
    else:
        parts.append(b"idx=" + _fmt(list(df.index)).encode())
    for c in df.columns:
        s = df[c]
        parts.append(_fmt(str(c)).encode())
        if pd.api.types.is_numeric_dtype(s):
            parts.append(np.ascontiguousarray(s.to_numpy(dtype="float64")).tobytes())
        else:
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
    return _callable_token(strategy.run)


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
        "fit_is_source": fit_src,
        "bt_mechanism_source": bt_src,
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
        "strategy_source_hash": strategy_source_hash(strategy),
        "strategy_contract_hash": strategy_contract_hash(strategy),
        "data_schema_hash": schema_hash(df),
        "data_hash": frame_hash(df),
        "dataspec_hash": dataspec_hash(spec),
        "config_hash": _h(_fmt(cfg)),          # full canonical cfg (callables by source)
        "cost_hash": _h(_fmt(cfg.get("cost"))),
    }
    manifest["manifest_hash"] = _h(manifest)
    return manifest


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
