"""V2 audit types & config — the client-facing contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import pandas as pd

# Strategy run: df (never mutated) + params -> metrics dict.
# Must return {"pnl": float, "trades": int} and may add "rets": array-like.
RunFn = Callable[[pd.DataFrame, Dict], Dict]
# Mechanism bt: operates on a frame that CONTAINS the exposed signal column
# (v1-style pure function used by lag / expansion / randomized control).
BtFn = Callable[[pd.DataFrame], Dict]


@dataclass
class Strategy:
    name: str
    run: RunFn                                   # run(df, params) -> metrics
    default_params: Dict = field(default_factory=dict)
    param_grid: Optional[Dict[str, list]] = None  # {"lookback": [3,5,8,13]}
    entry_semantics: str = "next_open"
    description: str = ""
    # if True, run() honours the reserved params '_from_bar' (and optional '_to_bar'):
    # entries before '_from_bar' are ignored -> enables warm-up-context OOS runs
    supports_from_bar: bool = False
    # ---- optional: expose the signal for the full mechanism suite -------------
    # signal_col: name of the signal column the strategy consumes from df
    # bt_mechanism: pure bt(df) over a frame that carries signal_col (v1-style)
    signal_col: Optional[str] = None
    bt_mechanism: Optional[BtFn] = None


@dataclass
class DataSpec:
    bar_seconds: int = 300
    source: str = "client-provided"
    description: str = ""


def as_strategy(name: str, run_df: Callable[[pd.DataFrame], Dict],
                entry_semantics: str = "next_open",
                description: str = "", supports_from_bar: bool = False) -> Strategy:
    """Adapt a plain run(df)->metrics function into a black-box Strategy.

    Parameter sensitivity is only meaningful when run(df, params) and a param_grid
    are supplied; a plain run(df) strategy reports robustness accordingly.
    """
    def _run(df: pd.DataFrame, params: dict) -> Dict:
        return run_df(df)
    return Strategy(name=name, run=_run, entry_semantics=entry_semantics,
                    description=description, supports_from_bar=supports_from_bar)


def as_code_strategy(name: str, df: pd.DataFrame, signal_col: str,
                     bt: BtFn, run_df=None,
                     entry_semantics: str = "next_open",
                     description: str = "") -> Strategy:
    """Strategy from code-level artefacts: a signal column + its backtest fn.

    Enables the full mechanism suite (lag, expansion, randomized control) in
    addition to the generic checks.
    """
    if run_df is None:
        def _run(frame: pd.DataFrame, params: dict) -> Dict:
            return bt(frame)
        run_fn = _run
    else:
        run_fn = run_df
    return Strategy(name=name, run=run_fn, entry_semantics=entry_semantics,
                    description=description, signal_col=signal_col,
                    bt_mechanism=bt)


def default_config(**overrides) -> Dict:
    cfg = {
        "seed": 42,
        "n_shuffles": 200,
        "expansion_confirmation": None,   # None | "shifted" | "completed"
        "oos_frac": 0.30,
        "cost": None,                     # None -> Costs section NOT VERIFIED
        "run_robustness": True,
    }
    cfg.update(overrides)
    return cfg


def run_metrics(strategy: Strategy, df: pd.DataFrame,
                params: Optional[dict] = None) -> Dict:
    """Run a strategy safely; guarantee required keys exist."""
    res = dict(strategy.run(df, params if params is not None else strategy.default_params))
    res.setdefault("pnl", res.get("total_pnl", 0.0))
    res.setdefault("trades", 0)
    return res
