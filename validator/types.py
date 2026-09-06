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
    # ---- V3.7 parameter-provenance contract ----------------------------------
    # Declare BOTH to get OOS parameter provenance verified:
    #   fit_is(df)      -> params learned on the IS window ONLY (pure, stateless)
    #   accepts_frozen  -> run(df, params) treats reserved '_frozen' params as
    #                      authoritative and NEVER re-fits internally
    fit_is: Optional[Callable[[pd.DataFrame], Dict]] = None
    accepts_frozen: bool = False


@dataclass
class DataSpec:
    bar_seconds: int = 300
    source: str = "client-provided"
    description: str = ""
    # V3: higher-timeframe frames for MTF temporal-availability checks.
    # name -> DataFrame with a DatetimeIndex (bar OPEN) and the value column used.
    timeframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Data semantics contract: MTF temporal availability and the execution
    # timeline model the frame index as the bar OPEN time. Frames indexed by bar
    # CLOSE must declare so (data_integrity reports DATA_TS_SEMANTICS instead of
    # blessing a CLOSE-indexed frame with OPEN-semantics checks).
    bar_timestamp_semantics: str = "OPEN"   # "OPEN" | "CLOSE"
    # ---- V3.6 instrument / execution-realism contract -------------------------
    # 0.0 / default = NOT declared -> the corresponding realism sub-check is
    # reported NOT VERIFIED, never assumed clean.
    qty_step: float = 0.0        # qty must be expressible as a multiple of qty_step
    min_qty: float = 0.0         # fills below min_qty cannot execute (ghost fills)
    min_notional: float = 0.0    # notional = qty * contract_size * price floor
    contract_size: float = 1.0   # futures: qty is in contracts; notional = qty*size*price


def as_strategy(name: str, run_df: Callable[[pd.DataFrame], Dict],
                entry_semantics: str = "next_open",
                description: str = "", supports_from_bar: bool = False,
                fit_is=None, accepts_frozen: bool = False) -> Strategy:
    """Adapt a plain run(df)->metrics function into a black-box Strategy.

    Parameter sensitivity is only meaningful when run(df, params) and a param_grid
    are supplied; a plain run(df) strategy reports robustness accordingly.
    """
    def _run(df: pd.DataFrame, params: dict) -> Dict:
        return run_df(df)
    return Strategy(name=name, run=_run, entry_semantics=entry_semantics,
                    description=description, supports_from_bar=supports_from_bar,
                    fit_is=fit_is, accepts_frozen=accepts_frozen)


def as_code_strategy(name: str, df: pd.DataFrame, signal_col: str,
                     bt: BtFn, run_df=None,
                     entry_semantics: str = "next_open",
                     description: str = "",
                     fit_is=None, accepts_frozen: bool = False) -> Strategy:
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
                    bt_mechanism=bt, fit_is=fit_is,
                    accepts_frozen=accepts_frozen)


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
