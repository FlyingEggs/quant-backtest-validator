"""examples/demo.py — run the validator on five reproducible archetypes.

Deterministic (fixed seeds). Prints the tables used in the README and case study.
Run from the repository root:

    python3 examples/demo.py

Python >= 3.9. numpy/pandas only.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import (full_audit, lag_sensitivity, period_expansion,
                       fill_timing_sensitivity, randomized_control,
                       return_independence)


# ---------------------------------------------------------------------------
# shared strategy: enter at NEXT bar open when signal says +1, hold `hold` bars
# ---------------------------------------------------------------------------

def next_open_hold(hold: int):
    """Pure strategy fn factory: signal at bar close -> fill at next open, exit after
    `hold` bars at the open (exit bar's open). Returns {pnl, trades, rets}."""
    def run(df: pd.DataFrame) -> dict:
        sig = df["sig"].fillna(0).astype(float).values
        o = df["open"].values
        n = len(df)
        pnl, trades, pos, entry_i, rets = 0.0, 0, 0.0, -1, []
        for i in range(n):
            if pos != 0.0 and i >= entry_i + hold:
                px = o[min(i, n - 1)]
                pnl += (px - pos) * 1.0
                rets.append((px - pos) / pos if pos else 0.0)
                pos = 0.0
            if pos == 0.0 and sig[i] == 1.0 and i + 1 < n:
                pos = o[i + 1]
                entry_i = i
                trades += 1
        return {"pnl": pnl, "trades": trades, "rets": np.asarray(rets, dtype=float)}
    return run


def frame(closes: np.ndarray) -> pd.DataFrame:
    n = len(closes)
    open_ = np.empty(n)
    open_[0] = closes[0]
    open_[1:] = closes[:-1]                      # no overnight gaps (5-min bars)
    high = np.maximum(open_, closes) * 1.0005
    low = np.minimum(open_, closes) * 0.9995
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": closes})


# ---------------------------------------------------------------------------
# archetype data builders (deterministic)
# ---------------------------------------------------------------------------

def same_bar_leak_df(n: int = 2000) -> pd.DataFrame:
    """Step signal (+1 then -1) with deterministic intraday moves.

    Profit is entirely intraday: an impossible same-bar fill "knows" each bar's move.
    Shift fills one bar later and all profit disappears.
    """
    sig = np.repeat([1.0, -1.0], n // 2)
    open_ = np.empty(n)
    close = np.empty(n)
    open_[0] = 3000.0
    for i in range(n):
        close[i] = open_[i] * (1 + 0.008 * sig[i])
        if i + 1 < n:
            open_[i + 1] = close[i]
    df = pd.DataFrame({"open": open_, "close": close})
    df["sig"] = sig
    return df


def same_bar_bt(df: pd.DataFrame) -> dict:
    """Trade INSIDE the bar: decide at bar close (signal), fill at that bar's open."""
    sig = df["sig"].astype(float).values
    o = df["open"].values
    c = df["close"].values
    n = len(df)
    pnl = 0.0
    trades = 0
    for i in range(1, n):
        pnl += (c[i] - o[i]) * sig[i]
        trades += 1
    return {"pnl": pnl, "trades": trades, "rets": np.full(n - 1, 0.008)}


def daily_signal_df(days: int = 80, bars_per_day: int = 48) -> pd.DataFrame:
    """A DAY-level label repeated over every intraday bar (implicit look-ahead)."""
    n = days * bars_per_day
    rng = np.random.default_rng(5)
    ret = rng.normal(0.0002, 0.002, n)
    closes = 3000 * np.cumprod(1 + ret)
    df = frame(closes)
    day_idx = np.arange(n) // bars_per_day
    # day's direction decided by close vs open of the SAME day (future info within day)
    day_dir = df.groupby(day_idx)["close"].transform("last") > \
        df.groupby(day_idx)["open"].transform("first")
    df["sig"] = np.where(day_dir, 1.0, -1.0)
    return df


def regime_trend_df(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    """Slow regime trend — used for the randomized-control EDGE demonstration."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.0015, n)
    for d in range(0, n // 48, 20):
        s = d * 48
        if s + 20 * 48 < n:
            ret[s:s + 10 * 48] += 0.004
            ret[s + 10 * 48:s + 20 * 48] -= 0.004
    closes = 3000 * np.cumprod(1 + ret)
    df = frame(closes)
    e20 = pd.Series(closes).ewm(span=20, adjust=False).mean().shift(1)
    e60 = pd.Series(closes).ewm(span=60, adjust=False).mean().shift(1)
    df["sig"] = np.where(e20 > e60, 1.0, -1.0)
    return df


def noise_df(n: int = 3000, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 3000 * np.cumprod(1 + rng.normal(0.0002, 0.002, n))
    df = frame(closes)
    rng2 = np.random.default_rng(seed + 1)
    df["sig"] = np.where(rng2.random(n) < 0.5, 1.0, -1.0)
    return df


def ar1_rets(n: int = 400, rho: float = 0.8, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print("\n" + "#" * 62)
    print("# " + title)
    print("#" * 62)


def main() -> None:
    # --- 1. honest next-open trend: gates clean with confirmed per-bar semantics ------
    banner("1) Honest next-open trend strategy  ->  PASS (gate checks)")
    df = regime_trend_df()
    aud = full_audit(df, "sig", next_open_hold(5), bar_seconds=300,
                     expansion_confirmation="completed",   # continuous trailing EMA sign:
                     seed=11, verbose=True)                # analyst confirms per-bar semantics
    print(f"   verdict: {aud['verdict']}   (problems: {len(aud['problems'])}, "
          f"warnings: {len(aud['warnings'])})")
    print(f"   note: expansion SUSPECT is resolved by an explicit 'completed' confirmation "
          f"(signal is recomputed every bar from trailing data); the other gates are clean")
    print(f"   note: randomized control EDGE_CONFIRMED and independence are informational")

    # --- 2. same-bar fill leak: lag PASS but fill-timing FAIL ----------------------
    banner("2) Same-bar confirmation + same-bar fill  ->  caught by fill-timing only")
    df2 = same_bar_leak_df()
    lag2 = lag_sensitivity(df2, "sig", same_bar_bt, verbose=True)
    fill2 = fill_timing_sensitivity(df2, same_bar_bt, verbose=True)
    print(f"   verdict: signal column is {lag2['verdict']} under lag, but "
          f"fill-timing is {fill2['verdict']} -> execution-level look-ahead")
    print(f"   (the column itself is a slow step; its period expansion would be "
          f"SUSPECT too — that is archetype 3's job)")

    # --- 3. daily label at bar level: period expansion is a hard gate ---------------
    banner("3) Day-level signal reused at 5-min bar level  ->  FAIL without confirmation")
    df3 = daily_signal_df()
    exp3 = period_expansion(df3, "sig", bar_seconds=300, verbose=True)
    aud3 = full_audit(df3, "sig", next_open_hold(5), bar_seconds=300,
                      expansion_confirmation=None, seed=7, verbose=True)
    print(f"   without explicit shifted/completed confirmation -> verdict: {aud3['verdict']}")
    aud3b = full_audit(df3, "sig", next_open_hold(5), bar_seconds=300,
                       expansion_confirmation="completed", seed=7, verbose=False)
    print(f"   with    explicit 'completed' confirmation      -> verdict: {aud3b['verdict']} "
          f"(expansion no longer blocks; the analyst must be able to justify it)")

    # --- 4. randomized control: edge vs beta ---------------------------------------
    banner("4) Randomized control: real edge vs bull-market beta")
    dft = regime_trend_df()
    rc_edge = randomized_control(dft, "sig", next_open_hold(5), n_shuffles=200,
                                 seed=11, verbose=True)
    dfn = noise_df()
    rc_noise = randomized_control(dfn, "sig", next_open_hold(5), n_shuffles=200,
                                  seed=12, verbose=True)
    print(f"   trend signal : {rc_edge['verdict']}  (real {rc_edge['real_pnl']:,.0f} vs "
          f"p95 {rc_edge['p95']:,.0f}, percentile {rc_edge['percentile']}%)")
    print(f"   noise signal : {rc_noise['verdict']}  (real {rc_noise['real_pnl']:,.0f} vs "
          f"p50 {rc_noise['p50']:,.0f}, p={rc_noise['p_value']})")

    # --- 5. overlapping returns: N_eff ---------------------------------------------
    banner("5) Return independence: '400 trades' that behave like ~90")
    for label, x in (("AR(1) rho=0.8 (overlapping)", ar1_rets()),
                     ("white noise (independent)", np.random.default_rng(99).normal(0, 1, 400))):
        rep = return_independence(x, verbose=True)
        print(f"   {label}: {rep['verdict']}  n={rep['n']} -> N_eff={rep['n_eff']}")

    print("\nDone. All outputs are deterministic (fixed seeds).")


if __name__ == "__main__":
    main()
