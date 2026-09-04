"""examples/demo.py — run the validator on six reproducible archetypes.

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
    """Slow regime trend — used for the clean-PASS and shuffle-control demonstrations."""
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


def markov_short_df(n: int = 20000, q: float = 0.65, seed: int = 1) -> pd.DataFrame:
    """A LEGITIMATE 1-bar-horizon signal: sign of today's move predicts tomorrow's move
    (sign-persistence q=0.65), entered at next open, exited at the following open.

    Deliberately short-horizon: both a 1-bar signal lag and a 1-bar fill delay destroy
    ~70% of the edge. The engine must NOT auto-fail this — it reports CONDITIONAL PASS
    and asks for execution-semantics review (short horizon is legal, same-bar is not).
    """
    rng = np.random.default_rng(seed)
    s = np.empty(n)
    s[0] = 1.0
    for i in range(1, n):
        s[i] = s[i - 1] if rng.random() < q else -s[i - 1]
    rets = s + rng.normal(0.0, 0.5, n)
    closes = np.cumsum(rets)
    df = frame(closes)
    df["sig"] = np.sign(closes - df["open"].to_numpy())     # sign of the bar's move
    return df


def markov_bt(df: pd.DataFrame) -> dict:
    """Legal next-open, 1-bar horizon: enter at open[i+1] on +1 signal, exit open[i+2]."""
    sig = df["sig"].fillna(0).to_numpy()
    o = df["open"].to_numpy()
    n = len(df)
    pnl, trades = 0.0, 0
    for i in range(n):
        if sig[i] > 0 and i + 2 < n:
            pnl += o[i + 2] - o[i + 1]
            trades += 1
    return {"pnl": pnl, "trades": trades, "rets": np.full(trades, 0.01)}


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


def issue_counts(aud: dict):
    p0 = sum(1 for i in aud["issues"] if i["severity"] == "P0")
    p1 = sum(1 for i in aud["issues"] if i["severity"] == "P1")
    return p0, p1


def main() -> None:
    # --- 1. honest next-open trend: PASS ----------------------------------------------
    banner("1) Honest next-open trend strategy  ->  PASS")
    df = regime_trend_df()
    aud = full_audit(df, "sig", next_open_hold(5), bar_seconds=300,
                     expansion_confirmation="completed",   # trailing EMA sign: analyst
                     seed=11, verbose=True)                # confirms per-bar semantics
    p0, p1 = issue_counts(aud)
    print(f"   verdict: {aud['verdict']}   (P0={p0}, P1={p1}, "
          f"issues={len(aud['issues'])}, warnings={len(aud['warnings'])})")

    # --- 2. legitimate 1-bar-horizon signal: CONDITIONAL PASS (not auto-FAIL) ---------
    banner("2) Legitimately short-horizon signal (1-bar edge)  ->  CONDITIONAL PASS")
    dfm = markov_short_df()
    audm = full_audit(dfm, "sig", markov_bt, bar_seconds=300, seed=5, verbose=True)
    p0m, p1m = issue_counts(audm)
    print(f"   verdict: {audm['verdict']}   (P0={p0m}, P1={p1m})")
    print(f"   note: lag AND fill both collapse ~70% - for a 1-bar-horizon signal the two "
          f"shifts are the same perturbation, so this is reported CONDITIONAL (manual "
          f"execution-semantics review), never auto-FAIL")

    # --- 3. same-bar fill leak: lag STABLE but fill-timing FAIL ----------------------
    banner("3) Same-bar confirmation + same-bar fill  ->  FAIL (fill-timing, P0)")
    df2 = same_bar_leak_df()
    lag2 = lag_sensitivity(df2, "sig", same_bar_bt, verbose=True)
    fill2 = fill_timing_sensitivity(df2, same_bar_bt, verbose=True)
    print(f"   signal column is {lag2['verdict']} under lag, but fill-timing is "
          f"{fill2['verdict']} -> execution-level look-ahead")
    print(f"   (the column itself is a slow step; its period expansion would be SUSPECT "
          f"too - a separate gate, see next block)")

    # --- 4. daily label at bar level: period expansion is a hard gate ----------------
    banner("4) Day-level signal reused at 5-min bar level  ->  FAIL without confirmation")
    df3 = daily_signal_df()
    exp3 = period_expansion(df3, "sig", bar_seconds=300, verbose=True)
    aud3 = full_audit(df3, "sig", next_open_hold(5), bar_seconds=300,
                      expansion_confirmation=None, seed=7, verbose=True)
    print(f"   without explicit shifted/completed confirmation -> verdict: {aud3['verdict']}")
    aud3b = full_audit(df3, "sig", next_open_hold(5), bar_seconds=300,
                       expansion_confirmation="completed", seed=7, verbose=False)
    print(f"   with    explicit 'completed' confirmation      -> verdict: {aud3b['verdict']} "
          f"(expansion no longer blocks; the analyst must be able to justify it)")

    # --- 5. randomized control vs the shuffled null ----------------------------------
    banner("5) Randomized control: signal vs its time-shuffled null")
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
    print(f"   note: the null keeps the average long exposure - it controls static-beta; "
          f"it is NOT a standalone alpha proof")

    # --- 6. overlapping returns: N_eff (linear-rho ESS of the mean) ------------------
    banner("6) Return independence: '400 trades' that behave like ~50")
    for label, x in (("AR(1) rho=0.8 (overlapping)", ar1_rets()),
                     ("white noise (independent)", np.random.default_rng(99).normal(0, 1, 400))):
        rep = return_independence(x, verbose=True)
        print(f"   {label}: {rep['verdict']}  n={rep['n']} -> N_eff={rep['n_eff']} "
              f"(inflation x{rep['inflation_factor']})")

    print("\nDone. All outputs are deterministic (fixed seeds).")


if __name__ == "__main__":
    main()
