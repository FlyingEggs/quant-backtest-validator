# quant-backtest-validator

**Can this strategy backtest be trusted?**

A small, dependency-light reference implementation of the *mechanistic* checks that separate a
trustworthy backtest from an overstated one. Pure functions over a DataFrame + a strategy
function — no exchange, no data vendor, no strategy parameters to leak.

- Python ≥ 3.9 · numpy + pandas only · deterministic (fixed seeds)
- MIT licensed

```bash
python3 examples/demo.py                      # run the five archetypes
python3 -m unittest discover -s tests -v      # 11 unit tests
```

## Why backtests overstate performance

| Channel | How it inflates | Mechanistic check |
|---|---|---|
| Look-ahead bias | Signal uses the same bar's close, fills at that close | **Lag sensitivity** — shift the signal 1 bar; edge collapse ⇒ look-ahead |
| Same-bar / intraday fills | "I enter immediately" — but the bar isn't closed | **Fill-timing** — shift fills 1 bar later; collapse ⇒ impossible fills |
| Low-frequency column at bar level | A daily label repeated on every intraday bar | **Period expansion** — longest constant run ⇒ SUSPECT, hard gate |
| β as alpha | Random longs also made money in this bull market | **Randomized control** — permutation test, real PnL vs random |
| Overlapping positions | 400 trades that behave like ~90 | **Return independence** — Ljung-Box + ACF + N_eff |

## What the code does

`validator/core.py` exposes five standalone engines plus a `full_audit` synthesizer:

- `lag_sensitivity(df, col, bt, lag)` — re-run with the state column shifted; PASS/WARN/FAIL.
- `period_expansion(df, col, bar_seconds)` — longest constant run in hours ⇒ SUSPECT/OK.
- `fill_timing_sensitivity(df, bt)` — shift every fill price one bar; collapse ⇒ FAIL.
- `randomized_control(df, col, bt, n_shuffles, seed)` — EDGE_CONFIRMED / EDGE_WEAK / NO_EDGE.
- `return_independence(rets)` — AUTOCORRELATED (N_eff << n) / INDEPENDENT / INSUFFICIENT.
- `full_audit(...)` — gate synthesis → **PASS / FAIL**, with problems + informational warnings.

A strategy is any **pure** function `bt(df) -> {"pnl": float, "trades": int[, "rets": array]}`.

## Reproducible demo output

```
1) Honest next-open trend (per-bar semantics confirmed "completed")
   full audit -> PASS      (lag & fill clean; expansion confirmed; RC EDGE_CONFIRMED informational)

2) Same-bar confirmation + same-bar fill        <- execution look-ahead
   lag sensitivity : PASS     (signal column itself is clean)
   fill-timing     : FAIL     pnl 17,319,566 -> 22.7   (~100% of profit was intraday)

3) Day-level signal reused at 5-min bars        <- period expansion
   without confirmation -> FAIL  (longest constant run 720 bars = 60h)
   with 'completed'     -> PASS  (the analyst must be able to justify it)

4) Randomized control (200 shuffles)
   trend signal : EDGE_CONFIRMED   real 48,884 vs random p95 4,093 (p=0.005)
   noise signal : NO_EDGE          real  2,238 vs random p50 2,402 (p=0.786)

5) Return independence
   AR(1) rho=0.8 overlapping trades : AUTOCORRELATED  n=400 -> N_eff=88
   white noise                       : INDEPENDENT     n=400 -> N_eff=379
```

Every number above is produced by running this repository's `examples/demo.py`
(deterministic seeds) — not by a private engine.

## Coverage vs. the full audit methodology

This repository implements the **mechanistic core** (the five engines above). A full client
methodology also covers OOS splits, walk-forward, parameter sensitivity sweeps, cost/funding
models, and live-vs-backtest execution review — performed per engagement with bespoke tooling.
See `docs/methodology_backtest_validation.md` for the end-to-end framework this core feeds into.

## Honesty notes

- All data is **synthetic**. The engines demonstrate *mechanisms*, not market alpha.
- This is a reference implementation for validation methodology — it is not the production
  engine behind any specific strategy, and it makes no claims about future returns.
- Audit ≠ investment advice.

## License

MIT © 2026 AuroraCode. See `LICENSE`.
