# quant-backtest-validator

**Can this strategy backtest be trusted?**

A small, dependency-light reference implementation of the *mechanistic* checks that separate a
trustworthy backtest from an overstated one. Pure functions over a DataFrame + a strategy
function — no exchange, no data vendor, no hidden strategy parameters.

- Python ≥ 3.9 · numpy + pandas only · deterministic (fixed seeds) · MIT licensed

```bash
python3 examples/demo.py                      # run the six archetypes
python3 -m unittest discover -s tests -v      # 19 unit tests
```

## Why backtests overstate performance

| Channel | How it inflates | Mechanistic check |
|---|---|---|
| Look-ahead / future functions | Signal uses the same bar's close, fills at that close | **Lag sensitivity** — shift the signal 1 bar |
| Same-bar / intraday fills | "I enter immediately" — but the bar isn't closed | **Fill-timing** — shift fills 1 bar later |
| Low-frequency column at bar level | A daily label repeated on every intraday bar | **Period expansion** — constant-run hard gate |
| Static-exposure beta as alpha | Random longs also made money in this bull market | **Randomized control** — time-shuffled null |
| Overlapping positions | 400 correlated trades reported as 400 samples | **Return independence** — Ljung-Box + N_eff |

## Verdict model

A full audit returns **PASS / CONDITIONAL PASS / FAIL / INSUFFICIENT** with a **P0–P4 issue log**:

| Verdict | Meaning |
|---|---|
| PASS | no P0/P1 findings |
| CONDITIONAL PASS | no P0, but P1 evidence needing manual confirmation (e.g. lag-dependence) |
| FAIL | ≥1 P0 finding (execution look-ahead, unconfirmed low-frequency expansion, non-next-open entry) |
| INSUFFICIENT | no trades in baseline — not assessable |

Severity: **P0** = result-distorting · **P1** = materially uncertain / needs review ·
**P2/P3** = conventions & hygiene · **P4** = polish. Full report is a JSON-serializable dict
(`validator.save_report`).

## Epistemology (what the checks do — and don't — prove)

- **Lag collapse is evidence, not proof.** A genuinely short-horizon strategy also loses money
  when its signal is one bar stale. That is why lag-dependence is reported P1
  (CONDITIONAL PASS + manual construction/timestamp review), never an automatic FAIL.
  *Measured:* for a legitimate 1-bar-horizon signal, a one-bar signal lag and a one-bar fill
  delay are mathematically the **same** perturbation (~70% of the edge in both cases).
- **Fill-timing FAIL is graded by retention.** If shifting fills one bar later leaves **<10%**
  of profit, fills were effectively same-bar (P0). A 10–50% retention is consistent with a
  legitimate short holding horizon and is P1 review.
- **Randomized control compares against the signal's own time-shuffled null** (same value set,
  timing destroyed; the null keeps average long exposure). Verdicts are
  `BEATS_SHUFFLED_NULL` / `WEAK_VS_SHUFFLED_NULL` / `NO_EDGE_VS_SHUFFLED_NULL` — evidence of
  timing information beyond static exposure, **not** a standalone alpha certificate.
- **N_eff is the effective sample size of the mean** (linear-rho inflation factor
  `1 + 2 Σ (1-k/n) ρ_k`, Kass et al. 1998). Autocorrelation *detection* uses Ljung-Box
  (squared rho) — different questions, different formulas.

## Reproducible demo output (`examples/demo.py`, deterministic)

```
1) Honest next-open trend (EMA trailing signal, confirmed semantics)
   -> PASS            lag STABLE · fill PASS · expansion confirmed ·
                      RC BEATS_SHUFFLED_NULL (informational)

2) Legitimate 1-bar-horizon signal
   -> CONDITIONAL PASS   lag LAG_DEPENDENT (P1) · fill retains 31% (P1)
                          a legal fast strategy is never auto-FAILED

3) Same-bar confirmation + same-bar fill
   -> FAIL (P0)  lag STABLE, but fill-timing pnl 17,319,566 -> 22.7
                  (<0.01% retained = fills knew the bar before it closed)

4) Day-level signal reused at 5-min bars
   -> FAIL (P0)  without confirmation (longest run 720 bars = 60h)
   -> PASS       with explicit 'completed' confirmation

5) Randomized control vs shuffled null (200 shuffles)
   trend signal : BEATS_SHUFFLED_NULL   real 48,884 vs p95 4,093 (p=0.005)
   noise signal : NO_EDGE_VS_SHUFFLED_NULL  real 2,238 vs p50 2,402 (p=0.786)

6) Return independence (linear-rho N_eff)
   AR(1) rho=0.8 overlapping trades : AUTOCORRELATED  400 -> N_eff=53
   white noise                      : INDEPENDENT     400 -> N_eff=400
```

Every number above is produced by running this repository — no private engine involved.

## Fill-timing scope note

This check shifts the fill-relevant price columns (default `open`; pass `high/low/close` too if
your stops/limits touch them) and injects a `__fill_shifted__` marker so path-dependent
strategies can delay exit scans. A full intrabar execution model (stop/limit simulation,
slippage, partial fills) is a separate module on the roadmap, not part of this check.

## Coverage vs. the full audit methodology

This repository implements the **mechanistic core**. A client engagement additionally covers data
integrity, cost/funding models, OOS, walk-forward, parameter robustness, and live-vs-backtest
parity — see `docs/methodology_backtest_validation.md`. Architecture direction:

```
Quant Backtest Validator
├── Core Engine        (this repo: lookahead · execution · expansion · control · independence)
├── Audit Modules      (data · cost · OOS · walk-forward · robustness · live parity)
├── Report Engine      (three-tier verdict · P0-P4 issue log · JSON)
└── Certification      (signature / no-signature-no-deployment)
```

## Honesty notes

- All data is **synthetic**; the engines demonstrate *mechanisms*, not market alpha.
- Reference implementation for validation methodology — not the production engine behind any
  specific strategy; makes no claims about future returns.
- Audit ≠ investment advice.

## License

MIT © 2026 AuroraCode. See `LICENSE`.
