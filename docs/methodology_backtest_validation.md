# Backtest Validation Methodology (client-facing)

**Independent Quant Strategy Validation** — 19 checks in 9 families · Verdict-based output.
This is the checklist an engagement follows. Severity-graded findings; no parameter tuning.

> Repository note: `quant-backtest-validator` implements the **mechanistic core** of this
> framework — lag sensitivity (look-ahead), period-expansion gate, fill-timing, randomized
> control, and return independence (N_eff). The remaining checks (data integrity, MTF
> alignment, cost/slippage models, OOS / walk-forward / parameter sweeps, live-vs-backtest
> parity) are executed per engagement with bespoke tooling and the client's own data.

---

## Verdict rules

| Verdict | Meaning |
|---|---|
| **PASS** | No P0/P1 findings; P2+ recorded and do not change the conclusion |
| **CONDITIONAL PASS** | No P0; fixable P1 or P2 issues → verdict issued together with the corrected re-run |
| **FAIL** | Any P0 (result-distorting flaw) → report must not be used as a deployment/performance basis |

**Severity:** P0 = directly distorts backtest results · P1 = materially changes magnitude ·
P2 = statistical/convention issue · P3 = engineering quality · P4 = polish.

A 0–100 reliability score is provided as a communication aid only; the verdict is governed by
P0/P1 presence, never by the score.

---

## A. Data integrity

1. **Source & alignment** — timestamp semantics (open/close instants, timezone), inclusion of
   unclosed bars, OHLC ordering/units, duplicates/gaps, mixing of backfill vs incremental data.
2. **Indicator warm-up** — rolling windows with partial `min_periods`; NaN→0 fill artifacts;
   normalization/despiking that uses full-sample statistics (leak).

## B. Future function & look-ahead

3. **Future functions** — any signal input not available at decision time (incl. full-sample
   scaling). Detection: shift the state column 1 bar (or one low-frequency segment) and re-run —
   profit collapse ⇒ look-ahead.
4. **Low-frequency column reuse** — a daily-level state used as an intraday signal repeats the
   same value for many bars (implicit look-ahead). Detection: longest constant-run ≥ threshold ⇒
   SUSPECT; the column must be explicitly confirmed as shifted/completed before it may enter.

## C. Execution semantics

5. **Entry/exit timing** — signal bar must close before a fill is legal; next-open is the only
   default-legal entry; same-bar/intraday fills are look-ahead by construction.
6. **Fill-timing sensitivity** — shift all fill prices 1 bar later and re-run. A strategy that
   collapses was implicitly "knowing the close, filling at the open". Catches execution leaks
   that column checks alone cannot see.

## D. Multi-timeframe alignment

7. **MTF resample edges** — higher-timeframe bars aggregated from lower TF must not snapshot an
   unfinished higher-TF bar; left/right boundary labels must not shift; indicator values must be
   identical between backtest and the intended live path.

## E. Transaction costs & slippage

8. **Fees & funding** — per-side taker fee floor (no zero-fee backtests); for crypto-perpetuals:
   real historical funding settlements, with an explicit degrade path if unavailable (never a
   constant positive "free money for shorts" assumption).
9. **Slippage & contract precision** — fill-price assumptions vs achievable prices; position
   rounding to contract size and the resulting notional error; order size vs liquidity.

## F. Position accounting & risk

10. **Accounting & liquidation** — compounding vs fixed-fraction conventions; concurrent
    positions and shared-account conflicts; liquidation / auto-deleveraging / negative balance
    modeled where relevant; a single source of truth for open state.
11. **Drawdown & tail conventions** — drawdown algorithm (peak-to-trough, deposits/withdrawals);
    skewed/tail distributions reported as **median + tail**, never averaged into a headline.

## G. Statistics

12. **Metric accounting** — win-rate / profit-factor / Sharpe conventions stated; infinite PF
    serialization; zero-PnL trades in the denominator; annualization-frequency assumption declared.
13. **Return independence / effective sample size** — overlapping positions make adjacent trades
    correlated: report Ljung-Box/ACF and N_eff, and discount significance claims accordingly.

## H. Robustness

14. **Out-of-sample** — chronological split only (random K-fold is invalid for time series);
    no re-tuning after the split; OOS reported separately.
15. **Walk-forward** — parameters fitted inside each window, executed out-of-window; window
    count/step declared; per-window metrics.
16. **Parameter sensitivity** — neighborhood sweep; isolated peaks / cliffs flagged;
    no "failed → change parameter → retry until pretty" loop (must fix the signal construction).
17. **Randomized / permutation control** — shuffle the signals and compare real PnL to the
    random distribution (percentile & p). Separates genuine edge from bull-market beta
    ("random longs also made money").

## I. Backtest/live consistency & engineering

18. **Signal parity** — live and backtest must share one implementation (import, not copy-paste);
    identical entry-confirmation semantics, exits, and parameters on both sides.
19. **Certification & execution hygiene** — audit signature attached to any result used for
    decisions ("no signature, no deployment"); a failed re-audit must invalidate the previous
    pass; execution paths check state sync/reconciliation, order hygiene, and restart behavior.

---

## Deliverables

1. Verdict: **PASS / CONDITIONAL PASS / FAIL** + one-line rationale.
2. Severity-graded issue log (P0–P4): location · impact · evidence · fix suggestion.
3. Reliability score (0–100, communication aid).
4. Optional independent re-test: reported vs independent metrics, side by side.

*Scope & limits are stated per engagement. Validation confirms what the history shows; it does
not guarantee future performance. Not investment advice.*
