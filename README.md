# quant-backtest-validator

[![CI](https://github.com/FlyingEggs/quant-backtest-validator/actions/workflows/ci.yml/badge.svg)](https://github.com/FlyingEggs/quant-backtest-validator/actions/workflows/ci.yml)
<!-- Codecov badge: link the repo at codecov.io and uncomment.
[![codecov](https://codecov.io/gh/FlyingEggs/quant-backtest-validator/branch/main/graph/badge.svg)](https://codecov.io/gh/FlyingEggs/quant-backtest-validator)
-->

**Can this strategy backtest be trusted? — a one-call audit engine.**

A dependency-light audit framework (Python ≥ 3.9, numpy + pandas only, MIT) that separates
trustworthy backtests from overstated ones. V2 turns the mechanistic core into a
client-facing engine:

```python
from validator import audit, DataSpec, Strategy, audit_text

report = audit(strategy, df, DataSpec(bar_seconds=300), config={...})
print(audit_text(strategy, df, ...))
# -> QUANT BACKTEST VALIDATION REPORT ... Overall Verdict / Reliability / per-section
```

```bash
python3 examples/audit_demo.py                   # two full client-style reports -> ./reports/
python3 examples/demo.py                         # six mechanistic archetypes (primitives)
python3 -m unittest discover -s tests -v         # 177 unit tests (adversarial + V3 engines)
```

## What it is (and is not)

| | |
|---|---|
| **Is** | A validation engine: data integrity · look-ahead evidence · execution perturbation · return independence · OOS · parameter robustness — with an honest `NOT VERIFIED` for anything not actually checked |
| **Is not** | A backtest you can copy-paste a strategy into without declaring its signal, costs, and execution semantics |
| **Honesty rule** | Unimplemented capability (real cost model, MTF module) is reported `NOT VERIFIED`, never silently assumed clean |

## Structure

```
quant-backtest-validator/
├── validator/
│   ├── core.py            # primitives: lag / expansion / fill / RC / independence
│   ├── data_integrity.py  # frame sanity (index, OHLC, NaN)
│   ├── lookahead.py       # lag + expansion section (needs exposed signal column)
│   ├── execution.py       # entry semantics + fill-timing perturbation (always)
│   ├── statistics.py      # return independence / N_eff (needs per-trade rets)
│   ├── robustness.py      # randomized control + chronological OOS + param cliffs
│   ├── costs.py           # cost section gate (NOT VERIFIED/DECLARED/VERIFIED)
│   ├── costengine.py      # V3.2 net-PnL engine (adverse fills, tick, spread/slip/impact)
│   ├── wf.py               # V3.3 OOS/WF contract (boundary policy, param freeze)
│   ├── surface.py          # V3.4 2D parameter surface (plateau/island/ridge) + clustering
│   ├── mtf.py             # V3 temporal-availability engine (legal vs naive)
│   ├── report.py          # verdict assembly, reliability score, text render
│   ├── audit.py           # audit() / audit_text() entry points
│   └── types.py           # Strategy / DataSpec contracts
├── examples/  (audit_demo.py, demo.py)
├── tests/     (177 unit tests)
└── reports/   (sample JSON reports produced by audit_demo.py)
```

## Adversarial suite (the Validator is attacked, not just tested)

`tests/test_adversarial.py` encodes an expected verdict per scenario — cheating
strategies must never get a clean bill, legitimate ones must never be auto-FAILED:

| Scenario | Expected | Result |
|---|---|---|
| 01 honest trend | PASS (scoped) / INCOMPLETE (full scope: MTF roadmap) | ✓ |
| 02 same-bar fill cheat | FAIL (P0 fill + entry) | ✓ |
| 03 future-column signal | CONDITIONAL (P1 evidence; leak proof = code review) | ✓ not PASS |
| 04 low-frequency reuse | FAIL unconfirmed | ✓ |
| 06 costs none/understated | NOT VERIFIED / DECLARED (never auto-PASS) | ✓ |
| 07 IS-fit / OOS-broken | CONDITIONAL (OOS_INSTABILITY) | ✓ |
| 08 parameter cliff (1D) | PARAM_CLIFF detected (overall INCOMPLETE, no P0/P1) | ✓ |
| 09 heavily overlapping returns | Statistics CONDITIONAL + STAT_DEPENDENCE; overall PASS within complete scope | ✓ |
| 10 short-horizon legitimate | CONDITIONAL (never auto-FAIL) | ✓ |
| 11 overlapping-but-legit | PASS (scoped); dependence discounts, never kills | ✓ |
| 12 regime strategy | PASS (scoped) | ✓ |

Known capability gaps are `skip`ped with the reason stated (survivorship bias,
mechanical leak *proof*) — boundaries are documented, never faked as PASS.
2D parameter island detection was a documented gap until V3.4 and is now a live
assertion (`config['surface']`), not a skip.

## Statistics grading (v2.1.1)

Return dependence is never a silent PASS: `N_eff/n < 0.2` ⇒ section CONDITIONAL +
`STAT_DEPENDENCE`; `0.2–0.8` ⇒ P3 note. By policy it discounts significance claims
but does not flip the overall verdict (overlapping trades are not invalid trades).

## Verdict model — 4-state Audit Verdict Contract (V2.2)

Per-section statuses: **PASS / CONDITIONAL PASS / FAIL / DECLARED / NOT VERIFIED**. The
**overall** verdict is 4-state over the declared audit scope:

```
P0 present                -> FAIL
no P0, P1 present         -> CONDITIONAL PASS
no P0/P1, some NOT VERIFIED -> INCOMPLETE
no P0/P1, fully verified  -> PASS     (only "everything I checked is clean")
```

`PASS` therefore means *"I checked the declared scope and it is clean"* — never *"nothing is
wrong anywhere"*. Un-checked capability is `INCOMPLETE`, reported separately from blocking
findings. `config['scope']` narrows the declared scope (a scoped audit must state its scope;
PASS is only meaningful within it).

> ⚠️ **PASS ≠ green light for production.** PASS means *no violation detected in the scopes
> we checked*, not *"this strategy will definitely work live"*. Since V3.4.2 every report
> carries an `Interpretation:` line directly under the verdict that says exactly this — read
> the verdict and the interpretation together.

Statistical confidence is reported separately so a PASS can never mask weak significance:
`Significance: DISCOUNTED (N_eff 54/297, ratio 0.18) — verdict ≠ significance verdict`.
Costs: no config ⇒ `NOT VERIFIED` · config ⇒ `DECLARED` · `independently_verified: true` ⇒
`VERIFIED`.

## Sample report (reproduced by `examples/audit_demo.py`, deterministic)

```
QUANT BACKTEST VALIDATION REPORT
Strategy : EMA-trend (next-open, hold 5)
Engine   : 3.4.2
============================================================
Overall Verdict : INCOMPLETE
Interpretation  : No defect in the verified scope, but key dimensions were not
                  verified - missing evidence is not a clean bill.
Verified Score  : 93/100 (over VERIFIED scope only)
Audit Coverage  : 86%
Blocking        : P0=0  P1=0  P2=1
Significance    : DISCOUNTED (N_eff 54.4 / 297, ratio 0.18) - verdict != significance verdict
------------------------------------------------------------
AUDIT SCOPE
  ✓ Data Integrity   PASS
  ✓ Look-ahead       PASS
  ✓ Execution        PASS
  ✓ Statistics       CONDITIONAL PASS
  ✓ Robustness       PASS
  ✓ Costs            VERIFIED
  △ MTF              NOT VERIFIED
------------------------------------------------------------
Findings (severity-ordered):
  [P2] (Statistics) STAT_DEPENDENCE: heavy return dependence: N_eff/n=0.18 ...
  [P3] (Look-ahead) PERIOD_EXPANSION_CONFIRMED: expansion SUSPECT resolved by ...
  [P4] (MTF) MTF_MODULE: no MTF binding supplied ...
------------------------------------------------------------
Recommendation: No blocking findings in the VERIFIED scope, but the audit is
INCOMPLETE (coverage 86%; NOT VERIFIED: MTF). PASS is only granted when every
scope item is VERIFIED (DECLARED != VERIFIED) - complete the scope first.
============================================================
```

With MTF declared out of scope (and costs supplied), the same strategy audits to genuine
**PASS**. A leaky strategy (same-bar fills, declared `same_bar`) audits to **FAIL · 13/100**
with P0 `ENTRY_SEMANTICS` + `EXECUTION_FILL` listed first and **DO NOT DEPLOY**.

## OOS with warm-up context

Chronological OOS runs the strategy over the **full history** and filters entries to the OOS
window via the reserved `_from_bar` param (`Strategy.supports_from_bar=True`) — indicators warm
up on real prior data with no cold start and no look-ahead. Strategies without this opt-in are
flagged: *"OOS ran on a cold slice — indicator strategies may suffer cold-start"*. IS vs OOS
comparison is normalized (PnL/trade, trade counts reported).

## Two tiers, stated honestly

- **Code-level tier** — the strategy exposes its signal column + backtest function
  (`Strategy.signal_col`, `bt_mechanism`): full mechanism suite runs (lag sensitivity,
  period expansion, randomized control).
- **Black-box tier** — only `run(df, params)` metrics: data integrity, execution
  perturbation, independence, OOS and parameter sweeps still run; look-ahead and
  randomized control report `NOT VERIFIED` (they need the actual signal column).
  *Refusing to fake a signal-level check on a black box is the point.*

## Epistemology (primitives, v1.1+)

- Lag collapse is **evidence, not proof** — legitimately short-horizon signals collapse too;
  it is P1 (review), never auto-FAIL.
- Fill-timing is a **perturbation test**: <10% retention ⇒ P0 evidence of same-bar fills;
  10–50% ⇒ P1 (short horizon possible). Corroborate with an execution model.
- Randomized control compares against the signal's own time-shuffled null — evidence of
  timing information, **not** a standalone alpha certificate.
- N_eff is the linear-rho effective sample size of the mean (Kass et al. 1998);
  Ljung-Box (squared rho, adaptive lag order) handles detection; chi2 is exact via scipy
  when available, Wilson–Hilferty otherwise.

## V3 — MTF Temporal Availability Engine

Not a "does the 4h data exist" checker. Given a signal column on a LOW frame and a HIGH
frame (both in `DataSpec.timeframes`), the engine reconstructs, per decision time
`t_dec` (= low-bar close), two counterfactuals:

```
legal(t) = value of the last HIGH bar whose close_time <= t_dec   (usable)
naive(t) = value of the last HIGH bar whose open   <= t_dec       (may still be forming)
```

* column == `naive` on ≥90% of forming bars ⇒ **MTF_LEAK (P0)** — the signal used a
  higher-TF bar that had not closed yet (e.g., today's 23:59 daily close at 09:00).
* column == `legal` throughout ⇒ **PASS** — aligned with last-completed high bars
  (e.g., yesterday's close used all day: legal).
* partial/mixed matches (<90%) ⇒ **NOT VERIFIED** — chance-level alignment is not
  declared a leak (FP guard).

```python
spec = DataSpec(bar_seconds=300, timeframes={"h1": high_1h_df})
cfg = {"mtf": {"col": "sig", "frame": "h1",
               "frame_seconds": 3600, "transform": "sign_diff"}}
```

## V3.1 — Execution / Information Boundary

The perturbation test cannot prove the "decide at 09:35 close, fill at 09:35 close"
cheat. V3.1 adds a mechanical timeline over per-trade records
(`run()` returns optional `trades_log=[{"signal_ts", "entry_ts", ...}]`):

```
t_information (bar close)  ->  t_decision  ->  t_order  ->  t_fill
```

Every fill with `entry_ts <= signal_ts (+ min_latency_s)` is **EXECUTION_TIMELINE
(P0)** — information used before it was actionable. Legal next-open fills pass; a
strategy without per-trade timestamps reports this sub-check as NOT VERIFIED
(honest; the perturbation test still runs). `timeline_audit()` is unit-testable
directly and accepts datetime64 / pandas Timestamp / epoch ints.

## V3.2 — Realistic Cost & Net PnL Audit

`config['cost']` + per-trade `trades_log` runs the net engine. Every fill is adjusted
**against the trader** and tick-quantised against the trader (BUY rounds up, SELL rounds
down), so a cost model can never accidentally improve a fill. Layers report separately:

| Layer | Model | Anti-cheat |
|---|---|---|
| Commission | amount/notional/per-contract/fixed; `open_rate != close_rate` | C2 fee asymmetry |
| Tick size | adversarial quantisation to the legal grid | C3 100.003 -> legal tick |
| Spread | fixed | pct (half-spread per fill, adverse) | C4 never improves |
| Slippage | bps | pct | fixed | callable (separate from spread) | C4 never improves |
| Market impact | none (NOT VERIFIED) | linear | sqrt | callable | interface-first |
| Financing | funding_bps_per_day over held days (needs `entry_ts/exit_ts`) | honest NOT VERIFIED |

Result: a `PERFORMANCE AUDIT` table (Gross, per-layer drags, Net, Cost Drag) + per
sub-model PASS/NOT VERIFIED. Pure per-trade (order-invariant, no future/cross-row data) -
Case-5. Costs section states: no config => NOT VERIFIED · config, no trades_log =>
DECLARED · config + trades_log => VERIFIED (net audited).

## V3.5 — Invariants (adverse-cost, data-finite, state-consistency)

Three hard guarantees added after an adversarial cost round (V3.5.0):

* **Adverse-cost invariant** — the promise *"every fill is adjusted AGAINST the trader"*
  is now *enforced*, not just intended. A configured cost layer that would PAY the trader
  is rejected: static negative params (`COST_NEGATIVE`, P0) and runtime negative charges
  from callable modes (`COST_ENGINE_INVARIANT`, P0, engine verdict FAIL). **Financing is
  the sole exempt layer** — real funding regimes go negative (longs are paid), so a
  negative funding parameter is market semantics, not a cheat.
* **Data-finite invariant** — `+-inf` OHLC is a hard P0 (`DATA_NONFINITE`); it passes
  neither `<= 0` nor `isna()` and would silently poison signal/PnL/statistics/surface.
* **State-consistency** — VERIFIED with a dead sub-model is state leakage: a configured
  sub-model that cannot be verified (e.g. financing without `entry_ts/exit_ts`) downgrades
  the Costs section to NOT VERIFIED (`COST_SUB_INCOMPLETE`, P3) and the overall audit to
  INCOMPLETE — never a clean VERIFIED.
* **Config hygiene** — legacy flat-bps keys (`fee_bps`/`slippage_bps`) are not consumed by
  the V3.2 engine; they are reported `COST_CONFIG_UNUSED` (P2) instead of silently doing
  nothing while the section claims VERIFIED.
* **Declared data semantics** — `DataSpec.bar_timestamp_semantics` (`OPEN` default; MTF
  and execution timelines model the index as bar OPEN). A `CLOSE`-indexed frame is flagged
  `DATA_TS_SEMANTICS` (P3). MTF custom callable transforms are DECLARED
  (`MTF_TRANSFORM_DECLARED`, P3): the engine cannot see inside a callable, so a custom
  transform never yields a verified PASS — a detected leak on one still FAILs.

## V3.6 — Instrument realism (execution realism on the fills)

`DataSpec` carries an instrument contract (`qty_step` / `min_qty` / `min_notional` /
`contract_size`; all default = not declared). Enforced inside the net cost engine, so a
backtest whose fills **cannot actually execute** is surfaced instead of blessed:

* `EXEC_QTY_STEP` (P1) — qty not expressible as a multiple of `qty_step` (e.g. 1.237 on a
  0.1 lot); `EXEC_MIN_QTY` (P1) — ghost fills below `min_qty`; `EXEC_MIN_NOTIONAL` (P1) —
  `qty × contract_size × price` below the floor. Fills carrying P1 move Costs to
  CONDITIONAL PASS.
* `contract_size ≠ 1` scales the notional base for commission / financing (futures
  semantics; default 1.0 changes nothing).
* Market impact gains `volume_linear` (`coeff × (qty/volume) × price`, participation-rate
  model) — per-trade `volume` comes from `trades_log`. Configured but no volume data ⇒ the
  sub-model is NOT VERIFIED and the Costs section downgrades (state-consistency rule).
* Partial fills / queue position are NOT simulated — declared on the roadmap (intrabar
  execution model). Nothing declared ⇒ `execution` sub-check NOT VERIFIED, never assumed
  clean.

## V3.3 — OOS / Walk-Forward research contract

Activated by `config["oos"]`. Three machine contracts:

* **Trade boundary policy** — `ENTRY_IN_WINDOW | EXIT_IN_WINDOW |
  FULL_TRADE_IN_WINDOW`; the chosen policy is reported and cross-boundary trades are
  counted, never silently assigned.
* **Parameter freeze** — OOS runs must use frozen IS parameters. Two probes:
  *determinism* (same df+params => same output; else P0 `NON_DETERMINISTIC`) and
  *refit probe* (declared tunable `param_grid` yet identical output across grid
  extremes => P0 `PARAM_FREEZE`, internal re-fit / dead parameter suspected).
* **Walk forward** — expanding-IS / rolling-OOS windows; per-window IS/OOS
  PnL, PnL/trade, trades, status; aggregates positive-window %, expectancy
  consistency, trade adequacy (never a bare positive-window ratio).

## V3.4 — Parameter Surface (plateau / island / ridge) + trade clustering

Activated by `config["surface"] = {"x","y","x_values","y_values"}`. On a 2D pnl grid:

* **PLATEAU**  - >=60% of cells within 70% of best -> robust region (healthy).
* **ISLAND**   - best point isolated (all orthogonal neighbours < 70% of best) AND
  plateau < 25% -> parameter-mining island, **PARAM_ISLAND (P1)**.
* **RIDGE**    - best along one axis only -> one parameter informs, the other is noise
  (**PARAM_RIDGE P2**).
* **NOISY**    - fragmented surface.
* **NON_FINITE_PNL** - a grid cell exploded (NaN / inf / raised): the surface is
  *not classifiable* and is reported as **PARAM_NONFINITE_PNL (P1)** with the bad
  cells located — never silently reclassified, never mistaken for an island.
* **DEGENERATE_GRID** - fewer than 2 values on an axis: a point/line cannot support
  plateau/island/ridge claims -> **PARAM_DEGENERATE_GRID (P1)**, not a fake PLATEAU.
* `cluster_audit(trades_log)` - trades concentrated in few calendar days =>
  **TRADE_CLUSTERING (P3)** block dependence (not iid samples). Timestamps are
  normalised like the execution timeline (epoch ns/ms/s ints, datetime, ISO strings).

## Performance

Measured on a 2015 MacBook Pro (Intel Core i5-5350U @ 1.80 GHz), Python 3.9,
`audit()` over the full default scope. Reproduce with:

```bash
RUN_BENCHMARK=1 python3 -m unittest tests.test_benchmark -v        # daily + 5-min
RUN_BENCHMARK=1 BENCH_BIG=1 python3 -m unittest tests.test_benchmark -v   # + minute
```

| Dataset | Bars | RC shuffles | Wall time |
|---------|------|-------------|-----------|
| 5 years daily | 1,260 | 200 | **2.8 s** |
| 10 years daily | 2,520 | 200 | **13.1 s** |
| 1 year 5-min | 105,120 | 200 | **238.7 s** |
| 1 year minute | 525,600 | 50 | **333.9 s** |

*Randomization Control dominates: each shuffle reruns the full strategy backtest, so
wall time scales ~linearly with `n_shuffles` and with the strategy loop's row cost.
The numbers above use the pure-Python demo strategy (`next_open_hold`); a vectorised
backtest scales far better. RC=200 on 500k+ rows with a Python-loop strategy is the
one case to budget for — drop `n_shuffles` or vectorise, don't silently skip RC.*

## Roadmap (open items, by design)

| Module | Status |
|---|---|
| MTF temporal availability (V3) | ✅ legal-vs-naive engine |
| Execution / information boundary (V3.1) | ✅ timeline audit (`trades_log`), EXECUTION_TIMELINE P0 |
| Real cost engine (V3.2) | ✅ net-PnL engine - adverse fills, tick, spread/slip/impact/commission/financing |
| Intrabar execution model (partial fills, queue) | roadmap |
| OOS / Walk-Forward contract (V3.3) | ✅ boundary policy + parameter freeze + multi-window WF |
| Parameter surface (V3.4) | ✅ 2D plateau/island/ridge + trade clustering |
| CI / lint (V3.4.2) | ✅ GitHub Actions: unittest matrix 3.9–3.12 + coverage artifact + `mypy validator/` baseline clean (see `mypy.ini`) |
| Full `mypy --strict` cleanup | roadmap — needs report-container TypedDicts across the engine (heterogeneous report dicts currently use `Dict[str, Any]` deliberately) |
| Instrument realism (V3.6) | ✅ `DataSpec` qty_step/min_qty/min_notional/contract_size; EXEC_QTY_STEP/EXEC_MIN_QTY/EXEC_MIN_NOTIONAL; volume-aware impact (`volume_linear`); partial-fill/queue declared NOT VERIFIED (intrabar engine stays roadmap) |
| Parameter provenance (V3.7) | **planned** — `Strategy.fit_is` + `accepts_frozen` contract; frozen-vs-adversarial injection probe → PARAM_PROVENANCE P0 on hidden refit; frozen_hash into OOS evidence |
| Certification contract (V3.8) | **planned** — audit_id / generated_at / strategy_hash / data_hash anchors; L0-L4 certification level over sections (L5 adversarial suite, L6 live parity, L7 signed immutable audits: engine max L4, higher levels are product roadmap) |
| Statistical significance certification (V3.9+) | **planned** — multiple-testing correction · White's Reality Check / SPA · Deflated & Probabilistic Sharpe Ratio · regime/bootstrap block dependence (own workstream before implementation) |

## Who's Using This

<!-- If you use quant-backtest-validator in your workflow and it helped you catch a
     look-ahead bug or an overstated backtest, open an issue/PR to be featured here. -->
_None yet — this section is a placeholder. First case study in progress._

## Honesty notes

All data is synthetic; the engine demonstrates *mechanisms*, not market alpha. Reference
implementation for validation methodology — not the production engine behind any specific
strategy. Audit ≠ investment advice.

## License

MIT © 2026 AuroraCode. See `LICENSE`.
