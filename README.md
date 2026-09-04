# quant-backtest-validator

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
python3 -m unittest discover -s tests -v         # 53 unit tests (adversarial + V3 MTF)
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
│   ├── costs.py           # cost model gate (NOT VERIFIED until supplied)
│   ├── mtf.py             # V3 temporal-availability engine (legal vs naive)
│   ├── report.py          # verdict assembly, reliability score, text render
│   ├── audit.py           # audit() / audit_text() entry points
│   └── types.py           # Strategy / DataSpec contracts
├── examples/  (audit_demo.py, demo.py)
├── tests/     (27 unit tests)
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

Known capability gaps are `skip`ped with the reason stated (survivorship bias, 2D
parameter island, MTF temporal engine, mechanical leak *proof*) — boundaries are
documented, never faked as PASS.

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

Statistical confidence is reported separately so a PASS can never mask weak significance:
`Significance: DISCOUNTED (N_eff 54/297, ratio 0.18) — verdict ≠ significance verdict`.
Costs: no config ⇒ `NOT VERIFIED` · config ⇒ `DECLARED` · `independently_verified: true` ⇒
`VERIFIED`.

## Sample report (reproduced by `examples/audit_demo.py`, deterministic)

```
QUANT BACKTEST VALIDATION REPORT
Strategy : EMA-trend (next-open, hold 5)        Engine : 2.2.0
Overall Verdict : INCOMPLETE        (MTF on the roadmap - not a clean bill)
Verified Score  : 91/100 (over VERIFIED scope only)
Audit Coverage  : 86%
Blocking        : P0=0  P1=0  P2=1
Significance    : DISCOUNTED (N_eff 54.4/297, ratio 0.18)
AUDIT SCOPE     ✓ Data Integrity · ✓ Look-ahead · ✓ Execution ·
                ✓ Statistics (CONDITIONAL) · ✓ Robustness · ✓ Costs (DECLARED) · △ MTF
Findings: [P2] STAT_DEPENDENCE · [P3] expansion confirmed · [P3] costs declared
Recommendation: no blocking in VERIFIED scope, but INCOMPLETE - complete scope for PASS.
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

## Roadmap (open items, by design)

| Module | Status |
|---|---|
| Execution model (intrabar stops/limits, slippage) | roadmap |
| Real cost engine (funding history, fee schedules) | gate in place — `NOT VERIFIED` until supplied |
| MTF temporal availability (V3) | ✅ implemented — legal-vs-naive engine, needs binding + frames |
| OOS trade-boundary policy + multi-window walk-forward | roadmap |
| Parameter surface (2D island) / cluster dependence | roadmap |

## Honesty notes

All data is synthetic; the engine demonstrates *mechanisms*, not market alpha. Reference
implementation for validation methodology — not the production engine behind any specific
strategy. Audit ≠ investment advice.

## License

MIT © 2026 AuroraCode. See `LICENSE`.
