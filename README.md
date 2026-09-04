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
python3 -m unittest discover -s tests -v         # 27 unit tests
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
│   ├── report.py          # verdict assembly, reliability score, text render
│   ├── audit.py           # audit() / audit_text() entry points
│   └── types.py           # Strategy / DataSpec contracts
├── examples/  (audit_demo.py, demo.py)
├── tests/     (27 unit tests)
└── reports/   (sample JSON reports produced by audit_demo.py)
```

## Verdict model

Overall and per-section: **PASS / CONDITIONAL PASS / FAIL / NOT VERIFIED**, with a
severity-ordered **P0–P4 issue log** and a reliability score (100 − Σ severity weights).
P0 = result-distorting · P1 = needs manual confirmation · P2/P3 = conventions/hygiene ·
P4 = polish. `NOT VERIFIED` sections never subtract score — they are listed as open items.

## Sample report (reproduced by `examples/audit_demo.py`, deterministic)

```
QUANT BACKTEST VALIDATION REPORT
Strategy : EMA-trend (next-open, hold 5)        Engine : 2.0.0
Overall Verdict : PASS                          Reliability : 96/100
Data Integrity  PASS     Look-ahead PASS     Execution PASS
Statistics      PASS     Robustness PASS     Costs    PASS
MTF             NOT VERIFIED
Findings: [P4] MTF roadmap · [P3] expansion confirmed · [P3] costs declared, not re-run
Recommendation: no blocking findings; treat NOT VERIFIED (MTF) as open.
```

A leaky strategy (same-bar fills, declared `same_bar`) audits to **FAIL · 13/100** with P0
`ENTRY_SEMANTICS` + `EXECUTION_FILL` and **DO NOT DEPLOY** recommendation.

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

## Roadmap (open items, by design)

| Module | Status |
|---|---|
| Execution model (intrabar stops/limits, slippage) | roadmap |
| Real cost engine (funding history, fee schedules) | gate in place — `NOT VERIFIED` until supplied |
| MTF alignment checks | roadmap — `NOT VERIFIED` |
| Walk-forward / multi-window parameter robustness | robustness.py has single-split OOS + param cliffs; WF next |

## Honesty notes

All data is synthetic; the engine demonstrates *mechanisms*, not market alpha. Reference
implementation for validation methodology — not the production engine behind any specific
strategy. Audit ≠ investment advice.

## License

MIT © 2026 AuroraCode. See `LICENSE`.
