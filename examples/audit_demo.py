"""examples/audit_demo.py — the V2 one-call audit on two synthetic strategies.

Run from the repository root:

    python3 examples/audit_demo.py

Prints a client-style report and writes JSON reports into ./reports/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validator import (audit, audit_text, DataSpec, Strategy, as_code_strategy,
                       save_report)
from examples import demo as D


def honest_strategy() -> Strategy:
    df = D.regime_trend_df()
    bt = D.next_open_hold(5)
    strat = as_code_strategy("EMA-trend (next-open, hold 5)", df, "sig", bt,
                             entry_semantics="next_open",
                             description="trailing EMA-20/60 regime trend")
    return strat, df


def leaky_strategy() -> Strategy:
    df = D.same_bar_leak_df()
    strat = as_code_strategy("same-bar confirmation fill", df, "sig", D.same_bar_bt,
                             entry_semantics="same_bar",          # declared illegal
                             description="decides at bar close, fills at that bar's open")
    return strat, df


def main() -> None:
    os.makedirs("reports", exist_ok=True)
    spec = DataSpec(bar_seconds=300, source="synthetic")

    print("\n################  AUDIT #1: HONEST STRATEGY  ################")
    strat1, df1 = honest_strategy()
    cfg1 = {"expansion_confirmation": "completed",
            "cost": {"commission": {"mode": "bps", "open_rate": 4.0, "close_rate": 4.0},
                 "slippage": {"mode": "bps", "value_bps": 2.0},
                 "tick_size": None},
            "seed": 11}
    rep1 = audit(strat1, df1, spec, cfg1)
    print(audit_text(strat1, df1, spec, cfg1))
    save_report(rep1, "reports/audit_honest.json")

    print("\n################  AUDIT #2: LEAKY STRATEGY  ################")
    strat2, df2 = leaky_strategy()
    cfg2 = {"expansion_confirmation": "completed", "seed": 7}   # costs left unverified
    rep2 = audit(strat2, df2, spec, cfg2)
    print(audit_text(strat2, df2, spec, cfg2))
    save_report(rep2, "reports/audit_leaky.json")

    print("\nDone. JSON reports written under ./reports/")


if __name__ == "__main__":
    main()
