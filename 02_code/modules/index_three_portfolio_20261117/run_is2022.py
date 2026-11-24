"""Diagnose the 2018-2022 combined IS for the 2023-start frozen choices."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

import engine

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "04_results/index_forecast_development_20261117"
BASE = ROOT / "04_results/index_forecast_is_oos_20261117"
OUT = ROOT / "04_results/index_three_portfolio_is2022_20261118"


def main():
    OUT.mkdir(exist_ok=True)
    sources = [DEV / x for x in ["selected_IS_only.csv", "accuracy.csv", "execution_targets.parquet"]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    spec = importlib.util.spec_from_file_location("combined_is_resources", ROOT / "02_code/modules/index_forecast_is_oos_20261117/evaluate.py")
    resources = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = resources
    spec.loader.exec_module(resources)
    resources.write = lambda name, data: None
    px, tv, actions = resources.resources()
    schedule = pd.read_csv(BASE / "case_schedule.csv", parse_dates=["entry_date", "exit_date", "information_date"])
    accuracy = pd.read_csv(DEV / "accuracy.csv", low_memory=False)
    accuracy = accuracy[accuracy.train_end.eq(2022)]
    targets = pd.read_parquet(DEV / "execution_targets.parquet")
    targets = targets[targets.train_end.eq(2022)]
    choices = pd.read_csv(DEV / "selected_IS_only.csv")
    choices = choices[choices.train_end.eq(2022) & choices.role.eq("IS_sharpe_selected")]
    assert len(choices) == 3 and set(choices.index_id) == {"KOSPI100", "KOSPI200", "KRX300"}
    m, d, e, l, h, checks = engine.simulate(choices, schedule, accuracy, targets, px, tv, actions,
                                           shared=True, start="2018-01-02", end="2022-12-29")
    assert checks["all_events_closed"] and checks["shared_security_day_cap"]
    assert checks["min_cash"] >= -.001 and checks["max_limit_excess"] <= .001
    assert np.isclose(m.loc[m.index_id.eq("COMBINED"), "profit_krw"].iloc[0],
                      m.loc[m.index_id.ne("COMBINED"), "profit_krw"].sum(), atol=.01)
    combined = d[d.index_id.eq("COMBINED")].sort_values("date")
    first_krx = pd.Timestamp(schedule[(schedule.index_id.eq("KRX300")) & schedule.event.eq("2018-12") & schedule.lead.eq(80)].entry_date.iloc[0])
    x = combined[combined.date.ge(first_krx)].return_daily
    common_sharpe = float(x.mean() / x.std(ddof=1) * np.sqrt(252))
    m.to_csv(OUT / "performance.csv", index=False, encoding="utf-8-sig")
    d.to_parquet(OUT / "daily.parquet", index=False)
    e.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    result = dict(IS=["2018-01-02", "2022-12-29"], oos=["2023-01-02", "2026-06-30"],
                  IS_selection_frozen_from="2018-2022 per-strategy historical Sharpe selection",
                  common_KRX300_first_entry=str(first_krx.date()),
                  combined_sharpe_from_KRX_first_entry_no_nav_reset=common_sharpe,
                  shared_cash_and_liquidity=True, budget_per_sleeve_krw=3e9,
                  no_combined_parameter_tuning=True,
                  source_hashes=hashes, checks=checks)
    (OUT / "protocol.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    print(m.to_string(index=False), flush=True)
    print("KRX first entry", first_krx, "same-account IS Sharpe from date", common_sharpe, flush=True)
    print("events", e.groupby("index_id").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
