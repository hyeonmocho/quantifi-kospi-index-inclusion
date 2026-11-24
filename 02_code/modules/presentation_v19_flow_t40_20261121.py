"""Rebase the public-basket flow chart at T-40 with T-41 market-cap denominators."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "04_results/presentation_v5_20261118/flow_stock_paths.parquet"
OUT = ROOT / "04_results/kospi_pair_policy_20261121"


def main():
    stock = pd.read_parquet(SOURCE)
    assert set(stock.k.unique()) >= {-40, -1, 0, 20}
    assert stock.code.nunique() == 19 and stock.event.nunique() == 3
    codes = sorted(stock.code.unique())
    mcap = pd.read_parquet(
        ROOT / "01_data/processed/calendar_main/data_v2_csv_20261029/market_cap.parquet",
        columns=codes,
    )
    mcap.index = pd.to_datetime(mcap.index)
    index_dates = pd.read_parquet(
        ROOT / "01_data/processed/calendar_main/benchmark_fnguide_2000_20261029/kospi200_price.parquet"
    ).index
    index_dates = pd.DatetimeIndex(index_dates).sort_values()
    cal = pd.read_csv(ROOT / "04_results/passive_demand_validation_20261110/event_calendar.csv")
    rows = []
    for event, group in stock.groupby("event"):
        t = pd.Timestamp(cal.loc[cal.event.eq(event), "rebalance_trade_date"].iloc[0])
        j = index_dates.get_loc(t)
        event_codes = sorted(group.code.unique())
        old_base = mcap.loc[index_dates[j - 81], event_codes]
        new_base = mcap.loc[index_dates[j - 41], event_codes]
        assert (old_base > 0).all() and (new_base > 0).all()
        for (actor, code), g in group.groupby(["actor", "code"]):
            g = g.set_index("k").sort_index()
            anchor = float(g.loc[-40, "cumulative_pct"])
            ratio = float(old_base[code] / new_base[code])
            for k, r in g.loc[-40:20].iterrows():
                rows.append(dict(event=event, actor=actor, code=code, k=k,
                                 date=r.date,
                                 cumulative_pct=(float(r.cumulative_pct) - anchor) * ratio))
    rebased = pd.DataFrame(rows)
    assert rebased[rebased.k.eq(-40)].cumulative_pct.abs().max() < 1e-10
    event_path = rebased.groupby(["event", "actor", "k"], as_index=False).agg(
        cumulative_pct=("cumulative_pct", "mean"), n_stocks=("code", "nunique")
    )
    mean_path = event_path.groupby(["actor", "k"], as_index=False).agg(
        cumulative_pct=("cumulative_pct", "mean"), n_events=("event", "nunique")
    )
    assert mean_path.n_events.eq(3).all()
    OUT.mkdir(parents=True, exist_ok=True)
    rebased.to_parquet(OUT / "flow_t40_stock_paths.parquet", index=False)
    event_path.to_csv(OUT / "flow_t40_event_paths.csv", index=False, encoding="utf-8-sig")
    mean_path.to_csv(OUT / "flow_t40_mean_path.csv", index=False, encoding="utf-8-sig")
    (OUT / "flow_t40_protocol.json").write_text(json.dumps({
        "cohort": "TIGER200 publicly observed new basket members, 19 stocks, 3 events",
        "window": "T-40 close anchored at zero through T+20 close",
        "denominator": "each stock market capitalization on T-41, fixed throughout window",
        "method": "For each stock, subtract original T-40 cumulative flow and rescale from T-81 market capitalization to T-41 market capitalization; then equal-weight stocks within each event and equal-weight 3 events",
        "limitation": "Descriptive actor flow, not identified passive-fund orders or matched counterparties",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(mean_path[mean_path.k.isin([-40, -1, 0, 20])].pivot(index="k", columns="actor", values="cumulative_pct"))


if __name__ == "__main__":
    main()
