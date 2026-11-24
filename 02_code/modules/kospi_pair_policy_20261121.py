"""Primary KOSPI200/KOSPI100 account: sell announced misses from next close.

The published list is used only after its information date. Confirmed entrants
continue to the scheduled index-rebalance close. Buy and sell orders share the
same per-stock daily participation cap in the existing execution engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common_index_aligned_windows_20261119 as windows
import v18_long_replay as replay
from compare_krx300_engine_swap import holding_period_sharpes
from index_event_cash_comparison import match_fills
from index_three_portfolio_20261117 import engine_selected_checkpoint as engine

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "04_results/common_index_engine_20261119"
OUT = ROOT / "04_results/kospi_pair_policy_20261121"
IDS = ("KOSPI200", "KOSPI100")
MODEL = "shrunk_trend__count"
LEADS = {1: [80], 2: [80, 60], 3: [80, 60, 40], 4: [80, 60, 40, 20]}


def announcement_exit_map(forecasts, targets, sessions):
    """Freeze the first sale date and the available published outcome by event."""
    publication = pd.read_csv(
        ROOT / "01_data/processed/index_data_extension_20261117/kospi200_dated_membership_snapshots.csv"
    )
    publication["known_from"] = pd.to_datetime(publication.known_from, errors="coerce", format="mixed")
    publication["source_publication_date"] = pd.to_datetime(
        publication.source_publication_date, errors="coerce", format="mixed"
    )
    # 2017-06/2021-06 also have a later final revision. Use the initial
    # announcement vintage; never use a final revision to trade earlier.
    publication = publication.sort_values(["event", "known_from"])
    first_vintage = publication.groupby("event").known_from.transform("min")
    publication = publication[publication.known_from.eq(first_vintage)].copy()
    assert publication.groupby("event").code.nunique().eq(200).all()
    rosters200 = {ev: set(g.code) for ev, g in publication.groupby("event")}
    known200 = publication.groupby("event").known_from.first()
    announced200 = publication.groupby("event").source_publication_date.first()
    certified200 = publication.groupby("event").full_snapshot_certified.all()

    # The distinct KOSPI100 announcement date is incomplete. Where absent,
    # use the KOSPI200 announcement of the same regular review as a timing
    # assumption. Outcomes come from KOSPI100's observed effective membership.
    schedule100 = pd.read_csv(
        ROOT / "04_results/k100_long_20261117/case_schedule.csv",
        parse_dates=["announcement_date"],
    )
    announcement100 = schedule100.groupby("event").announcement_date.first()
    roster100 = pd.read_parquet(
        ROOT / "01_data/processed/index_data_extension_20261117/kospi100_membership_daily.parquet"
    )
    roster100 = roster100.reindex(columns=sorted(set(targets.code) & set(roster100.columns)))
    sessions = pd.DatetimeIndex(sessions)
    observations, exits = [], {}
    filtered = targets[targets.model.eq(MODEL)].merge(
        forecasts[["case_id", "index_id", "event", "entry_date", "exit_date"]],
        on="case_id", validate="many_to_one"
    )
    for row in filtered.itertuples():
        event = row.event
        if event not in rosters200:
            raise ValueError(f"No dated KOSPI200 published roster for {event}")
        if row.index_id == "KOSPI200":
            first_sale = sessions[sessions >= known200[event]][0]
            included = row.code in rosters200[event]
            date_basis = "official_KOSPI200_first_vintage"
            outcome_basis = "official_KOSPI200_first_vintage"
            certified = bool(certified200[event])
        else:
            own_date = announcement100.get(event, pd.NaT)
            proxy = pd.isna(own_date)
            announce = announced200[event] if proxy else own_date
            first_sale = sessions[sessions > announce][0]
            first_sale = max(first_sale, sessions[sessions >= known200[event]][0]) if proxy else first_sale
            effective_session = roster100.index[roster100.index > row.exit_date][0]
            included = bool(roster100.at[effective_session, row.code]) if row.code in roster100 else False
            date_basis = "same_review_KOSPI200_date_proxy" if proxy else "KOSPI100_dated_record"
            outcome_basis = "KOSPI100_effective_membership_observed_later"
            certified = False
        assert row.entry_date < first_sale <= row.exit_date, (row.case_id, row.code, first_sale)
        if not included:
            exits[(row.case_id, row.code)] = first_sale
        observations.append(dict(case_id=row.case_id, index_id=row.index_id, event=event,
                                 code=row.code, included_at_announcement=included,
                                 first_sale_date=first_sale, date_basis=date_basis,
                                 outcome_basis=outcome_basis, source_certified=certified))
    evidence = pd.DataFrame(observations)
    assert not evidence.duplicated(["case_id", "code"]).any()
    return exits, evidence


def event_results(orders, daily):
    joint = daily[daily.index_id.eq("COMBINED")].set_index("date")
    dates = pd.DatetimeIndex(joint.index)
    rf = engine.cd91_rates().loc[dates[:-1]].to_numpy()
    clock = pd.Series(np.r_[0., np.cumsum(rf * (dates[1:] - dates[:-1]).days / 365)], index=dates)
    events = match_fills(orders, clock).groupby(["index_id", "event"])[["pnl", "cost", "interest"]].sum()
    events = pd.concat([events.reset_index(),
                        events.groupby("event").sum().reset_index().assign(index_id="COMBINED")])
    events["excess_return"] = (events.pnl - events.interest) / events.cost
    return events


def replay_account(choices, schedule, targets, px, tv, actions, exits, phase, start, end, splits, prefix):
    allowed = {idx: LEADS[splits] for idx in choices.index_id}
    subset = schedule[schedule.event.between(start[:7], end[:7])]
    m, daily, lots, orders, holdings, audit = engine.simulate(
        choices, subset, schedule, targets, px, tv, actions, start=start, end=end,
        lead_map=allowed, mark_open=phase == "IS", early_exit_dates=exits,
    )
    assert audit["profit_matches_event_cashflows"] and audit["max_limit_excess"] <= .001
    assert audit["min_cash"] >= -.001
    assert phase == "IS" or audit["all_events_closed"]
    used = orders[orders.side.isin(["buy", "sell"])].groupby(["date", "code"]).agg(
        gross=("notional", "sum"), cap=("limit", "min"))
    assert (used.gross <= used.cap + .01).all()
    if exits:
        early = orders[orders.side.eq("sell")].merge(
            pd.DataFrame([{"index_id": cid.split("_")[0], "event": cid.split("_")[1],
                           "lead": int(cid.rsplit("L", 1)[1]), "code": code, "first_sale": date}
                          for (cid, code), date in exits.items()]),
            on=["index_id", "event", "lead", "code"], how="inner")
        assert (early.date >= early.first_sale).all()
    if prefix:
        for key, data in [("daily", daily), ("lots", lots), ("orders", orders), ("holdings", holdings)]:
            data.to_parquet(OUT / f"{prefix}_{key}.parquet", index=False)
        if phase == "OOS":
            event_results(orders, daily).to_csv(OUT / f"{prefix}_events.csv", index=False)
        pd.DataFrame([holding_period_sharpes(g, idx, phase, m.loc[m.index_id.eq(idx), "sharpe"].item())
                      for idx, g in daily.groupby("index_id")]).to_csv(OUT / f"{prefix}_sharpe.csv", index=False)
    return m, daily, orders


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts, targets = windows.restore_early_signals(
        pd.read_csv(SRC / "forecasts.csv", parse_dates=["entry_date", "exit_date"]),
        pd.read_csv(SRC / "targets.csv"), MODEL)
    forecasts = forecasts[forecasts.index_id.isin(IDS) & forecasts.model.eq(MODEL)]
    targets = targets[targets.case_id.isin(forecasts.case_id) & targets.model.eq(MODEL)]
    resources = replay.load("02_code/modules/index_forecast_is_oos_20261117/evaluate.py", "policy_resources")
    resources.write = lambda *args: None
    px, tv, actions = replay.supplement(*resources.resources())
    engine.BUDGET = 1e10
    exits, evidence = announcement_exit_map(forecasts, targets, px.index)
    evidence.to_csv(OUT / "announcement_decisions.csv", index=False, encoding="utf-8-sig")
    comparisons, main_rows = [], []
    for idx, start in [("KOSPI200", "2004-01-01"), ("KOSPI100", "2006-01-01")]:
        choice = pd.DataFrame({"index_id": [idx], "model": [MODEL]})
        for phase, s, e in [("IS", start, "2022-12-31"), ("OOS", "2023-01-01", "2026-06-30")]:
            for splits in LEADS:
                print("simulate", idx, phase, splits, flush=True)
                prefix = f"{idx}_{phase}" if splits == 4 else None
                m, _, _ = replay_account(choice, forecasts, targets, px, tv, actions,
                                         exits, phase, s, e, splits, prefix)
                comparisons.append(m[m.index_id.eq(idx)].assign(phase=phase, n_splits=splits))
    pd.concat(comparisons).to_csv(OUT / "standalone_splits.csv", index=False, encoding="utf-8-sig")
    choice = pd.DataFrame({"index_id": list(IDS), "model": [MODEL] * len(IDS)})
    for phase, s, e in [("IS", "2006-01-01", "2022-12-31"),
                        ("OOS", "2023-01-01", "2026-06-30")]:
        print("simulate COMBINED", phase, flush=True)
        m, daily, orders = replay_account(choice, forecasts, targets, px, tv, actions,
                                          exits, phase, s, e, 4, phase)
        main_rows.append(m.assign(phase=phase))
    results = pd.concat(main_rows)
    results.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "protocol.json").write_text(json.dumps({
        "primary_rule": "confirmed entrants sell from rebalance close; misses sell from first close after announcement, subject to same daily turnover cap",
        "budget_each_krw": 10_000_000_000,
        "source_200": "dated KRX complete 200-stock roster, first public vintage only",
        "source_100": "observed daily membership at effective date",
        "KOSPI100_publication_gap": "when KOSPI100 date absent, KOSPI200 same-review publication date is a timing proxy",
        "KOSPI200_pending_events": ["2017-06", "2021-06"],
        "selection": "four entries frozen from original IS decision; all 1-4 entry sensitivity shown under new exit rule",
        "fee_each_side": engine.FEE,
        "daily_participation": engine.PARTICIPATION,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results[["index_id", "phase", "total_return", "cagr", "sharpe", "mdd"]].to_string(index=False))


if __name__ == "__main__":
    main()
