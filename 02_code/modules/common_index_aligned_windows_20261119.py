"""Select on the longest reproducible standalone histories, then replay a shared OOS."""
import json

import numpy as np
import pandas as pd

import v18_long_replay as replay


def compromise(candidates):
    """Equal-weight min-max scores within one index, using IS metrics only."""
    frame = pd.DataFrame(candidates).set_index("n_splits")
    assert frame.phase.eq("IS").all() and set(frame.index) == {1, 2, 3, 4}
    metrics = frame[["sharpe", "mdd"]]
    assert np.isfinite(metrics).all().all() and metrics.mdd.le(0).all()
    spread = metrics.max() - metrics.min()
    normalized = ((metrics - metrics.min()) / spread.replace(0, np.nan)).fillna(0.5)
    scores = normalized.mean(axis=1)
    order = frame.assign(score=scores).reset_index().sort_values(
        ["score", "mdd", "n_splits"], ascending=[False, False, True])
    return int(order.iloc[0].n_splits), scores


def restore_early_signals(forecasts, targets, model):
    """Recover archived trend forecasts, including price tests without an outcome label."""
    assert model == "shrunk_trend__count"
    gate = pd.read_parquet(replay.R / "two_axis_literal_20261119/model_panel.parquet",
                           columns=["case_id", "code", "opportunity"])
    additions, weights = [], []
    for idx, frames in replay.FRAME_ROOT.items():
        if idx == "KRX300":
            continue
        archive = pd.read_csv(frames.parent / "targets.csv")
        archive = archive[archive.model.eq("shrunk_trend") & archive.event.lt("2023")
                          & ~archive.case_id.isin(forecasts.case_id)]
        if "train_end" in archive:
            archive = archive[archive.train_end.eq(replay.SPLIT[idx])]
        for cid, g in archive.groupby("case_id"):
            r = g.iloc[0]
            # A held A009840 has no observable 2003-04-16 price or settlement.
            # The earliest common executable KOSPI200 review is June 2004.
            if idx == "KOSPI200" and r.event == "2003-06":
                continue
            current_gate = gate[gate.case_id.eq(cid)]
            if len(current_gate):
                eligible = set(current_gate.loc[current_gate.opportunity, "code"])
            else:
                # Two early KOSPI200 reviews have forecasts but no verified realized list.
                assert idx == "KOSPI200" and r.event == "2005-06"
                frame = pd.read_pickle(frames / f"{cid}.pkl")["frame"]
                eligible = set(frame.index[frame.eligibility_state.eq("pass") & ~frame.cur
                                           & frame.special_exclusion.fillna("").eq("")])
            codes = sorted(set(g.code) & eligible)
            assert pd.Timestamp(r.information_date) < pd.Timestamp(r.entry_date)
            additions.append(dict(index_id=idx, case_id=cid, model=model, event=r.event, lead=r.lead,
                                  entry_date=pd.Timestamp(r.entry_date), exit_date=pd.Timestamp(r.exit_date),
                                  forecast_available=True, phase="IS"))
            weights.extend(dict(case_id=cid, code=c, model=model, weight=1/len(codes)) for c in codes)
    return pd.concat([forecasts, pd.DataFrame(additions)], ignore_index=True), pd.concat(
        [targets, pd.DataFrame(weights)], ignore_index=True)


def rescore_prior_window(source, destination, engine):
    archive = source / "aligned_compromise"
    paths = pd.read_parquet(archive / "daily.parquet").sort_values("date")
    prior = pd.read_csv(archive / "performance.csv")
    keys = ["index_id", "phase", "n_splits"]
    sharpes = paths.groupby(keys).apply(engine.cd91_sharpe, include_groups=False)
    prior["sharpe"] = sharpes.reindex(pd.MultiIndex.from_frame(prior[keys])).to_numpy()
    for idx, group in prior.groupby("index_id"):
        selected, scores = compromise(group[group.phase.eq("IS")].to_dict("records"))
        prior.loc[group.index, "aligned_IS_choice"] = group.n_splits.eq(selected)
        insample = group[group.phase.eq("IS")]
        prior.loc[insample.index, "IS_score"] = insample.n_splits.map(scores)
    prior.to_csv(destination / "prior_window_cd91.csv", index=False, encoding="utf-8-sig")


def main():
    source = replay.R / "common_index_engine_20261119"
    original_splits = json.loads((source / "selection.json").read_text())
    forecasts = pd.read_csv(source / "forecasts.csv", parse_dates=["entry_date", "exit_date"])
    targets = pd.read_csv(source / "targets.csv")
    reference = pd.read_csv(source / "performance.csv")
    model, = forecasts.model.unique()
    forecasts, targets = restore_early_signals(forecasts, targets, model)
    assert not forecasts.case_id.duplicated().any()
    resources = replay.load("02_code/modules/index_forecast_is_oos_20261117/evaluate.py", "aligned_resources")
    resources.write = lambda *args: None
    px, turnover, actions = replay.supplement(*resources.resources())
    engine = replay.load("02_code/modules/index_three_portfolio_20261117/engine_selected_checkpoint.py", "aligned_orders")
    engine.BUDGET = 10_000_000_000.0
    destination = source / "long_IS_compromise"
    destination.mkdir(exist_ok=True)
    rescore_prior_window(source, destination, engine)
    rows, daily, choices = [], [], {}
    for idx, old_n in original_splits.items():
        sleeve = pd.DataFrame([dict(index_id=idx, model=model)])
        selected = None
        for phase, start, end in [
            ("IS", "2004-01-01" if idx=="KOSPI200" else replay.START[idx], "2022-12-31"),
            ("OOS", "2023-01-01", "2026-06-30")
        ]:
            cases = forecasts[forecasts.index_id.eq(idx) & forecasts.event.between(start[:7], end[:7])]
            counts_by_event = cases.groupby("event").lead.nunique()
            incomplete = set(counts_by_event[counts_by_event < 4].index)
            assert incomplete == ({"2015-06","2017-06"} if idx=="KOSPI100" and phase=="IS" else set())
            phase_rows = []
            for n in (1, 2, 3, 4):
                leads = replay.LEADS[:n]
                metrics, path, lots, orders, _, checks = engine.simulate(
                    sleeve, cases, forecasts, targets, px, turnover, actions, start=start, end=end,
                    lead_map={idx: leads}, mark_open=phase == "IS",
                )
                assert checks["profit_matches_event_cashflows"] and len(lots) == cases.lead.isin(leads).sum()
                assert phase == "IS" or checks["all_events_closed"]
                result = metrics.set_index("index_id").loc[idx].to_dict()
                if phase == "OOS" or idx == "KRX300":
                    old_phase = "OOS7" if phase == "OOS" else "IS"
                    old = reference.query("index_id == @idx and n_splits == @n and phase == @old_phase").iloc[0]
                    for field in ("cagr", "mdd", "profit_krw"):
                        assert np.isclose(old[field], result[field]), (idx, phase, n, field)
                    assert np.isclose(old.sharpe, result["sharpe"] if "sharpe_zero_rf" in old
                                      else result["sharpe_zero_rf"])
                open_lots = lots[lots.actual_exit.isna()]
                counts = orders[orders.side.eq("buy")].groupby("event").code.nunique().reindex(cases.event.unique(), fill_value=0)
                daily.append(path[path.index_id.eq(idx)].assign(phase=phase, n_splits=n))
                phase_rows.append(dict(result, index_id=idx, phase=phase, n_splits=n,
                                       first_review=cases.event.min(), last_review=cases.event.max(),
                                       missing_tranches=n*cases.event.nunique()-len(lots),
                                       mean_names=counts.mean(), min_names=counts.min(), max_names=counts.max(),
                                       open_tranches_at_end=len(open_lots),
                                       open_value_krw=open_lots.get("open_market_value", pd.Series(dtype=float)).sum(),
                                       reviews=cases.event.nunique(), original_choice=n == old_n))
            if phase == "IS":
                selected, scores = compromise(phase_rows)
                choices[idx] = selected
            rows.extend(dict(row, IS_score=scores[row["n_splits"]] if phase=="IS" else np.nan,
                             aligned_IS_choice=row["n_splits"] == selected) for row in phase_rows)
    output = pd.DataFrame(rows)
    output.to_csv(destination / "performance.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily, ignore_index=True).to_parquet(destination / "daily.parquet", index=False)
    (destination / "selection.json").write_text(json.dumps(choices, indent=2))
    sleeves = pd.DataFrame([dict(index_id=idx, model=model) for idx in choices])
    mapping = {idx: replay.LEADS[:n] for idx, n in choices.items()}
    for phase, start, end in [("IS", "2018-07-01", "2022-12-31"), ("OOS", "2023-01-01", "2026-06-30")]:
        schedule = forecasts[forecasts.event.between(start[:7], end[:7])]
        metrics, path, lots, orders, holdings, checks = engine.simulate(
            sleeves, schedule, forecasts, targets, px, turnover, actions, start=start, end=end,
            lead_map=mapping, mark_open=phase == "IS")
        assert checks["profit_matches_event_cashflows"] and (phase=="IS" or checks["all_events_closed"])
        used = orders[orders.side.isin(["buy","sell"])].groupby(["date","code"]).agg(
            actual=("notional","sum"), cap=("limit","min"))
        assert (used.actual <= used.cap + .01).all() and path.cash.min() >= -.01
        if phase == "IS":
            path.to_parquet(destination / "portfolio_IS_daily.parquet", index=False)
        else:
            metrics.to_csv(destination / "portfolio.csv", index=False)
            for name, frame in [("daily",path),("tranches",lots),("orders",orders),("holdings",holdings)]:
                frame.to_parquet(destination / f"portfolio_{name}.parquet", index=False)
    cols = ["index_id", "phase", "reviews", "n_splits", "total_return", "cagr", "sharpe", "mdd"]
    print(output[output.aligned_IS_choice | output.original_choice][cols].round(5).to_string(index=False))
    print(output.query("phase=='IS'").pivot(index="index_id", columns="n_splits", values="sharpe").round(4))


if __name__ == "__main__":
    main()
