"""Select trend parameters on standalone IS accounts and replay the frozen OOS."""

from itertools import product
import json

import numpy as np
import pandas as pd

import kospi_pair_policy_20261121 as policy
import refresh_pair_fnguide_20261123 as latest
from kospi_pair_h2_information_policy_20261122 import announcement_state

ROOT = policy.ROOT
REPLAY = policy.replay
SOURCE = latest.OUT
OUT = ROOT / "04_results/kospi_pair_trend_parameter_selection"
BASE = (0.25, 0.4)
GRID = [(0.0, 0.4), *product((0.10, 0.25, 0.50, 0.75, 1.0), (0.2, 0.3, 0.4, 0.6, np.inf))]
IS_START = {"KOSPI200": "2004-01-01", "KOSPI100": "2006-01-01"}
METRICS = ["ending_equity_krw", "total_return", "cagr", "volatility", "sharpe", "sharpe_zero_rf", "mdd"]


def historical_selectors(schedule):
    panel = pd.read_parquet(ROOT / "04_results/two_axis_literal_20261119/model_panel.parquet")
    groups = dict(tuple(panel[panel.case_id.isin(schedule.case_id)].groupby("case_id")))
    models = None
    for case in schedule.itertuples():
        folder = REPLAY.FRAME_ROOT[case.index_id]
        path = folder / f"{case.case_id}.pkl"
        if not path.exists():
            path = ROOT / "04_results/two_axis_selection_20261118/frames" / path.name
        obj = pd.read_pickle(path)
        frame = obj["frame"]
        group = groups.get(case.case_id)
        source = obj["panel"] if group is None else group
        meta = source.iloc[0]
        rule = REPLAY.vintage.vintage(case.index_id, case.event, meta.information_date, meta.effective_date)
        known = group is not None and (
            rule.cap_coverage is not None if case.index_id == "KOSPI200" else rule.incumbent_buffer is not None
        )
        if known:
            prepared = REPLAY.E.prepare_selector(frame, REPLAY.E.profile_for(case.event, asof=meta.information_date))
            z = source.set_index("code").reindex(frame.index)
            eligible = set(source.loc[source.opportunity, "code"])
            current = z.current_member.fillna(False).to_numpy(bool)
            def select(cap):
                return dated_selection(case.index_id, prepared, frame, cap, current)
            fallback = frame.mc
            share_ratio = 1.0
        else:
            if models is None:
                from dataclasses import replace
                models = REPLAY.P.PaperModels()
                models.config = replace(models.config, allow_underfilled_k100=True)
                # The dated-replay module already loads this reviewed V2 revision.
                # Use it in this process only and require exact incumbent targets.
                REPLAY.P.v2_engine = lambda: REPLAY.E
                extension = models.engines.k200.__wrapped__.__globals__["build_extension_snapshot"].__globals__
                extension["v2_engine"] = lambda: REPLAY.E
            select, _ = models._selector(case.index_id, case.event, meta.information_date, frame, obj["baseline"])
            z = obj["panel"].set_index("code").reindex(frame.index)
            fallback = z.projected_cap
            share_ratio = 1.0
            if case.index_id == "KOSPI100":
                # The archived selector allows the corporate-input extension in its forecast.
                # Without an archived as-of ratio the multiplicative adjustment is neutral.
                share_ratio = z.get("known_future_share_ratio", 1.0)
            eligible = (set(group.loc[group.opportunity, "code"]) if group is not None else
                        set(frame.index[frame.eligibility_state.eq("pass") & ~frame.cur &
                                        frame.special_exclusion.fillna("").eq("")]))
        yield case.case_id, select, z, fallback, share_ratio, eligible
        if models is not None:
            models.engines.k200.cache_clear()


def dated_selection(index_id, prepared, frame, cap, current):
    selected = prepared.select(np.asarray(cap), frame.tv.to_numpy())
    if index_id == "KOSPI200":
        return set(frame.index[prepared.additions_mask(selected)])
    pool = np.flatnonzero(selected & frame.eligibility_state.eq("pass").to_numpy() &
                          frame.tv.gt(0).to_numpy() & frame.grp.notna().to_numpy() & (cap > 0))
    order = pool[np.argsort(-np.asarray(cap)[pool], kind="stable")]
    incumbents = order[:120][current[order[:120]]]
    additions = order[:80][~current[order[:80]]]
    chosen = np.r_[incumbents, additions][:100]
    rest = order[current[order] & ~np.isin(order, chosen)]
    chosen = np.r_[chosen, rest[:100-len(chosen)]]
    return set(frame.index[chosen[~current[chosen]]])


def forecast_sets(schedule, grid):
    result = {params: {} for params in grid}
    historic = schedule[schedule.event.ne("2026-12")]
    for cid, select, z, fallback, shares, eligible in historical_selectors(historic):
        trend = (np.log1p(z.momentum20.clip(lower=-0.99)) * z.remaining / 40).fillna(0)
        for alpha, limit in grid:
            cap = (z.past_sum + z.remaining * z.current_cap * shares *
                   np.exp((alpha * trend).clip(-limit, limit))) / z.full_days
            cap = cap.where(np.isfinite(cap) & cap.gt(0) & z.eligibility.eq("pass"), fallback)
            result[(alpha, limit)][cid] = set(select(cap)) & eligible
    tail = schedule[schedule.event.eq("2026-12")]
    if tail.empty:
        return result
    k100_roster = pd.read_parquet(latest.DATA / "KOSPI100.parquet")
    dates = pd.read_csv(ROOT / "01_data/processed/kospi_pair_nonfnguide_20261122/H2_signal_dates.csv",
                        parse_dates=["information_date"]).set_index("lead").information_date
    profile = next(p for p in REPLAY.E.PROFILES if p.name == "gics10_2024_underfill")
    for lead, cases in tail.groupby("lead"):
        frame = pd.read_parquet(SOURCE / f"frame_L{lead}.parquet")
        prepared = REPLAY.E.prepare_selector(frame, profile)
        current = frame.index.isin(k100_roster.columns[k100_roster.loc[dates[lead]]])
        eligible = set(frame.index[frame.eligibility_state.eq("pass") & frame.a_exclusion_reason.eq("")])
        trend = (np.log1p(frame.momentum20.clip(lower=-0.99)) * frame.remaining_sessions / 40).fillna(0)
        for alpha, limit in grid:
            cap = (frame.mc_sum_observed + frame.remaining_sessions * frame.mc_current *
                   np.exp((alpha * trend).clip(-limit, limit))) / frame.review_sessions
            cap = cap.where(np.isfinite(cap) & cap.gt(0) & frame.eligibility_state.eq("pass"), frame.mc)
            for case in cases.itertuples():
                names = dated_selection(case.index_id, prepared, frame, cap, current)
                result[(alpha, limit)][case.case_id] = names & eligible
    return result


def execution_inputs():
    import corporate_actions
    corporate_actions.dump = lambda *args: None
    resources = REPLAY.load("02_code/modules/index_forecast_is_oos_20261117/evaluate.py", "trend_parameter_resources")
    resources.write = lambda *args: None
    px, tv, actions = REPLAY.supplement(*resources.resources())
    raw = pd.read_parquet(ROOT / "01_data/processed/kospi_pair_nonfnguide_20261122/KRX_raw_market_observations.parquet",
                          columns=["BAS_DD", "ISU_CD"])
    raw["date"] = pd.to_datetime(raw.BAS_DD, format="%Y%m%d")
    raw["code"] = "A" + raw.ISU_CD.astype(str).str.zfill(6)
    observed = raw.assign(present=True).pivot(index="date", columns="code", values="present").notna()
    observed = observed.loc["2026-07-01":"2026-11-20"]
    price = pd.read_parquet(SOURCE / "close.parquet").reindex_like(observed).where(observed)
    turnover = pd.read_parquet(SOURCE / "turnover.parquet").reindex_like(observed).where(observed)
    px = pd.concat([px.loc[:"2026-06-30"], price]).sort_index()
    tv = pd.concat([tv.loc[:"2026-06-30"], turnover]).sort_index()
    rates = pd.read_csv(ROOT / "01_data/processed/fnguide/CD_91.csv")
    rates = pd.Series(pd.to_numeric(rates.iloc[:, 1]).to_numpy() / 100,
                      index=pd.to_datetime(rates.iloc[:, 0])).sort_index()
    rates = rates.reindex(pd.date_range("1999-12-01", "2026-11-20"), method="ffill")
    policy.engine.cd91_rates = lambda: rates
    policy.engine.BUDGET = 1e10
    policy.engine.FEE = 0.0025
    return px, tv, actions


def targets_from(names):
    return pd.DataFrame([
        (cid, code, policy.MODEL, 1 / len(codes))
        for cid, codes in sorted(names.items()) for code in sorted(codes)
    ], columns=["case_id", "code", "model", "weight"])


def exits_for(schedule, names, sessions):
    early = schedule[schedule.event.ne("2026-12")]
    exits, _ = policy.announcement_exit_map(early, targets_from({cid: names[cid] for cid in early.case_id}), sessions)
    for row in schedule[schedule.event.eq("2026-12")].itertuples():
        for code in names[row.case_id]:
            if announcement_state(row.index_id, code, "2026-11-20") == "confirmed_out":
                exits[(row.case_id, code)] = pd.Timestamp("2026-11-19")
    return exits


def assert_targets(actual, expected):
    mismatches = [(cid, sorted(expected[cid]-codes), sorted(codes-expected[cid]))
                  for cid, codes in actual.items() if codes != expected[cid]]
    if mismatches:
        raise ValueError(f"Baseline forecast mismatch ({len(mismatches)}): {mismatches}")


def compare_positions():
    names = pd.read_pickle(ROOT / "01_data/processed/signal_a_cache.pkl")["names"]
    snapshots, count_rows, event_rows = [], [], []
    for account in (*policy.IDS, "COMBINED"):
        holdings = {
            variant: pd.read_parquet(OUT / f"{variant}_{account}_holdings.parquet")
            for variant in ("baseline", "IS_selected")
        }
        orders = {
            variant: pd.read_parquet(OUT / f"{variant}_{account}_orders.parquet").query("side == 'buy'")
            for variant in holdings
        }
        for variant, frame in holdings.items():
            last = frame[frame.date.eq(pd.Timestamp("2026-11-20"))].copy()
            last["account"] = account
            last["variant"] = variant
            last["name"] = last.code.map(names)
            snapshots.append(last)
        keys = ["date", "index_id", "code"]
        before = holdings["baseline"].set_index(keys).market_value
        after = holdings["IS_selected"].set_index(keys).market_value
        values = pd.concat([before.rename("before"), after.rename("after")], axis=1)
        count_rows.append({"account": account,
                           "new_holding_rows": int((values.before.isna() & values.after.notna()).sum()),
                           "removed_holding_rows": int((values.before.notna() & values.after.isna()).sum()),
                           "dates_with_different_names": values[values.isna().any(axis=1)].index.get_level_values("date").nunique(),
                           "dates_with_different_positions": values[~np.isclose(values.before.fillna(0), values.after.fillna(0),
                                                                                 rtol=1e-10, atol=.01)].index.get_level_values("date").nunique()})
        events = sorted(set(orders["baseline"].event) | set(orders["IS_selected"].event))
        for event in events:
            b = set(orders["baseline"].loc[orders["baseline"].event.eq(event), "code"])
            a = set(orders["IS_selected"].loc[orders["IS_selected"].event.eq(event), "code"])
            event_rows.append({"account": account, "event": event, "baseline_bought": len(b),
                               "selected_bought": len(a), "added": "; ".join(f"{names.get(c, c)} ({c})" for c in sorted(a-b)),
                               "removed": "; ".join(f"{names.get(c, c)} ({c})" for c in sorted(b-a))})
    pd.concat(snapshots).to_csv(OUT / "OOS_final_holdings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(count_rows).to_csv(OUT / "OOS_holding_changes.csv", index=False)
    pd.DataFrame(event_rows).to_csv(OUT / "OOS_event_buy_changes.csv", index=False, encoding="utf-8-sig")


def validate_tail_prices():
    held = pd.read_parquet(OUT / "IS_selected_COMBINED_holdings.parquet")
    held = held[held.date.ge("2026-07-01")][["date", "code"]].drop_duplicates()
    fills = pd.read_parquet(OUT / "IS_selected_COMBINED_orders.parquet")
    opened = fills[fills.date.ge("2026-07-01")][["date", "code"]]
    observed = pd.concat([held, opened]).drop_duplicates()
    raw = pd.read_parquet(ROOT / "01_data/processed/kospi_pair_nonfnguide_20261122/KRX_raw_market_observations.parquet",
                          columns=["BAS_DD", "ISU_CD", "TDD_CLSPRC", "CMPPREVDD_PRC"])
    raw["code"] = "A" + raw.ISU_CD.astype(str).str.zfill(6)
    raw = raw[raw.code.isin(observed.code)].copy()
    raw["date"] = pd.to_datetime(raw.BAS_DD, format="%Y%m%d")
    raw = raw.sort_values(["code", "date"])
    price = pd.to_numeric(raw.TDD_CLSPRC.astype(str).str.replace(",", ""))
    change = pd.to_numeric(raw.CMPPREVDD_PRC.astype(str).str.replace(",", ""))
    raw["reference_gap"] = (price - change) / price.groupby(raw.code).shift() - 1
    price = pd.read_parquet(SOURCE / "close.parquet")
    adjusted = pd.read_parquet(SOURCE / "adjusted.parquet")
    ratio = adjusted.div(price)
    distance = ratio.div(ratio.shift()).sub(1).stack().rename("adjustment_change")
    distance.index.names = ["date", "code"]
    checks = observed.merge(raw[["date", "code", "reference_gap"]], on=["date", "code"], how="left")
    checks = checks.merge(distance.reset_index(), on=["date", "code"], how="left")
    assert not checks[["reference_gap", "adjustment_change"]].abs().gt(.01).any().any(), (
        checks[checks[["reference_gap", "adjustment_change"]].abs().gt(.01).any(axis=1)])
    print("H2 price adjustments checked", len(checks), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    schedule = pd.read_csv(SOURCE / "all_forecasts.csv", parse_dates=["entry_date", "exit_date"])
    expected = pd.read_csv(SOURCE / "all_targets.csv").groupby("case_id").code.agg(set).to_dict()
    expected = {cid: expected.get(cid, set()) for cid in schedule.case_id}
    train = schedule[schedule.event.lt("2023")]
    test = schedule[schedule.event.ge("2023")]
    assert schedule.case_id.is_unique
    train_sets = forecast_sets(train, GRID)
    assert_targets(train_sets[BASE], expected)
    print("IS baseline forecasts reproduced", flush=True)
    px, tv, actions = execution_inputs()
    def simulate(cases, names, ids, start, end, exits):
        choices = pd.DataFrame({"index_id": ids, "model": policy.MODEL})
        m, d, _, o, h, audit = policy.engine.simulate(
            choices, cases, cases, targets_from(names), px, tv, actions, start=start, end=end,
            lead_map={idx: policy.LEADS[4] for idx in ids}, mark_open=True, early_exit_dates=exits)
        assert audit["max_limit_excess"] <= .001 and audit["min_cash"] >= -.001
        assert audit["profit_matches_event_cashflows"]
        return m, d, o, h

    # Announcement decisions depend on case/code, not model parameters.
    train_union = {cid: set.union(*(d[cid] for d in train_sets.values())) for cid in train.case_id}
    train_exits = exits_for(train, train_union, px.index)
    grid_rows = []
    winners = {}
    old_standalone = pd.read_csv(SOURCE / "standalone_splits.csv")
    for idx in policy.IDS:
        cases = train[train.index_id.eq(idx)]
        memo = {}
        for params, pool in train_sets.items():
            names = {cid: pool[cid] for cid in cases.case_id}
            key = tuple((cid, tuple(sorted(codes))) for cid, codes in names.items())
            if key not in memo:
                m, *_ = simulate(cases, names, [idx], IS_START[idx], "2022-12-31", train_exits)
                memo[key] = m.iloc[0][METRICS].to_dict()
            if params == BASE:
                benchmark = old_standalone.query("index_id == @idx and phase == 'IS' and n_splits == 4").iloc[0]
                np.testing.assert_allclose(list(memo[key].values()), benchmark[METRICS].astype(float), rtol=1e-10, atol=.01)
            grid_rows.append(dict(index_id=idx, alpha=params[0], limit=params[1], **memo[key]))
            print(idx, params, round(memo[key]["sharpe"], 5), flush=True)
        print(idx, "distinct IS portfolios", len(memo), flush=True)
    grid_result = pd.DataFrame(grid_rows)
    grid_result.to_csv(OUT / "IS_grid.csv", index=False)
    for idx, group in grid_result.groupby("index_id"):
        # Highest IS Sharpe, then MDD, then closest to the incumbent settings.
        best = group.assign(distance=(group.alpha - BASE[0]).abs() +
                            (group.limit - BASE[1]).abs()).sort_values(
            ["sharpe", "mdd", "distance", "alpha", "limit"], ascending=[False, False, True, True, True]).iloc[0]
        winners[idx] = (float(best.alpha), float(best.limit))
    (OUT / "IS_selection.json").write_text(json.dumps({
        "criterion": "standalone IS CD91-adjusted daily Sharpe; MDD then distance to baseline break ties",
        "IS_end": "2022-12-31", "OOS": ["2023-01-02", "2026-11-20"],
        "parameters": {k: [v[0], v[1] if np.isfinite(v[1]) else "unbounded"] for k, v in winners.items()}
    }, indent=2), encoding="utf-8")
    print("IS frozen", winners, flush=True)

    test_sets = forecast_sets(test, list(dict.fromkeys([BASE, *winners.values()])))
    assert_targets(test_sets[BASE], expected)
    meta = test.set_index("case_id")
    optimized = {cid: test_sets[winners[meta.at[cid, "index_id"]]][cid] for cid in test.case_id}
    union = {cid: optimized[cid] | test_sets[BASE][cid] for cid in test.case_id}
    test_exits = exits_for(test, union, px.index)
    old_combined = pd.read_csv(SOURCE / "metrics.csv")
    comparisons = []
    for tag, names in [("baseline", test_sets[BASE]), ("IS_selected", optimized)]:
        targets_from(names).to_parquet(OUT / f"{tag}_targets.parquet", index=False)
        for ids in [[idx] for idx in policy.IDS] + [list(policy.IDS)]:
            cases = test[test.index_id.isin(ids)]
            key = ids[0] if len(ids) == 1 else "COMBINED"
            m, d, o, h = simulate(cases, names, ids, "2023-01-01", "2026-11-20", test_exits)
            summary = m.loc[m.index_id.eq(key)].iloc[0]
            if tag == "baseline":
                ref = old_combined if key == "COMBINED" else old_standalone.query("n_splits == 4")
                benchmark = ref.query("index_id == @key and phase == 'OOS'").iloc[0]
                np.testing.assert_allclose(summary[METRICS].astype(float), benchmark[METRICS].astype(float),
                                           rtol=1e-10, atol=.01)
            comparisons.append(summary.to_dict() | {"variant": tag})
            for label, frame in [("daily", d), ("orders", o), ("holdings", h)]:
                frame.to_parquet(OUT / f"{tag}_{key}_{label}.parquet", index=False)
            print(tag, key, summary[METRICS].to_dict(), flush=True)
    pd.DataFrame(comparisons).to_csv(OUT / "OOS_metrics.csv", index=False)
    changes = []
    for cid, names in optimized.items():
        before = test_sets[BASE][cid]
        for code in sorted(before ^ names):
            changes.append(dict(case_id=cid, index_id=meta.at[cid, "index_id"], event=meta.at[cid, "event"],
                                lead=meta.at[cid, "lead"], code=code, change="added" if code in names else "removed"))
    pd.DataFrame(changes, columns=["case_id", "index_id", "event", "lead", "code", "change"]).to_csv(
        OUT / "OOS_signal_changes.csv", index=False)
    compare_positions()
    validate_tail_prices()


if __name__ == "__main__":
    main()
