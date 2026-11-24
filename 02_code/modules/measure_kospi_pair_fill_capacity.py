"""Replay the frozen OOS orders at different KRW budgets and audit event fills."""
from functools import lru_cache
import sys
import pandas as pd
import compare_kospi_trend_parameters as trend


def main():
    engine = trend.policy.engine
    px, tv, actions = trend.execution_inputs()
    schedule = pd.read_csv(trend.SOURCE / "all_forecasts.csv", parse_dates=["entry_date", "exit_date"])
    schedule = schedule[schedule.event.ge("2023")].copy()
    targets = pd.read_parquet(trend.OUT / "IS_selected_targets.parquet")
    groups = targets.groupby("case_id").code.agg(set)
    names = {cid: groups.get(cid, set()) for cid in schedule.case_id}
    exits = trend.exits_for(schedule, names, px.index)
    cols = sorted(set(targets.code) | {a["new"] for a in actions if a["new"]})
    px = px.loc["2022-09-01":].reindex(columns=cols)
    tv = tv.reindex_like(px)
    out = trend.ROOT / "04_results/kospi_pair_capacity_20261124"
    out.mkdir(exist_ok=True)
    rows = []
    @lru_cache(None)
    def run(a, b):
        pairs = [(i, v) for i, v in zip(("KOSPI200", "KOSPI100"), (a, b)) if v]
        ids = [i for i, _ in pairs]
        assert len({v for _, v in pairs}) == 1
        engine.BUDGET = pairs[0][1] * 1e8
        choices = pd.DataFrame({"index_id": ids, "model": trend.policy.MODEL})
        _, _, g, _, _, checks = engine.simulate(choices, schedule, schedule, targets, px, tv, actions,
            start="2023-01-01", end="2026-11-20", lead_map={i: trend.policy.LEADS[4] for i in ids},
            mark_open=True, early_exit_dates=exits)
        assert checks["max_limit_excess"] < .001 and checks["min_cash"] > -.001
        summary = {}
        for idx, part in g.groupby("index_id"):
            e = part.groupby("event")[["entry_notional", "initial_target_notional"]].sum()
            e = e[e.initial_target_notional.gt(0)].copy()
            e["fill"] = e.entry_notional / e.initial_target_notional
            avg = e.entry_notional.sum()/e.initial_target_notional.sum()
            summary[idx] = (e.fill.min(), avg, e.fill.idxmin())
            for event, r in e.iterrows():
                rows.append((a,b,idx,event,r.initial_target_notional,r.entry_notional,r.fill,avg))
        return summary
    pooled = "--pooled" in sys.argv
    def save():
        pd.DataFrame(rows, columns=["kospi200_eok","kospi100_eok","index","event","target_krw","filled_krw","fill","aggregate_fill"]).to_csv(out/("shared_equal_by_event.csv" if pooled else "capacity_by_event.csv"), index=False)
    for amt in [(100,0),(0,100),(100,100)]:
        print(amt,run(*amt),flush=True)
    save()
    def boundary(pair, metric):
        def passed(x):
            sizes = pair(x)
            summary = run(*sizes)
            if pooled:
                by_event = {}
                for a,b,idx,event,order,fill,_,_ in rows:
                    if (a,b) == sizes:
                        prev = by_event.get(event, (0.,0.))
                        by_event[event] = (prev[0]+order, prev[1]+fill)
                return min(fill/order for order,fill in by_event.values()) >= .8 - 1e-12
            return min(r[metric] for r in summary.values()) >= .8 - 1e-12
        low, high = 1, 100
        if not passed(low):
            print("infeasible at 1 eok",pair(low),metric,run(*pair(low)),flush=True)
            save()
            return 0
        while passed(high):
            low, high = high, 2*high
        while high-low>1:
            mid = (high+low)//2
            if passed(mid):
                low=mid
            else:
                high=mid
        print("boundary",pair(low),metric,run(*pair(low)),pair(high),run(*pair(high)),flush=True)
        save()
        return low
    left=lambda x:(x,0)
    right=lambda x:(0,x)
    equal=lambda x:(x,x)
    if pooled:
        boundary(equal,0)
        return
    boundary(left,0)
    boundary(right,0)
    boundary(equal,0)
    for p in (left,right,equal):
        boundary(p,1)
    save()


if __name__ == "__main__":
    main()
