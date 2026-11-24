"""Point-in-time index methodology adapter for historical forecast frames.

The dated rule is selected by the target review date.  Inputs may only be
observed by the signal date.  This module separates a reproducible mechanical
selection from KRX discretionary exceptions and unavailable historical parents.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import importlib.util
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TMI = ROOT / "01_data/processed/index_membership_20261119"


@dataclass(frozen=True)
class Vintage:
    index_id: str
    event: str
    review_date: pd.Timestamp
    signal_date: pd.Timestamp
    industry: str
    parent: str
    cap_coverage: float | None
    liquidity_fraction: float | None
    incumbent_buffer: float | None
    entrant_buffer: float | None
    source: str
    status: str


def vintage(index_id, event, signal_date, review_date=None):
    """Resolve rules effective at a review; reject rules not yet public at signal."""
    s = pd.Timestamp(signal_date)
    r = pd.Timestamp(review_date if review_date is not None else event + "-01")
    if index_id == "KOSPI200":
        if r >= pd.Timestamp("2026-11-02"):
            if s < pd.Timestamp("2026-09-29"):
                return Vintage(index_id,event,r,s,"KRICS9","KOSPI",None,None,None,None,
                               "KRX 1120", "rule_not_known_at_signal")
            return Vintage(index_id,event,r,s,"KRICS9","KOSPI",.85,.85,1.10,.90,
                           "지수_방법론/KOSPI+200+지수+방법론( 최신).pdf; KRX 1120",
                           "rule_known;dated_KRICS_input_required")
        if r >= pd.Timestamp("2020-06-01"):
            if s < pd.Timestamp("2019-12-13"):
                return Vintage(index_id,event,r,s,"GICS10","KOSPI",None,None,None,None,
                               "KRX 643", "rule_not_known_at_signal")
            if r >= pd.Timestamp("2024-02-01") and s < pd.Timestamp("2023-12-23"):
                return Vintage(index_id,event,r,s,"GICS10","KOSPI",None,None,None,None,
                               "KRX 916", "rule_not_known_at_signal")
            ind = "GICS10" if r >= pd.Timestamp("2018-10-31") else "GICS9"
            return Vintage(index_id,event,r,s,ind,"KOSPI",.85,.85,1.10,.90,
                           "지수_방법론/KOSPI+200+지수+방법론(2411)_국문.pdf; KRX 643",
                           "rule_known;eligibility_case_review_pending")
        if r >= pd.Timestamp("2017-06-09"):
            return Vintage(index_id,event,r,s,"GICS9" if r < pd.Timestamp("2018-10-31") else "GICS10",
                           "KOSPI",.80,.85,1.10,.90,"KRX 506; 2021/2024 methodology history",
                           "partial_vintage;annual_and_underfill_exceptions")
        return Vintage(index_id,event,r,s,"KSIC8","KOSPI",None,None,None,None,
                       "KRX historical chronology; pre-2017 primary full methodology missing",
                       "uncertified_early_rules")
    if index_id == "KOSPI100":
        if r >= pd.Timestamp("2026-11-20"):
            if s < pd.Timestamp("2026-10-21"):
                return Vintage(index_id,event,r,s,"inherits_KOSPI200","predicted_KOSPI200",
                               None,None,None,None,"KRX 1121; 2026-10-20 notice",
                               "rule_not_known_at_signal")
            return Vintage(index_id,event,r,s,"inherits_KOSPI200","predicted_KOSPI200",
                           None,None,1.20,.80,
                           "지수_방법론/KOSPI+100·50지수+방법론(최신).pdf [2021-04]; KRX 1121 [reserve removal 2026-11-20]",
                           "rule_known;no_reserve_list")
        if r >= pd.Timestamp("2021-04-01"):
            if s < pd.Timestamp("2021-04-01"):
                return Vintage(index_id,event,r,s,"inherits_KOSPI200","predicted_KOSPI200",
                               None,None,None,None,"2021-04 methodology not yet public at signal",
                               "rule_not_known_at_signal")
            return Vintage(index_id,event,r,s,"inherits_KOSPI200","predicted_KOSPI200",
                           None,None,1.20,.80,
                           "지수_방법론/KOSPI+100·50지수+방법론(최신).pdf [document dated 2021-04]",
                           "rule_known;parent_forecast_required")
        return Vintage(index_id,event,r,s,"inherits_KOSPI200","predicted_KOSPI200",
                       None,None,None,None,"pre-2021 KOSPI100 vintage not locally authenticated",
                       "uncertified_early_rules")
    if index_id == "KRX300":
        if r < pd.Timestamp("2018-02-05"):
            return Vintage(index_id,event,r,s,"none","none",None,None,None,None,
                           "KRX 541", "index_not_launched")
        if r >= pd.Timestamp("2026-11-02"):
            if s < pd.Timestamp("2026-09-29"):
                return Vintage(index_id,event,r,s,"KRICS9","upcoming_midlarge_TMI",None,None,None,None,
                               "KRX 1120", "rule_not_known_at_signal")
            return Vintage(index_id,event,r,s,"KRICS9","upcoming_midlarge_TMI",.80,.80,1.10,None,
                           "지수_방법론/KRX+300+지수+방법론(최신).pdf; KRX 1120",
                           "rule_known;future_TMI_parent_must_be_forecast")
        if r >= pd.Timestamp("2025-01-13"):
            if s < pd.Timestamp("2025-01-09"):
                return Vintage(index_id,event,r,s,"GICS10","transition_unpublished",.80,.80,1.10,None,
                               "KRX 995", "rule_not_known_at_signal")
            return Vintage(index_id,event,r,s,"GICS10","upcoming_midlarge_TMI",.80,.80,1.10,None,
                           "지수_방법론/KRX+300+지수+방법론(2501).pdf; KRX 995",
                           "rule_known;future_TMI_parent_must_be_forecast")
        return Vintage(index_id,event,r,s,"GICS10","historical_BMI",.80,.80,1.10,None,
                       "지수_방법론/KRX300 방법론.pdf [document dated 2020-05]",
                       "rule_known;BMI_parent_roster_missing")
    raise KeyError(index_id)


def _frame(frame):
    f=frame.copy()
    f=f[f.eligibility_state.eq("pass") & f.mc.gt(0) & f.tv.gt(0) & f.grp.notna()]
    return f


def _first_stage(f,coverage,liquidity):
    chosen=set();ranks={};current_n={};liq=set()
    for _,g in f.groupby("grp"):
        g=g.sort_values(["mc"],ascending=[False],kind="mergesort")
        q=g.tv.rank(ascending=False,method="min") <= liquidity*len(g)
        c=g.mc.cumsum().shift(fill_value=0)/g.mc.sum() < coverage
        chosen.update(g.index[q & c]);liq.update(g.index[q])
        n=int(g.cur.sum())
        for j,c in enumerate(g.index,1):ranks[c]=j;current_n[c]=n
    return chosen,ranks,current_n,liq


def select_kospi200(frame,rule):
    """Use the established detailed selector with the corrected dated profile."""
    if rule.index_id!="KOSPI200" or rule.cap_coverage is None:
        raise ValueError("No certified mechanical KOSPI200 rule for this vintage")
    name="dated_signal_a_rules_v2"
    if name not in sys.modules:
        path=ROOT/"02_code/modules/signal_a_rules_v2/engine.py"
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    eng=sys.modules[name]
    p=eng.profile_for(rule.event,asof=rule.signal_date)
    if p.sector_scheme.upper()!=rule.industry.upper():
        raise ValueError(("Vintage sector mismatch",p.sector_scheme,rule.industry))
    result=eng.run_selection(frame,p,largecap_policy="exclude_unverified")
    return result.selected,{"eligible_rows":result.funnel["eligible_pass"],
                           "mechanical_count":len(result.selected),
                           "unknown_eligibility_rows":result.funnel["eligibility_unknown"],
                           "reserved_incumbents":len(result.reserved_incumbents),
                           "selection_issues":";".join(result.issues),
                           "official_special_exceptions_certified":False}


def select_kospi100(frame,predicted_parent,rule):
    if rule.index_id!="KOSPI100" or rule.incumbent_buffer is None:
        raise ValueError("No authenticated KOSPI100 vintage")
    f=_frame(frame).loc[lambda x:x.index.isin(predicted_parent)].sort_values("mc",ascending=False)
    if len(f)<100:raise ValueError("Fewer than 100 eligible predicted KOSPI200 constituents")
    ranks=pd.Series(np.arange(1,len(f)+1),index=f.index)
    incumbents=set(f.index[f.cur & ranks.le(120)])
    entrants=set(f.index[~f.cur & ranks.le(80)])
    selected=incumbents|entrants
    if len(selected)>100:
        order=sorted(selected,key=lambda c:(0 if bool(f.at[c,"cur"]) else 1,ranks[c],c))
        selected=set(order[:100])
    if len(selected)<100:
        selected.update([c for c in f.index if bool(f.at[c,"cur"]) and c not in selected][:100-len(selected)])
    return selected,{"eligible_parent_rows":len(f),"mechanical_count":len(selected),
                     "selection_underfill":max(0,100-len(selected)),
                     "parent_forecast_not_certified_by_this_function":True}


def tmi_asof(signal_date):
    """Observed current TMI list, never the future review list."""
    d=pd.Timestamp(signal_date)
    if d<pd.Timestamp("2025-01-13"):
        return None
    intervals=pd.read_parquet(TMI/"midlarge_tmi_membership_intervals.parquet")
    z=intervals[intervals.valid_from.le(d)&intervals.valid_until.ge(d)]
    if z.empty: return None
    return set(z.code)


def select_krx300_with_parent(frame,rule,observed_parent):
    """Mechanical KRX300 selection using an as-of forecast of the TMI parent."""
    if rule.index_id!="KRX300" or rule.parent!="upcoming_midlarge_TMI":
        raise ValueError("KRX300 TMI regime required")
    if observed_parent is None:raise ValueError("Contemporaneous TMI roster unavailable")
    f=_frame(frame).loc[lambda x:x.index.isin(observed_parent)]
    first,ranks,ncur,liq=_first_stage(f,rule.cap_coverage,rule.liquidity_fraction)
    selected=set(first)
    for _,g in f.groupby("grp"):
        q=g.sort_values("mc",ascending=False)
        selected.update(q.index[:int(np.ceil(.30*len(q)))].intersection(liq))
    global_rank=f.mc.rank(ascending=False,method="min")
    selected.update(f.index[f.cur & global_rank.le(330) & f.index.isin(liq)])
    if len(selected)>300:
        selected.difference_update(f.loc[list(selected)].sort_values("mc").index[:len(selected)-300])
    if len(selected)<300:
        pool=f.loc[~f.index.isin(selected)&f.index.isin(liq)].sort_values("mc",ascending=False)
        selected.update(pool.index[:300-len(selected)])
    return selected,{"supplied_parent_n":len(observed_parent),"eligible_parent_rows":len(f),
                     "mechanical_count":len(selected),"upcoming_TMI_parent_certified":False}


def select_krx300_current_parent_only(frame,rule,observed_parent):
    """Conservative observed-current-parent sensitivity, not the future list."""
    return select_krx300_with_parent(frame,rule,observed_parent)
