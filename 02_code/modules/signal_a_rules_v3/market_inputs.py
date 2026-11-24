"""Separate historical and current-observation inputs for V3 research.

The legacy adjusted price cache is never extended with raw prices. The current
namespace uses a specifically labelled raw-price positivity gate, not a return
series. New market-data collection time and quote dates remain distinct.
"""
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "04_results/signal_a_rules_v3_20261107"
V2 = ROOT / "02_code/modules/signal_a_rules_v2"
HISTORY = OUT / "history"


@lru_cache(maxsize=1)
def old_inputs():
    spec = importlib.util.spec_from_file_location("v3_old_inputs_readonly", V2 / "inputs.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def kst(value):
    t = pd.Timestamp(value)
    return t.tz_convert("Asia/Seoul").tz_localize(None) if t.tzinfo else t


def bool_series(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def read_history(name):
    path = HISTORY / (name + ".csv")
    return pd.read_csv(path, dtype={"code": str, "event": str}) if path.exists() else pd.DataFrame()


def known_rows(data, asof):
    if data.empty or "known_from" not in data:
        return data.iloc[:0].copy()
    known = data.known_from.map(lambda v: kst(v) if pd.notna(v) else pd.NaT)
    verified = bool_series(data.verified) if "verified" in data else pd.Series(False,index=data.index)
    result=data.loc[known.notna() & known.le(kst(asof)) & verified].copy()
    if "supersedes_source_id" in result and "source_id" in result:
        remove=pd.Series(False,index=result.index)
        for _,row in result.loc[result.supersedes_source_id.notna()].iterrows():
            value=str(row.supersedes_source_id).strip()
            if not value or value.lower() in {"nan","none"}:continue
            try:prior=json.loads(value) if value.startswith("[") else value.replace(",","|").split("|")
            except json.JSONDecodeError:raise ValueError("Invalid explicit correction source IDs")
            if not isinstance(prior,list):prior=[prior]
            prior=[str(x).strip() for x in prior]
            if str(row.source_id) in prior:raise ValueError("A source cannot supersede itself")
            mask=result.source_id.astype(str).isin(prior)
            scope=str(row.get("correction_scope", "row"))
            # Partial errata replace the same code/event, not every other
            # security in a multi-stock original notice. Full-source replacement
            # requires an explicit correction_scope='source'.
            if scope!="source":
                if "code" in result and pd.notna(row.get("code")):
                    mask &= result.code.eq(row.code)
                if "event" in result and pd.notna(row.get("event")):
                    mask &= result.event.eq(row.event)
            remove |= mask
        result=result.loc[~remove].copy()
    return result


def final_decisions(data):
    """A genuine document may describe a conditional, not final, decision."""
    if "decision_status" not in data:return data
    return data.loc[data.decision_status.fillna("unknown").astype(str).str.lower().isin(
        ["final","confirmed","final_confirmed"])].copy()


@lru_cache(maxsize=1)
def fresh_daily():
    p = OUT / "market/fresh_daily.parquet"
    if p.exists():
        data=pd.read_parquet(p)
        observation=OUT/"market/krics_quote_snapshot.parquet"
        if observation.exists():
            extra=pd.read_parquet(observation)
            assert extra.date.min()>data.date.max(),"A later partial quote must not overwrite marcap history"
            data=pd.concat([data,extra],ignore_index=True)
            assert not data.duplicated(["date","code"]).any()
        return data
    q = pd.read_parquet(ROOT / "01_data/processed/market_daily_update_20261106/daily_market.parquet")
    return q.rename(columns={"close": "raw_close", "turnover_krw": "turnover"})


@lru_cache(maxsize=1)
def current_native():
    old = old_inputs().native()
    fresh = fresh_daily()
    base_path = ROOT / "01_data/processed/market_daily_update_20261106/merged"
    raw = pd.read_parquet(base_path / "raw_close.parquet")
    sessions = old.S.union(pd.DatetimeIndex(fresh.date.unique())).sort_values()
    columns = old.px.columns.union(raw.columns).union(pd.Index(fresh.code.unique())).sort_values()
    mc = old.MCD.reindex(index=sessions, columns=columns).copy()
    tv = old.TVD.reindex(index=sessions, columns=columns).copy()
    # The legacy final09/07 row differs from the redistributed full-day source.
    # Replace this provisional tail only in the new V3 namespace.
    for column, frame, scale in [("market_cap_krw",mc,1e6),("turnover",tv,1.)]:
        patch = fresh.pivot(index="date",columns="code",values=column).reindex(columns=columns)
        patch = patch.loc[patch.index >= pd.Timestamp("2026-09-07")]
        frame.loc[patch.index,:] = patch / scale
    price = raw.reindex(index=sessions,columns=columns).copy()
    raw_last = raw.index[raw.notna().any(axis=1)][-1]
    tail = fresh.loc[fresh.date > raw_last].pivot(index="date",columns="code",values="raw_close")
    if len(tail):
        price.loc[tail.index,:] = tail.reindex(columns=columns)
    first = old.FIRST_PX.reindex(columns).copy()
    first_raw = price.gt(0).idxmax().where(price.gt(0).any())
    first = pd.concat([first,first_raw],axis=1).min(axis=1)
    listing = pd.read_csv(ROOT / "01_data/processed/kind_listing.csv",parse_dates=["list_date"])
    latest = fresh.sort_values("date").drop_duplicates("code",keep="last")
    source = json.loads((OUT / "market/source_audit.json").read_text(encoding="utf-8"))
    collected = kst(source["retrieved_at_utc"])
    quote_path=OUT/"market/krics_quote_audit.json"
    quote_audit=json.loads(quote_path.read_text(encoding="utf-8")) if quote_path.exists() else None
    if quote_audit is not None:
        collected=max(collected,kst(quote_audit["retrieved_at"]))
    names = dict(old.names)
    names.update(dict(zip(latest.code,latest["name"])))
    new_names = latest[["code","name"]].assign(date=collected)
    hist_names = pd.concat([old._nm_hist,new_names],ignore_index=True).sort_values("date")
    def market_at(t):
        state = old.market_at(min(kst(t),old.S[-1])).reindex(columns).copy()
        # t here is the market observation date. Information availability is
        # checked separately by get_float/build_extension_snapshot at decision
        # time; today's namespace can inspect an observed past market label.
        records = fresh.loc[fresh.date.le(kst(t))].sort_values("date").drop_duplicates("code",keep="last")
        state.loc[records.code] = records.market.map({"KOSPI":"유가증권시장","KOSDAQ":"코스닥"}).to_numpy()
        return state
    nonstock = listing.loc[listing.secu.isin(old.NON_STOCK)].groupby("code").list_date.min()
    management=old.MGT
    status_audit_path=OUT/"current_status/collection_audit.json"
    normalized_status=OUT/"current_status/management_positive_observations.csv"
    status_path=normalized_status if normalized_status.exists() else OUT/"current_status/admin_positive_observations.csv"
    status_audit=json.loads(status_audit_path.read_text(encoding="utf-8")) if status_audit_path.exists() else {}
    positive_count=0
    if status_path.exists():
        admin=pd.read_csv(status_path,dtype={"종목코드":str,"code":str})
        matching=[a for a in status_audit.get("attempts",[]) if a.get("source_id")=="admin_download" and a.get("status_code")==200]
        if len(admin) and (matching or "observed_at_utc" in admin):
            known=max(admin.observed_at_utc.map(kst)) if "observed_at_utc" in admin else kst(matching[-1]["retrieved_at_utc"])
            row=old.MGT.iloc[-1].reindex(columns).copy()
            codes_admin=admin.code if "code" in admin else "A"+admin["종목코드"].str.upper().str.zfill(6)
            found=columns.intersection(codes_admin)
            row.loc[found]="관리"
            positive_count=len(found)
            management=pd.concat([old.MGT.reindex(columns=columns),row.to_frame().T.set_axis([known])]).sort_index()
    namespace = SimpleNamespace(S=sessions,px=price,PXV=price.to_numpy(),MCD=mc,TVD=tv,
        CI={c:j for j,c in enumerate(columns)},FIRST_PX=first,KL=listing,
        MGT=management,market_at=market_at,_nm_hist=hist_names,EXCL=old.EXCL,
        CODE_OK=pd.Series([str(c).endswith("0") and not str(c).startswith("A9") for c in columns],index=columns),
        NONSTK_FROM=nonstock,names=names,GICS=old.GICS,WICS=old.WICS,SEC=old.SEC,IG_MAP=old.IG_MAP,
        source_known_from=collected,price_basis="observed_raw_or_delayed_current_quote_positivity_gate_ONLY_not_adjusted_returns",
        latest_quote_audit=quote_audit,
        management_status_audit=status_audit,management_positive_codes=positive_count)
    return namespace


class InputsAdapter:
    def __init__(self,current=False):
        self.current=current
        self.re=current_native() if current else old_inputs().native()
        # An adapter instance belongs to one frozen source vintage/run.
        self._route_cache={}

    def get_float(self,t,codes,lag_sessions=1):
        if not self.current:
            return old_inputs().get_float(t,codes,lag_sessions)
        t=kst(t)
        # This source vintage is only used for a decision made after collection.
        if t < self.re.source_known_from:
            raise ValueError("Current source vintage was not collected by information_asof")
        sessions=self.re.S
        j=sessions.searchsorted(t,side="right")-1-lag_sessions
        reference=sessions[j] if j>=0 else pd.NaT
        panel=old_inputs().float_panel()
        rows=panel.index[panel.index<=reference]
        observed=rows[-1] if len(rows) else pd.NaT
        values=panel.loc[observed].reindex(codes) if pd.notna(observed) else pd.Series(np.nan,index=codes)
        return pd.DataFrame(dict(float_ratio=values,ratio=values,
            status=np.where(values.notna(),"observed_vendor_latest_known","unknown_float"),
            source_date=observed,observation_date=observed,
            first_publication_verified=False,observation_age_days=(t.normalize()-observed).days if pd.notna(observed) else np.nan))

    def get_shares(self,t,codes,lag_sessions=1):
        if not self.current:return old_inputs().get_shares(t,codes,lag_sessions)
        t=kst(t)
        if t<self.re.source_known_from:raise ValueError("Share source vintage unavailable")
        j=self.re.S.searchsorted(t,side="right")-1-lag_sessions
        reference=self.re.S[j] if j>=0 else pd.NaT
        q=fresh_daily().loc[lambda f:f.date.le(reference)].sort_values("date").drop_duplicates("code",keep="last").set_index("code")
        return pd.DataFrame(dict(shares=q.listed_shares.reindex(codes),source_date=q.date.reindex(codes),
                                 status="observed_public_redistribution"))

    def get_sector(self,t,event,codes,profile=None):
        return old_inputs().get_sector(kst(t),event,codes,profile)

    def current_members(self,t,codes=None):
        t=kst(t)
        if codes is None:codes=self.re.px.columns
        baseline=old_inputs().current_members(min(t,self.re.S[-1]),codes=codes)
        base=baseline.attrs.get("base_snapshot")
        base=pd.Timestamp(base) if pd.notna(base) else pd.Timestamp("1900-01-01")
        result=baseline
        candidates=[known_rows(read_history(n),t) for n in ["current_members_snapshot","dated_membership_snapshots"]]
        full=pd.concat([d for d in candidates if len(d)],ignore_index=True) if any(len(d) for d in candidates) else pd.DataFrame()
        official_anchor=None
        if len(full) and "full_snapshot_certified" in full:
            full=full.loc[bool_series(full.full_snapshot_certified)]
            if len(full):
                full["asof_date"]=pd.to_datetime(full.asof_date,format="mixed")
                full=full.loc[full.asof_date<=t]
                if len(full):
                    date=full.asof_date.max()
                    chosen=full.loc[full.asof_date.eq(date)]
                    official_anchor=dict(date=str(date),members=len(set(chosen.code)),source_ids=sorted(set(chosen.source_id.astype(str))))
                    # A June official anchor does not erase later observed
                    # monthly changes when the intervening offcycle ledger is
                    # incomplete. Compare and report the gap, rather than
                    # silently calling the old official list current.
                    if date>=base:
                        result=pd.Series(pd.Index(codes).isin(chosen.code),index=codes)
                        base=date
                        result.attrs.update(base_snapshot=str(date),coverage="official_full_snapshot_plus_known_changes",
                                            source_vintage_certified=True,offcycle_ledger_complete=False)
                    else:
                        anchor_set=set(chosen.code)
                        result.attrs["unresolved_monthly_vs_official_anchor_delta"]=sorted(set(result.index[result])^anchor_set)
        ledger=final_decisions(known_rows(read_history("membership_decisions"),t))
        if len(ledger):
            effective=pd.to_datetime(ledger.effective_date,format="mixed",errors="coerce")
            ledger=ledger.loc[effective.gt(base)&effective.le(t)].copy()
            ledger["_effective"]=pd.to_datetime(ledger.effective_date,format="mixed")
            for row in ledger.sort_values("_effective").itertuples():
                if row.code in result.index:result.at[row.code]=str(row.side).lower() in {"add","편입","addition"}
        if result is baseline:
            result.attrs.update(coverage="monthly_proxy_plus_documented_changes_partial",source_vintage_certified=False)
        result.attrs.update(offcycle_ledger_complete=False,latest_official_anchor=official_anchor)
        return result

    def official_decisions(self,t,event=None):
        data=final_decisions(known_rows(read_history("membership_decisions"),t))
        return data.loc[data.event.eq(event)] if len(data) and event is not None else data

    def regular_committee_calendar(self,event,asof):
        data=known_rows(read_history("regular_review_calendar"),asof)
        data=data.loc[data.event.eq(event)] if len(data) else data
        return data

    def known_action_resets(self,t,event=None):
        data=known_rows(read_history("corporate_review_resets"),t)
        if data.empty:return data
        data=data.loc[data.reset_start_date.notna()].copy()
        if event is not None:
            year,month=map(int,event.split("-"))
            first=pd.Timestamp(year,month,1)
            target=first+pd.Timedelta(days=(3-first.weekday())%7+8)
            data=data.loc[pd.to_datetime(data.effective_date,format="mixed",errors="coerce").le(target)]
        data["event_type"]=data.case_type
        data["review_start_date"]=data.reset_start_date
        return data

    def corporate_action_flags(self,t,event=None):
        old=old_inputs().corporate_action_flags(kst(t),event)
        new=known_rows(read_history("corporate_event_flags"),t)
        if len(new):
            # Legacy builder accepts announced_at as an availability gate.
            # Preserve the raw announcement field and explicitly label this
            # adapter alias; a date-only known_from is not a claimed exact time.
            new["source_announced_at"]=new.get("announced_at",pd.NaT)
            new["announced_at"]=new.known_from.map(kst)
            new["announced_at_semantics"]="availability_gate_alias_not_claimed_original_timestamp"
            return pd.concat([old,new],ignore_index=True)
        return old

    def strategy_a_exclusion(self,code,event,information_date):
        key=(event,str(kst(information_date)))
        if key not in self._route_cache:
            self._route_cache[key]=self._offcycle_routes(event,information_date)
        return self._route_cache[key].get(code,"")

    def _offcycle_routes(self,event,information_date):
        d=final_decisions(known_rows(read_history("membership_decisions"),information_date))
        if d.empty:return {}
        # Only a documented off-cycle decision in the relevant review cycle
        # excludes this route. A transfer history is never a permanent blacklist.
        year,month=map(int,event.split("-"))
        if month==12:previous=f"{year}-06"
        elif year<=2020:previous=f"{year-1}-06"
        else:previous=f"{year-1}-12"
        prev_calendar=known_rows(read_history("regular_review_calendar"),information_date)
        prev_calendar=prev_calendar.loc[prev_calendar.event.eq(previous)] if len(prev_calendar) else prev_calendar
        if len(prev_calendar) and pd.to_datetime(prev_calendar.effective_date,format="mixed",errors="coerce").notna().any():
            start=pd.to_datetime(prev_calendar.effective_date,format="mixed",errors="coerce").max()
        else:
            # Nominal second-Thursday-next-session boundary is only used for
            # the cycle filter when the official previous calendar is missing.
            py,pm=map(int,previous.split("-"))
            first=pd.Timestamp(py,pm,1)
            start=first+pd.Timedelta(days=(3-first.weekday())%7+8)
        target_calendar=known_rows(read_history("regular_review_calendar"),information_date)
        target_calendar=target_calendar.loc[target_calendar.event.eq(event)] if len(target_calendar) else target_calendar
        if len(target_calendar) and pd.to_datetime(target_calendar.effective_date,format="mixed",errors="coerce").notna().any():
            end=pd.to_datetime(target_calendar.effective_date,format="mixed",errors="coerce").max()
        else:
            first=pd.Timestamp(year,month,1)
            end=first+pd.Timedelta(days=(3-first.weekday())%7+8)
        effective=pd.to_datetime(d.effective_date,format="mixed",errors="coerce")
        cycle=d.loc[effective.gt(start)&effective.le(end)].copy()
        if cycle.empty:return {}
        cycle["_effective"]=pd.to_datetime(cycle.effective_date,format="mixed")
        cycle["_known"]=cycle.known_from.map(kst)
        cycle=cycle.sort_values(["_effective","_known"],kind="stable").drop_duplicates("code",keep="last")
        added=cycle.side.astype(str).str.lower().isin(["add","편입","addition"])
        offcycle=cycle.decision_type.astype(str).str.contains("offcycle|수시",case=False,regex=True)
        return {code:"documented_offcycle_entry_in_current_review_cycle" for code in cycle.loc[added&offcycle,"code"]}


def write_market_audit(asof):
    if (OUT/"forward/protocol_v0001.json").exists():
        raise RuntimeError("Registered market snapshot is immutable; write a new dated observation version.")
    re=current_native()
    adapter=InputsAdapter(current=True)
    latest=re.S[-1]
    source=json.loads((OUT/"market/source_audit.json").read_text(encoding="utf-8"))
    floats=adapter.get_float(asof,re.px.columns)
    current=adapter.current_members(asof,re.px.columns)
    price=re.px.loc[latest]
    cap=re.MCD.loc[latest]
    tv=re.TVD.loc[latest]
    frame=pd.DataFrame(dict(raw_price_gate=price,mc_million_krw=cap,turnover_krw=tv,
                           float_ratio=floats.float_ratio,current_member=current,
                           market=re.market_at(asof)))
    frame.to_parquet(OUT/"market/latest_inputs.parquet")
    audit=dict(information_asof=str(kst(asof)),source_available_from=str(re.source_known_from),
        quote_last_date=str(latest.date()),native_previous_end=str(old_inputs().native().S[-1].date()),
        trading_sessions=len(re.S),codes=len(re.px.columns),
        latest_joint_observed=int((price.gt(0)&cap.gt(0)&tv.notna()).sum()),
        latest_actual_zero_turnover=int(tv.eq(0).sum()),float_observed=int(floats.float_ratio.notna().sum()),
        latest_quote_coverage_complete=False,latest_quote_source=re.latest_quote_audit,
        float_latest_date=str(floats.observation_date.max()),current_members=int(current.sum()),
        membership_state=current.attrs,price_basis=re.price_basis,market_cap_units="million_KRW",
        turnover_units="KRW",raw_prices_not_used_for_historical_returns=True,
        original_cache_modified=False,management_flag_latest=str(re.MGT.index[-1]),
        management_latest_state_certified=False,management_positive_codes=re.management_positive_codes,
        management_absence_is_not_normal=True,management_source=re.management_status_audit,
        complete_corporate_ledger=False,
        downloaded_source=source)
    (OUT/"market/latest_inputs_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return audit


if __name__=="__main__":
    print(json.dumps(write_market_audit(pd.Timestamp.now(tz="Asia/Seoul")),ensure_ascii=False,default=str),flush=True)
