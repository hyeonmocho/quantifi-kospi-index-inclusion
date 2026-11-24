"""Score original report rosters against dated K200 rule forecasts on the same date."""
import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np

MOD = Path(__file__).resolve().parent
ROOT = MOD.parents[1]
OUT = ROOT/'04_results/dated_report_replay_20261119'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(MOD/'index_forecast_broker_20261117'))
from asof_engines import Engines
import index_vintage_engine_20261119 as rules

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,MOD/file)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def main(trend=False,policy=False,output=None,trend_params=(.25,.4)):
    output = output or (ROOT/'04_results/kospi_pair_policy_20261121/broker_comparison' if policy else
              ROOT/'04_results/kospi_pair_trend_20261120/broker_comparison' if trend else OUT)
    output.mkdir(parents=True, exist_ok=True)
    eng=Engines()
    raw=ROOT/'01_data/raw/vintage_review_evidence_20261119'
    roster_log=pd.read_csv(raw/'report_date_rosters.csv',dtype={'requested_date':str})
    official={}
    for row in roster_log.loc[roster_log.get('file',pd.Series(index=roster_log.index,dtype=str)).notna()].itertuples():
        body=(raw/row.file).read_bytes()
        assert hashlib.sha256(body).hexdigest()==row.sha256
        names={'A'+item['isu_cd'] for item in json.loads(body)}
        if names:
            assert len(names)==row.rows
            official[pd.Timestamp(row.requested_date)]=names
    # Check the archived, complete exchange snapshots for exact signal-date matches.
    archive=pd.read_csv(ROOT/'01_data/processed/index_data_extension_20261117/kospi200_dated_membership_snapshots.csv')
    archive['known_from']=pd.to_datetime(archive.known_from,format='mixed',errors='coerce')
    initial=archive[archive.known_from.eq(archive.groupby('event').known_from.transform('min'))]
    announced_members={ev:set(g.code) for ev,g in initial.groupby('event')}
    first_sale={ev:g.known_from.iloc[0] for ev,g in initial.groupby('event')}
    archive=archive.loc[archive.verified.eq(True)&archive.full_snapshot_certified.eq(True)]
    for dt,group in archive.groupby('asof_date'):
        if pd.to_datetime(group.known_from).le(pd.Timestamp(dt)).all():
            official.setdefault(pd.Timestamp(dt),set(group.code))
    class ReportInputs:
        def __getattr__(self,name):
            return getattr(eng.adapter,name)

        def current_members(self,day,codes):
            known=official.get(pd.Timestamp(day).normalize())
            if known is None:
                return eng.adapter.current_members(day,codes=codes)
            return pd.Series(codes.isin(known),index=codes)
    inputs=ReportInputs()
    # Private snapshot builder wired to the dated selector. The archived
    # immutable V3 module and its checksum gate remain intact.
    frames=load('dated_report_frames','signal_a_rules_v3/engine_extension.py')
    selector=load('dated_report_selector','signal_a_rules_v2/engine.py')
    frames.v2_engine=lambda:selector
    resources=load('dated_report_prices','index_forecast_is_oos_20261117/evaluate.py')
    resources.write=lambda *args:None
    px,tv,actions=resources.resources()
    sys.path.insert(0,str(MOD/'index_engine_v8_revalidation_20261118'))
    from corporate_actions import supplement
    px,tv,actions=supplement(px,tv,actions)
    ex=load('dated_report_orders','index_forecast_development_20261117/execution.py')
    ex.B=1e10
    src=pd.read_csv(ROOT/'04_results/presentation_revision_20261118/all_source_registry.csv').fillna('')
    # Samsung's printed September 2024 list names OCI. The pre-report KIND
    # filing on July 24 identifies the issuer as 456040 (not OCI Holdings).
    resolved=src.report_id.eq('SAMSUNG_20240930_K200')
    assert src.loc[resolved,'primary_names'].tolist()==['효성중공업;에이피알;파라다이스;미원상사;OCI']
    src.loc[resolved,'primary_codes']='A298040;A278470;A034230;A002840;A456040'
    src.loc[resolved,'comparison_status']='historical_same_date_refresh'
    original=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/comparison/target_rosters.csv')
    audit=[];rows=[];stock_orders=[]
    @lru_cache(None)
    def predict(event,info,effective):
        rule=rules.vintage('KOSPI200',event,info,effective)
        if not rule.status.startswith('rule_known'): raise ValueError(rule.status)
        snap=frames.build_extension_snapshot(eng.re,event,signal=info,inputs=inputs,asof=info+pd.Timedelta(hours=15,minutes=30),allow_after_cutoff=True)
        f=snap.frame
        if trend:
            prices = px.loc[:info].tail(21).reindex(columns=f.index)
            momentum = prices.iloc[-1] / prices.iloc[-21] - 1
            alpha, limit = trend_params
            adjustment = (alpha * np.log1p(momentum.clip(lower=-.99)) * f.remaining_sessions / 40).fillna(0).clip(-limit, limit)
            cap = (f.mc_sum_observed + f.remaining_sessions * f.mc_current * np.exp(adjustment)) / f.review_sessions
            cap = cap.where(np.isfinite(cap) & cap.gt(0) & f.eligibility_state.eq('pass'), f.mc)
            prepared = selector.prepare_selector(f, selector.profile_for(event, asof=info))
            selected = prepared.select(cap.to_numpy(), f.tv.to_numpy())
            full = set(f.index[prepared.additions_mask(selected)])
        else:
            full,_=rules.select_kospi200(f,rule)
        ok=(~f.cur)&f.eligibility_state.eq('pass')&f.mc.gt(0)&f.tv.ge(0)&f.grp.notna()
        if 'strategy_a_route' in f:ok &= f.strategy_a_route.isin(['regular_candidate','regular_candidate_proxy'])
        if 'a_exclusion_reason' in f:ok &= f.a_exclusion_reason.fillna('').eq('')
        for col in ['float_observation_date','sector_source_date']:
            if col in f:ok &= ~pd.to_datetime(f[col]).gt(info)
        return full & set(f.index[ok])
    for r in src.itertuples():
        rec=dict(report_id=r.report_id,index_id=r.index_id,event=r.event,source_url=r.source_url)
        audit.append(rec)
        if r.index_id!='KOSPI200':
            rec['reason']='historical_BMI_unavailable' if r.status=='comparison_attempted' else 'incomplete_regular_roster'
            continue
        if r.event<'2020-06' or r.event>'2026-06':
            rec['reason']='outside_scored_rule_or_outcome_period';continue
        old=r.status=='comparison_attempted'
        if not old and r.comparison_status!='historical_same_date_refresh':
            rec['reason']=r.status;continue
        path=r.local_pdf_path or r.local_path
        if path and r.sha256: assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==r.sha256
        source=original[(original.report_id==r.report_id)&(original.who=='broker')&(original.design_id=='available_date_refresh')]
        broker=set(source.code.dropna()) if old else set(r.primary_codes.split(';'))-{''}
        assert broker and all(len(c)==7 and c.startswith('A') for c in broker),r.report_id
        special=set(r.special_codes.split(';'))-{''}
        fullsource=original[(original.report_id==r.report_id)&(original.who=='broker_source')]
        special.update(fullsource.loc[fullsource.in_regular_core_comparison.eq(False),'code'].dropna())
        broker-=special
        available=pd.Timestamp(r.assumed_available or r.printed_date)
        cal=eng.calendar('KOSPI200',r.event)
        assert available < cal['announcement']
        truth=eng.truth('KOSPI200',r.event)-special
        entry=eng.S[eng.S>available][0]
        for design,dt in [('release_day',available),('declared_inputs',r.information_asof)]:
            if not dt:continue
            info=eng.S[eng.S<=pd.Timestamp(dt)][-1]
            assert info<=available
            try: predicted=predict(r.event,info,cal['effective'])-special
            except ValueError as err:
                rec[design+'_reason']=str(err);continue
            for who,codes in [('engine',predicted),('broker',broker)]:
                schedule=pd.DataFrame([dict(index_id=r.index_id,event=r.event,case_id=r.report_id,lead=0,entry_date=entry,exit_date=cal['exit'])])
                targets=pd.DataFrame([(r.report_id,'report',c,1/len(codes)) for c in codes],columns=['case_id','model','code','weight'])
                overrides=({c: first_sale[r.event] for c in codes if c not in announced_members[r.event]}
                           if policy else None)
                z,d,e,order_ledger=ex.simulate(r.index_id,'report',0,entry,min(px.index[-1],cal['exit']+pd.Timedelta(days=120)),schedule,schedule.assign(model='report',forecast_available=True),targets,px,tv,actions,early_exit_dates=overrides)
                if policy and z['status']=='valid' and order_ledger is not None and len(order_ledger):
                    stock_orders.append(order_ledger.assign(report_id=r.report_id,design=design,who=who))
                rows.append(dict(report_id=r.report_id,broker_name=r.broker,event=r.event,design=design,info=info,available=available,entry=entry,exit=cal['exit'],who=who,codes=';'.join(sorted(codes)),actual_codes=';'.join(sorted(truth)),predicted=len(codes),hits=len(codes&truth),actual=len(truth),execution=z['status'],net=e.net_return.iloc[0] if z['status']=='valid' and len(e)==1 else np.nan))
            rec['reason']='paired'
    a=pd.DataFrame(rows)
    a['incumbent_source']=np.where(a['info'].isin(official),'KRX_same_day','archived_asof_adapter')
    a.to_csv(output/'pairs.csv',index=False,encoding='utf-8-sig')
    if policy:
        pd.concat(stock_orders,ignore_index=True).to_csv(output/'orders.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(audit).to_csv(output/'coverage.csv',index=False,encoding='utf-8-sig')
    result=[]
    for design,g in a.groupby('design'):
        valid=g.groupby('report_id').net.count().loc[lambda n:n==2].index
        for who,q in g.groupby('who'):
            counts=q.groupby('event')[['predicted','hits','actual']].mean().sum()
            ret=q[q.report_id.isin(valid)].groupby('event').net.mean()
            result.append(dict(design=design,who=who,reports=len(q),reviews=q.event.nunique(),precision=counts.hits/counts.predicted,recall=counts.hits/counts.actual,return_pairs=len(valid),return_reviews=len(ret),mean_net=ret.mean(),win=ret.gt(0).mean(),payoff=ret[ret>0].mean()/-ret[ret<0].mean()))
    summary=pd.DataFrame(result)
    summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    from scipy.stats import ttest_1samp
    paired=a.pivot(index=['design','event','report_id'],columns='who',values='net').dropna()
    spread=(paired.engine-paired.broker).groupby(['design','event']).mean().rename('net_difference')
    spread.to_csv(output/'return_differences_by_review.csv',encoding='utf-8-sig')
    tests=[]
    for design,q in spread.groupby(level='design'):
        test=ttest_1samp(q,0)
        tests.append(dict(design=design,reviews=len(q),mean_difference=q.mean(),t=test.statistic,p_two_sided=test.pvalue))
    pd.DataFrame(tests).to_csv(output/'paired_tests.csv',index=False,encoding='utf-8-sig')
    print(summary.to_string(index=False))
if __name__=='__main__':main(trend='--trend' in sys.argv,policy='--policy' in sys.argv)
