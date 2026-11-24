"""Re-evaluate dated mechanical cores on separate IS/OOS and common OOS.

This does not reuse the frozen ML panels or optimize a parameter on OOS.
"""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'04_results/index_method_vintage_20261119'
OUT=ROOT/'04_results/vintage_submission_20261119'
OUT.mkdir(parents=True,exist_ok=True)
SPLIT={'KOSPI200':2022,'KOSPI100':2022,'KRX300':2025}
START={'KOSPI200':'2020-01-01','KOSPI100':'2021-01-01','KRX300':'2025-01-01'}

def load(rel,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

def single_date_reference():
    a=pd.read_csv(SRC/'vintage_replay_case_audit.csv')
    a=a[a.index_id.isin(['KOSPI200','KOSPI100']) & a.status.eq('mechanical_replayed')]
    k=pd.read_csv(SRC/'tmi_parent_krx300_case_audit.csv')
    k=k[k.status.eq('projected_TMI_parent_research')]
    a=pd.concat([a,k],ignore_index=True)
    panel=pd.read_parquet(ROOT/'04_results/index_engine_v6_20261118/candidate_panel.parquet')
    dates=panel.groupby('case_id').first()[['entry_date','exit_date']]
    a=a.join(dates,on='case_id')
    for c in ['entry_date','exit_date','signal_date']:
        a[c]=pd.to_datetime(a[c])
    a['phase']=np.where(a.event.str[:4].astype(int)<=a.index_id.map(SPLIT),'IS','OOS')
    t0=pd.read_parquet(SRC/'vintage_replay_targets.parquet')
    t1=pd.read_parquet(SRC/'tmi_parent_krx300_targets.parquet')
    targets=pd.concat([t0[t0.index_id.ne('KRX300')],t1],ignore_index=True)
    schedule=a[['case_id','index_id','event','lead','entry_date','exit_date','phase']].copy()
    acc=a[['case_id','index_id','lead']].assign(model='dated_core',forecast_available=True)
    names=targets[['case_id','code']].drop_duplicates().copy()
    cnt=names.groupby('case_id').code.transform('size')
    names['weight']=1/cnt;names['model']='dated_core'
    resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','vintage_submission_resources')
    resource.write=lambda *args:None
    px,tv,actions=resource.resources()
    sys.path.insert(0,str(ROOT/'02_code/modules/index_engine_v8_revalidation_20261118'))
    corp=load('02_code/modules/index_engine_v8_revalidation_20261118/corporate_actions.py','vintage_submission_actions')
    px,tv,actions=corp.supplement(px,tv,actions)
    execution=load('02_code/modules/index_forecast_development_20261117/execution.py','vintage_submission_execution')
    execution.B=1e10
    rows=[];ds=[];es=[];ls=[];pred=[]
    for idx in SPLIT:
        for lead in [20,40,60,80]:
            for phase in ['IS','OOS']:
                q=a[a.index_id.eq(idx)&a.lead.eq(lead)&a.phase.eq(phase)]
                if q.empty:continue
                start=pd.Timestamp(START[idx] if phase=='IS' else f'{SPLIT[idx]+1}-01-01')
                last=q.exit_date.max();future=px.index[px.index>last]
                end=future[min(20,len(future)-1)] if len(future) else px.index.max()
                result,d,e,l=execution.simulate(idx,'dated_core',lead,start,end,schedule,acc,names,px,tv,actions)
                row=dict(index_id=idx,lead=lead,phase=phase,first_event=q.event.min(),last_event=q.event.max(),
                         event_count=q.event.nunique(),**result)
                rows.append(row)
                if d is not None:ds.append(d.assign(index_id=idx,lead=lead,phase=phase))
                if e is not None:
                    e=e.assign(index_id=idx,lead=lead,phase=phase)
                    es.append(e)
                if l is not None and len(l):ls.append(l.assign(index_id=idx,lead=lead,phase=phase))
                n=q.predicted.sum();h=q.hits.sum();y=q.actual.sum()
                pred.append(dict(index_id=idx,lead=lead,phase=phase,event_count=q.event.nunique(),
                                 predicted=n,hits=h,actual=y,precision=h/n if n else np.nan,
                                 recall=h/y if y else np.nan))
                print(idx,lead,phase,result.get('status'),q.event.nunique(),flush=True)
    perf=pd.DataFrame(rows);pred=pd.DataFrame(pred)
    if es:
        ev=pd.concat(es,ignore_index=True)
        inference=[]
        for (idx,lead,phase),g in ev.groupby(['index_id','lead','phase']):
            x=g.net_return.dropna().to_numpy();n=len(x)
            if n>=3 and np.std(x,ddof=1)>0:
                tt,pp=stats.ttest_1samp(x,0)
            else:tt=pp=np.nan
            inference.append(dict(index_id=idx,lead=lead,phase=phase,n_events=n,
                                  mean_event_net=np.mean(x) if n else np.nan,
                                  median_event_net=np.median(x) if n else np.nan,
                                  win_rate=np.mean(x>0) if n else np.nan,
                                  t=tt,p=pp,method='event one-sample t; descriptive only for small n'))
        pd.DataFrame(inference).to_csv(OUT/'inference.csv',index=False,encoding='utf-8-sig')
        ev.to_parquet(OUT/'events.parquet',index=False)
    if ds:pd.concat(ds,ignore_index=True).to_parquet(OUT/'daily.parquet',index=False)
    if ls:pd.concat(ls,ignore_index=True).to_parquet(OUT/'orders.parquet',index=False)
    perf.to_csv(OUT/'standalone_performance.csv',index=False,encoding='utf-8-sig')
    pred.to_csv(OUT/'prediction_accuracy.csv',index=False,encoding='utf-8-sig')
    schedule.to_csv(OUT/'case_schedule.csv',index=False,encoding='utf-8-sig')
    chosen=perf[perf.phase.eq('IS')&perf.status.eq('valid')].sort_values(
        ['index_id','sharpe','lead'],ascending=[True,False,True]).drop_duplicates('index_id')
    choices=chosen[['index_id','lead']].assign(model='dated_core')
    choices.to_csv(OUT/'IS_selected_leads.csv',index=False,encoding='utf-8-sig')
    common=set.intersection(*(set(a.loc[a.index_id.eq(r.index_id)&a.phase.eq('OOS')&a.lead.eq(r.lead),'event'])
                              for r in choices.itertuples()))
    common=sorted(common)
    joint=load('02_code/modules/index_three_portfolio_20261117/engine.py','vintage_submission_joint')
    joint.BUDGET=1e10
    eligible=schedule.loc[schedule.phase.eq('OOS')].merge(choices[['index_id','lead']],on=['index_id','lead'],validate='many_to_one')
    union_end=px.index[px.index>eligible.exit_date.max()][:21][-1]
    um,ud,ue,ul,uh,uc=joint.simulate(choices,eligible,acc,names,px,tv,actions,shared=True,start='2023-01-01',end=union_end)
    assert len(ue)==len(eligible) and ud.groupby('index_id').tail(1).held_value.abs().max()<1
    um.to_csv(OUT/'staggered_performance.csv',index=False,encoding='utf-8-sig')
    for label,frame in [('daily',ud),('events',ue),('orders',ul),('holdings',uh)]:
        frame.to_parquet(OUT/f'staggered_{label}.parquet',index=False)
    js=schedule[schedule.event.isin(common)&schedule.phase.eq('OOS')].merge(
        choices[['index_id','lead']],on=['index_id','lead'])
    if common:
        jd=js.entry_date.min();last=js.exit_date.max();future=px.index[px.index>last]
        je=future[min(20,len(future)-1)] if len(future) else px.index.max()
        m,d,e,l,h,checks=joint.simulate(choices,js,acc,names,px,tv,actions,shared=True,start=jd,end=je)
        m.to_csv(OUT/'joint_performance.csv',index=False,encoding='utf-8-sig')
        d.to_parquet(OUT/'joint_daily.parquet',index=False)
        e.to_parquet(OUT/'joint_events.parquet',index=False)
        l.to_parquet(OUT/'joint_orders.parquet',index=False)
        (OUT/'joint_checks.json').write_text(json.dumps(checks,default=str,ensure_ascii=False,indent=2),encoding='utf8')
    pair=choices[choices.index_id.isin(['KOSPI200','KOSPI100'])]
    pair_events=sorted(set.intersection(*(set(a.loc[a.index_id.eq(r.index_id)&a.phase.eq('OOS')&a.lead.eq(r.lead),'event'])
                                         for r in pair.itertuples())))
    ps=schedule[schedule.event.isin(pair_events)&schedule.phase.eq('OOS')].merge(
        pair[['index_id','lead']],on=['index_id','lead'])
    if pair_events:
        pd0=ps.entry_date.min();last=ps.exit_date.max();future=px.index[px.index>last]
        pd1=future[min(20,len(future)-1)] if len(future) else px.index.max()
        pm,pdly,pe,pl,ph,pcheck=joint.simulate(pair,ps,acc,names,px,tv,actions,shared=True,start=pd0,end=pd1)
        pm.to_csv(OUT/'pair_performance.csv',index=False,encoding='utf-8-sig')
        pdly.to_parquet(OUT/'pair_daily.parquet',index=False)
        pe.to_parquet(OUT/'pair_events.parquet',index=False)
        pl.to_parquet(OUT/'pair_orders.parquet',index=False)
    protocol=dict(asof='2026-11-19',core='dated mechanical selection, not trained logistic',
                  IS_end=SPLIT,first_source_year=START,leads=[20,40,60,80],
                  budget_each_krw=1e10,fee_each_side=.0025,
                  same_security_daily_limit_shared=True,common_OOS_events=common,
                  KOSPI200_100_common_OOS_events=pair_events,
                  IS_selected_leads=choices[['index_id','lead']].to_dict('records'),
                  KRX300='research TMI forecast with incomplete eligibility/cap data',
                  sample_warning='do not select best lead on OOS or claim independent unseen OOS')
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    print(perf[['index_id','lead','phase','status','event_count','total_return','sharpe','mdd']].to_string(index=False))

if __name__=='__main__':
    from common_index_engine_20261119 import select, run
    from common_index_aligned_windows_20261119 import main as run_common_windows
    select()
    run()
    run_common_windows()
