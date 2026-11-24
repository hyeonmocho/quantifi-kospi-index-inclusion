"""Timing sensitivity; no reselection from reused OOS, frozen V8 forecasts."""
from research import load,ROOT,OUT
import json,hashlib
import pandas as pd
import numpy as np
from datetime import datetime,timezone

DEST=OUT/'timing_comparison'
LEADS=[80,60,40,20]

def main():
 DEST.mkdir(exist_ok=True)
 choices=pd.read_csv(OUT/'performance_choices.csv').set_index('index_id')
 fr=pd.read_csv(OUT/'forecasts.csv',parse_dates=['information_date','entry_date','exit_date'])
 tr=pd.read_csv(OUT/'targets.csv')
 common_is=json.loads((OUT/'common_IS_events.json').read_text(encoding='utf8'))
 common_oos={};coverage=[]
 for idx,r in choices.iterrows():
  sets=[set(fr.loc[fr.index_id.eq(idx)&fr.model.eq(r.model)&fr.lead.eq(l)&fr.phase.eq('OOS'),'event']) for l in LEADS]
  common_oos[idx]=sorted(set.intersection(*sets))
  union=set.union(*sets)
  for l,s in zip(LEADS,sets):coverage.append(dict(index_id=idx,lead=l,available=len(s),common=len(common_oos[idx]),unavailable=';'.join(sorted(union-s)),excluded_from_common=';'.join(sorted(s-set(common_oos[idx])))))
 pd.DataFrame(coverage).to_csv(DEST/'coverage.csv',index=False,encoding='utf-8-sig')
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),leads=LEADS,selected_models=choices.model.to_dict(),budget=3e9,entry_sessions=10,fee_per_side=.0025,common_IS=common_is,common_OOS=common_oos,interpretation='same model family and decision rule across leads, lead-specific existing IS-fitted forecasts; not constant stock roster; reused OOS descriptive sensitivity, no new selection',hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['forecasts.csv','targets.csv','performance_choices.csv']})
 (DEST/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
 resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','timing_resource');resource.write=lambda *a:None
 px,tv,actions=resource.resources()
 from corporate_actions import supplement
 px,tv,actions=supplement(px,tv,actions)
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','timing_execution');ex.B=3e9
 rows=[];days=[];events=[];ledgers=[]
 for idx,r in choices.iterrows():
  for family,model in [('selected_family',r.model),('baseline','baseline')]:
   for phase in ['IS_forward','OOS']:
    for lead in LEADS:
     all_f=fr[fr.index_id.eq(idx)&fr.model.eq(model)&fr.lead.eq(lead)&fr.phase.eq(phase)].copy()
     common=common_is[idx] if phase=='IS_forward' else common_oos[idx]
     f=all_f[all_f.event.isin(common)].copy()
     cohorts=[('common',f)]
     if phase=='OOS' and len(all_f)!=len(f):cohorts.append(('available_supplement',all_f))
     for cohort,q in cohorts:
      start=pd.Timestamp(f'{int(q.year.min())}-01-01') if phase=='IS_forward' else pd.Timestamp('2023-01-02')
      end=pd.Timestamp('2022-12-30') if phase=='IS_forward' else pd.Timestamp('2026-06-30')
      ss=q[['case_id','index_id','event','lead','entry_date','exit_date']]
      aa=ss.assign(model=model,forecast_available=True);tt=tr[tr.model.eq(model)&tr.case_id.isin(ss.case_id)]
      result,d,g,l=ex.simulate(idx,model,lead,start,end,ss,aa,tt,px,tv,actions)
      meta=dict(index_id=idx,model=model,lead=lead,phase=phase,family=family,cohort=cohort,budget_krw=3e9)
      rows.append(dict(**result,**meta));print(idx,model,lead,phase,cohort,result['status'],flush=True)
      if d is not None:days.append(d.assign(**meta))
      if g is not None:events.append(g.assign(**meta))
      if l is not None and len(l):ledgers.append(l.assign(**meta))
 pd.DataFrame(rows).to_csv(DEST/'performance.csv',index=False,encoding='utf-8-sig')
 for name,v in [('daily',days),('events',events),('ledger',ledgers)]:pd.concat(v,ignore_index=True).to_parquet(DEST/(name+'.parquet'),index=False)
 validate()

def validate():
 protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf8'))
 results=pd.read_csv(DEST/'performance.csv');dd=pd.read_parquet(DEST/'daily.parquet');ee=pd.read_parquet(DEST/'events.parquet');ll=pd.read_parquet(DEST/'ledger.parquet')
 # Share exchanges/cash deliveries are corporate actions, not market orders.
 ll=ll[ll.side.isin(['buy','sell'])].copy()
 checks={'all_accounts_valid':bool(results.status.eq('valid').all()),'cash_nonnegative':bool(dd.cash.ge(-1e-4).all()),'fees_correct':bool(np.allclose(ll.fee,ll.notional*.0025)),'choices_unchanged':hashlib.sha256((OUT/'performance_choices.csv').read_bytes()).hexdigest()==protocol['hashes']['performance_choices.csv']}
 keys=['index_id','model','lead','phase','family','cohort']
 pnl=ee.groupby(keys).pnl.sum();ends=dd.sort_values('date').groupby(keys).equity.last()
 checks['event_PNL_matches_NAV']=bool(np.allclose(ends.reindex(pnl.index)-3e9,pnl,atol=.001))
 checks['within_daily_liquidity']=bool((ll.groupby(keys+['date','code']).notional.sum()<=ll.groupby(keys+['date','code']).limit.first()+1e-4).all())
 (DEST/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');assert all(checks.values()),checks
 print(results[results.phase.eq('OOS')&results.family.eq('selected_family')&results.cohort.eq('common')].to_string(index=False),flush=True)

if __name__=='__main__':main()
