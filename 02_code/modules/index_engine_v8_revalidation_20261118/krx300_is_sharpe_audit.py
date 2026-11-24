"""Reconcile historical KRX300 IS/OOS Sharpe without selecting new settings."""
from research import ROOT,OUT,load
import pandas as pd,numpy as np,json
from threadpoolctl import threadpool_limits

def main():
 threadpool_limits(1)
 dest=OUT/'rule_margin_audit/is_sharpe_reconciliation';dest.mkdir(exist_ok=True)
 resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','is_audit_resources');resource.write=lambda *a:None
 px,tv,actions=resource.resources()
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','is_audit_execution')
 old=ROOT/'04_results/index_forecast_development_20261117'
 schedule=pd.read_csv(ROOT/'04_results/index_forecast_is_oos_20261117/case_schedule.csv',parse_dates=['information_date','entry_date','exit_date','review_cutoff'])
 acc=pd.read_csv(old/'accuracy.csv');acc=acc[acc.train_end.eq(2022)]
 targets=pd.read_parquet(old/'execution_targets.parquet');targets=targets[targets.train_end.eq(2022)]
 common=['2020-12','2021-06','2021-12','2022-06','2022-12']
 rows=[];events=[];daily=[]
 panel=pd.read_parquet(OUT/'rule_margin_audit/model_panel.parquet')
 gates=panel[['case_id','code','opportunity']]
 for name,budget,matched in [('old_9events_3eok',3e8,False),('old_5events_3eok',3e8,True),('old_5events_30eok',3e9,True),('old_current_eligibility_cash',3e9,True),('old_current_eligibility_equalweight',3e9,True)]:
  ss=schedule[schedule.event.isin(common)] if matched else schedule
  start=pd.Timestamp('2020-01-01' if matched else '2018-07-01');ex.B=budget
  tt=targets
  if 'current_eligibility' in name:
   tt=targets.merge(gates,on=['case_id','code'],how='left',validate='many_to_one')
   tt=tt[tt.opportunity.eq(True)].copy()
   if name.endswith('equalweight'):tt['weight']=1/tt.groupby(['case_id','model']).code.transform('size')
  z,d,g,l=ex.simulate('KRX300','attractiveness',80,start,pd.Timestamp('2022-12-31'),ss,acc,tt,px,tv,actions)
  assert z['status']=='valid',z
  rows.append(dict(scenario=name,budget_krw=budget,**z));events.append(g.assign(scenario=name));daily.append(d.assign(scenario=name))
  print(name,z['sharpe'],z['mean_event_net'],flush=True)
 prior=pd.read_csv(old/'IS_grid.csv');ref=prior[prior.train_end.eq(2022)&prior.index_id.eq('KRX300')&prior.model.eq('attractiveness')&prior.lead.eq(80)].iloc[0]
 assert np.isclose(rows[0]['sharpe'],ref.sharpe)
 newpath=OUT/'rule_margin_audit';newd=pd.read_parquet(newpath/'daily.parquet');newe=pd.read_parquet(newpath/'events.parquet')
 d=newd[newd.index_id.eq('KRX300')&newd.phase.eq('IS_forward')&newd.model.eq('legacy_rules')].sort_values('date')
 e=newe[newe.index_id.eq('KRX300')&newe.phase.eq('IS_forward')&newe.model.eq('legacy_rules')]
 p=pd.read_csv(newpath/'performance.csv');p=p[p.index_id.eq('KRX300')&p.phase.eq('IS_forward')&p.model.eq('legacy_rules')].iloc[0]
 recalculated=d.return_daily.mean()/d.return_daily.std(ddof=1)*np.sqrt(252)
 assert np.isclose(recalculated,p.sharpe)
 rows.append(dict(scenario='current_35features_5events_30eok',**p.to_dict()))
 events.append(e.assign(scenario='current_35features_5events_30eok'));daily.append(d.assign(scenario='current_35features_5events_30eok'))
 pd.DataFrame(rows).to_csv(dest/'bridge.csv',index=False,encoding='utf-8-sig')
 pd.concat(events,ignore_index=True).to_csv(dest/'events.csv',index=False,encoding='utf-8-sig')
 pd.concat(daily,ignore_index=True).to_parquet(dest/'daily.parquet',index=False)
 audit=dict(old_IS_reproduced=True,current_IS_recalculated=True,current_daily_n=len(d),daily_mean=float(d.return_daily.mean()),daily_std=float(d.return_daily.std()),
  current_sharpe=recalculated,current_total_profit=float(e.pnl.sum()),fees=float(e.fees.sum()),
  notes=['old attraction changes candidate set/count using p>=.6, runup and expected-return filter; new classification matches baseline count',
  'old nine-event IS includes warmup basic-rule trades; current five-event IS is common completed-learning evaluation',
  'old same-five replay retains legacy targets and input universe: remaining difference to new is combined model and data-policy difference, not model-only causality',
  'no new parameter tuning, OOS selection or production change'])
 (dest/'checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
 print(pd.DataFrame(rows)[['scenario','n_events','sharpe','mean_event_net','mdd']].to_string(index=False))

if __name__=='__main__':main()
