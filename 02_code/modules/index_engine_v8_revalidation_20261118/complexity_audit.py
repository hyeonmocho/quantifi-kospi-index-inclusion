"""Matched V8 model-complexity audit, frozen leads and historical OOS."""
from research import ROOT,OUT,load
import json,hashlib,itertools
from datetime import datetime,timezone
import pandas as pd,numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
from threadpoolctl import threadpool_limits

DEST=OUT/'complexity_audit'
def main():
 threadpool_limits(1);DEST.mkdir(exist_ok=True)
 full=pd.read_parquet(OUT/'model_panel.parquet')
 fr=pd.read_csv(OUT/'forecasts.csv',parse_dates=['information_date','entry_date','exit_date'])
 targets=pd.read_csv(OUT/'targets.csv')
 choices=pd.read_csv(OUT/'performance_choices.csv').set_index('index_id')
 common=json.loads((OUT/'common_IS_events.json').read_text(encoding='utf8'))
 models=['baseline','frequency_count','calibration_count','logistic_count','boost_count','frequency_p50','calibration_p50','logistic_p50','boost_p50','calibration_p60']
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),fixed_leads=choices.lead.astype(int).to_dict(),models=models,selection='none; historical OOS complexity sensitivity, leads locked to previous IS performance choice',training='same V8 completed prior events capped at 2022; single-variable calibration positive-frequency opportunity subset; original C=1, logit clipping .005-.995; at least 3 positive-frequency past events and two classes, evaluated only on existing V8 >=4-event eligible cases',matched_count='baseline candidate count per case; common projected-cap/code tie break for frequency and calibration; logistic/boost unchanged V8 targets',variable_threshold='0.5 for paired threshold comparison; original calibration 0.6 supplementary',budget=3e9,fee_each_side=.0025,max_entry_sessions=10,hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['model_panel.parquet','forecasts.csv','targets.csv','performance_choices.csv']})
 (DEST/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
 case_rows=[];target_rows=[];fitrows=[];scores=[];cache={}
 existing=['baseline','logistic_count','boost_count','logistic_p50','boost_p50']
 basefr=fr[fr.model.eq('baseline')&fr.phase.isin(['IS_forward','OOS'])]
 for idx,r in choices.iterrows():
  cases=basefr[basefr.index_id.eq(idx)&basefr.lead.eq(r.lead)]
  cases=cases[~cases.phase.eq('IS_forward')|cases.event.isin(common[idx])]
  for row in cases.itertuples():
   g=full[full.case_id.eq(row.case_id)].copy();q=g[g.opportunity].copy()
   train=full[full.index_id.eq(idx)&full.year.le(2022)&full.effective_date.lt(row.information_date)&full.event.ne(row.event)&full.opportunity&full.actual_addition.notna()&full.raw_frequency.gt(0)].copy()
   assert train.event.nunique()>=3 and train.actual_addition.nunique()==2
   key=(idx,tuple(sorted(train.event.unique())))
   def X(s):
    a=s.clip(.005,.995);return np.log(a/(1-a)).to_numpy().reshape(-1,1)
   if key not in cache:
    w=1/train.groupby('event').event.transform('size');w*=len(w)/w.sum()
    cache[key]=LogisticRegression(C=1,solver='lbfgs',max_iter=500).fit(X(train.raw_frequency),train.actual_addition.astype(int),sample_weight=w)
   lr=cache[key];q['calibration']=np.nan;ok=q.raw_frequency.notna()
   q.loc[ok,'calibration']=lr.predict_proba(X(q.loc[ok,'raw_frequency']))[:,1];q.loc[q.raw_frequency.eq(0),'calibration']=0
   fitrows.append(dict(case_id=row.case_id,index_id=idx,event=row.event,phase=row.phase,lead=row.lead,train_events=train.event.nunique(),train_rows=len(train),latest_label=str(train.effective_date.max()),information_date=str(row.information_date),intercept=float(lr.intercept_[0]),slope=float(lr.coef_[0,0])))
   n=int(g.baseline_selected.sum());truth=set(g.loc[g.actual_addition.eq(True),'code'])
   positive=q[q.raw_frequency.gt(0)].copy()
   for model in models:
    if model in existing:
     selected=set(targets.loc[targets.case_id.eq(row.case_id)&targets.model.eq(model),'code'])
    elif model.endswith('_count'):
     col='raw_frequency' if model=='frequency_count' else 'calibration'
     selected=set(positive.sort_values([col,'projected_cap','code'],ascending=[False,False,True]).head(n).code)
    else:
     col='raw_frequency' if model.startswith('frequency_') else 'calibration'
     threshold=.6 if model.endswith('p60') else .5
     selected=set(positive.loc[positive[col].ge(threshold),'code'])
    case_rows.append(dict(case_id=row.case_id,index_id=idx,model=model,lead=int(row.lead),event=row.event,year=row.year,phase=row.phase,information_date=row.information_date,entry_date=row.entry_date,exit_date=row.exit_date,hits=len(selected&truth),predicted=len(selected),actual=row.actual))
    target_rows.extend(dict(case_id=row.case_id,model=model,code=c,weight=1/len(selected)) for c in sorted(selected))
   for z in positive.itertuples():scores.append(dict(case_id=row.case_id,index_id=idx,event=row.event,phase=row.phase,code=z.code,y=int(z.actual_addition),raw=z.raw_frequency,calibration=z.calibration))
 c=pd.DataFrame(case_rows);tt=pd.DataFrame(target_rows);fits=pd.DataFrame(fitrows)
 c.to_csv(DEST/'forecasts.csv',index=False,encoding='utf-8-sig');tt.to_csv(DEST/'targets.csv',index=False,encoding='utf-8-sig');fits.to_csv(DEST/'calibration_fits.csv',index=False,encoding='utf-8-sig')
 accuracy=[]
 for (idx,model,phase),q in c.groupby(['index_id','model','phase']):
  h,n,a=q[['hits','predicted','actual']].sum();accuracy.append(dict(index_id=idx,model=model,phase=phase,n_events=len(q),hits=h,predicted=n,actual=a,precision=h/n if n else np.nan,recall=h/a,f1=2*h/(n+a)))
 pd.DataFrame(accuracy).to_csv(DEST/'accuracy.csv',index=False,encoding='utf-8-sig')
 # Reliability evaluates the same positive-frequency candidates, equal event weights.
 prob=pd.DataFrame(scores);prob.to_parquet(DEST/'probability_rows.parquet',index=False);rel=[]
 for name in ['raw','calibration']:
  prob['brier']=(prob[name]-prob.y)**2
  prob['logloss']=-(prob.y*np.log(prob[name].clip(1e-6,1-1e-6))+(1-prob.y)*np.log(1-prob[name].clip(1e-6,1-1e-6)))
  for (idx,phase),q in prob.groupby(['index_id','phase']):
   a=q.groupby('event')[['brier','logloss']].mean().mean();rel.append(dict(index_id=idx,phase=phase,model=name,n_events=q.event.nunique(),candidate_rows=len(q),**a.to_dict()))
 pd.DataFrame(rel).to_csv(DEST/'calibration_reliability.csv',index=False,encoding='utf-8-sig')
 resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','complexity_resource');resource.write=lambda *a:None
 px,tv,actions=resource.resources()
 from corporate_actions import supplement
 px,tv,actions=supplement(px,tv,actions)
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','complexity_execution');ex.B=3e9
 results=[];days=[];events=[];ledgers=[]
 for (idx,model,phase),q in c.groupby(['index_id','model','phase']):
  lead=int(q.lead.iloc[0]);start=pd.Timestamp(f'{int(q.year.min())}-01-01') if phase=='IS_forward' else pd.Timestamp('2023-01-02');end=pd.Timestamp('2022-12-30') if phase=='IS_forward' else pd.Timestamp('2026-06-30')
  ss=q[['case_id','index_id','event','lead','entry_date','exit_date']];aa=ss.assign(model=model,forecast_available=True)
  z,d,g,l=ex.simulate(idx,model,lead,start,end,ss,aa,tt[tt.model.eq(model)&tt.case_id.isin(ss.case_id)],px,tv,actions)
  meta=dict(index_id=idx,model=model,lead=lead,phase=phase,budget_krw=3e9);results.append(dict(**z,**meta))
  if d is not None:days.append(d.assign(**meta))
  if g is not None:events.append(g.assign(**meta))
  if l is not None and len(l):ledgers.append(l.assign(**meta))
  print(idx,model,phase,z['status'],flush=True)
 pd.DataFrame(results).to_csv(DEST/'performance.csv',index=False,encoding='utf-8-sig')
 for name,items in [('daily',days),('events',events),('ledger',ledgers)]:pd.concat(items,ignore_index=True).to_parquet(DEST/(name+'.parquet'),index=False)
 validate()

def validate():
 p=pd.read_csv(DEST/'performance.csv');d=pd.read_parquet(DEST/'daily.parquet');ev=pd.read_parquet(DEST/'events.parquet');l=pd.read_parquet(DEST/'ledger.parquet');l=l[l.side.isin(['buy','sell'])]
 c=pd.read_csv(DEST/'forecasts.csv');t=pd.read_csv(DEST/'targets.csv');fits=pd.read_csv(DEST/'calibration_fits.csv');protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf8'))
 keys=['index_id','model','phase','lead'];s=l.groupby(keys+['date','code']);pnl=ev.groupby(keys).pnl.sum();last=d.sort_values('date').groupby(keys).equity.last()
 panel=pd.read_parquet(OUT/'model_panel.parquet');gate=panel[['case_id','code','opportunity']];joined=t.merge(gate,on=['case_id','code'],validate='many_to_one')
 checks=dict(all_valid=bool(p.status.eq('valid').all()),cash_nonnegative=bool(d.cash.ge(-1e-4).all()),fees=bool(np.allclose(l.fee,l.notional*.0025)),liquidity=bool((s.notional.sum()<=s.limit.first()+1e-4).all()),PNL_matches_NAV=bool(np.allclose(last.reindex(pnl.index)-3e9,pnl,atol=.001)),no_future_labels=bool((pd.to_datetime(fits.latest_label)<pd.to_datetime(fits.information_date)).all()),all_labels_IS=bool(pd.to_datetime(fits.latest_label).dt.year.le(2022).all()),all_targets_candidate_eligible=bool(joined.opportunity.eq(True).all()),source_inputs_unchanged=all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in protocol['hashes'].items()))
 # Exact equality for previously published selected model at its fixed lead.
 old=pd.read_csv(OUT/'portfolio_roles.csv');old=old[old.role.eq('IS_performance_selected')]
 delta=old.merge(p,on=['index_id','model','lead','phase'],suffixes=('_old','_new'));checks['published_selected_results_reproduced']=bool(len(delta)==6 and np.allclose(delta.sharpe_old,delta.sharpe_new))
 (DEST/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');assert all(checks.values()),checks
 rng=np.random.default_rng(18112026);pairs=[]
 for idx,q in ev[ev.phase.eq('OOS')].groupby('index_id'):
  w=q.pivot(index='event',columns='model',values='net_return')
  for model in w.columns:
   if model=='baseline':continue
   z=(w[model]-w.baseline).dropna();n=len(z);yrs=z.index.str[:4];agg=pd.Series(z.to_numpy()-z.mean(),index=yrs).groupby(level=0).sum();k=len(agg);se=np.sqrt(k/(k-1)*(agg**2).sum()/n**2) if k>1 else np.nan;tval=z.mean()/se if se>0 else np.nan
   flips=np.array([np.mean(z.to_numpy()*s) for s in itertools.product([-1,1],repeat=n)])
   ci=np.quantile(z.to_numpy()[rng.integers(0,n,(10000,n))].mean(axis=1),[.025,.975])
   pairs.append(dict(index_id=idx,model=model,n_events=n,mean_net_difference=z.mean(),better_events=int(z.gt(1e-12).sum()),year_clusters=k,year_cluster_t=tval,year_cluster_p=2*stats.t.sf(abs(tval),k-1),paired_signflip_p=float((np.abs(flips)>=abs(z.mean())-1e-12).mean()),review_bootstrap_low=ci[0],review_bootstrap_high=ci[1],note='exploratory reused OOS; signflip assumes symmetry; review bootstrap not year-dependence-adjusted; multiplicity unadjusted'))
 pd.DataFrame(pairs).to_csv(DEST/'paired_improvement.csv',index=False,encoding='utf-8-sig')
 print(p[p.phase.eq('OOS')][['index_id','model','lead','mean_event_net','sharpe','mdd']].to_string(index=False),flush=True)

if __name__=='__main__':main()
