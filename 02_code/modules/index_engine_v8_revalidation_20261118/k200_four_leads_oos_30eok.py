"""Two independently selected presentation axes; original standalone IS only."""
from select_standalone_balanced import ROOT,OUT,load,SPLIT,ML,xfreq,fit
from pathlib import Path
SOURCE=ROOT/'04_results/two_axis_selection_20261118'
DEST=ROOT/'04_results/k200_oos_20_40_60_80_20261118';DEST.mkdir(exist_ok=True);DEST.mkdir(exist_ok=True)
import pandas as pd,numpy as np,json,sys,hashlib
from datetime import datetime,timezone
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

OLD=ROOT/'04_results/standalone_balanced_selection_20261118'
def save(n,x):pd.DataFrame(x).to_csv(DEST/(n+'.csv'),index=False,encoding='utf-8-sig')
def dump(n,x):(DEST/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def main():
 threadpool_limits(1)
 choices=[dict(index_id='KOSPI200',axis='performance',model='shrunk_trend__count',family='shrunk_trend',policy='count',lead=lead) for lead in [20,40,60,80]]
 full=pd.read_parquet(SOURCE/'model_panel.parquet')
 # All IS rows are byte-value equivalent to the preceding run. Only an OOS row group was restored.
 oldpanel=pd.read_parquet(OUT/'rule_margin_audit/model_panel.parquet')
 pd.testing.assert_frame_equal(full[full.case_id.isin(oldpanel.case_id)][oldpanel.columns].reset_index(drop=True),oldpanel,check_dtype=False)
 det={}
 for idx,path in {'KOSPI200':'index_two_axes_20261117/k200_long/targets.csv','KOSPI100':'k100_long_20261117/targets.csv','KRX300':'index_two_axes_20261117/three_indices/targets.csv'}.items():
  a=pd.read_csv(ROOT/'04_results'/path,low_memory=False)
  if 'train_end' in a:a=a[a.train_end.eq(SPLIT[idx])]
  for (cid,m),g in a[a.index_id.eq(idx)&a.model.isin(['ma20','shrunk_trend'])].groupby(['case_id','model']):det[(cid,m)]=set(g.code)
 res=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','twoaxis_resources');res.write=lambda *a:None
 px,tv,actions=res.resources()
 from corporate_actions import supplement
 px,tv,actions=supplement(px,tv,actions)
 full['edge_label']=np.nan
 for cid,g in full.groupby('case_id'):
  r=g.iloc[0]
  if r.entry_date not in px.index or r.exit_date not in px.index:continue
  rr=px.loc[r.exit_date].reindex(g.code).to_numpy()/px.loc[r.entry_date].reindex(g.code).to_numpy()-1
  z=pd.Series(rr,index=g.index);full.loc[g.index,'edge_label']=z-z.groupby(g.group).transform('mean')
 denom=pd.read_csv(ROOT/'04_results/index_engine_v6_20261118/candidate_coverage.csv').set_index('case_id').actual.to_dict()
 denom.update(pd.read_csv(SOURCE/'repaired_denominator.csv').set_index('case_id').actual.to_dict())
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','twoaxis_execution');ex.B=3e9
 metrics=[];allfr=[];alltargets=[];days=[];evs=[];ledgers=[];audits=[];cache={}
 oldperf=pd.read_csv(OLD/'performance.csv');oldfr=pd.read_csv(OLD/'forecasts.csv');olde=pd.read_parquet(OLD/'events.parquet');oldd=pd.read_parquet(OLD/'daily.parquet')
 for pick in choices:
  idx=pick['index_id'];axis=pick['axis'];family=pick['family'];model=pick['model'];lead=pick['lead'];policy=pick['policy']
  sel=lambda a:a[a.index_id.eq(idx)&a.model.eq(model)&a.lead.eq(lead)&a.phase.eq('IS')]
  ff=sel(oldfr).copy(); tt0=pd.read_csv(OLD/'targets.csv');tt0=tt0[tt0.model.eq(model)&tt0.case_id.isin(ff.case_id)]
  ss0=ff[['case_id','index_id','event','lead','entry_date','exit_date']].copy()
  for c in ['entry_date','exit_date']:ss0[c]=pd.to_datetime(ss0[c])
  zz,dd,ee,ll=ex.simulate(idx,model,lead,pd.Timestamp('2002-01-01' if idx=='KOSPI200' else '2006-01-01'),pd.Timestamp('2016-12-31'),ss0,ss0.assign(model=model,forecast_available=True),tt0,px,tv,actions)
  tag=dict(index_id=idx,axis=axis,phase='IS',model=model,lead=lead,budget_eok=30)
  metrics.append(dict(**zz,**tag));days.append(dd.assign(**tag));evs.append(ee.assign(**tag));ledgers.append(ll.assign(**tag));allfr.extend(ff.assign(axis=axis).to_dict('records'))
  used=ll[ll.side.isin(['buy','sell'])].groupby(['date','code']).agg(used=('notional','sum'),limit=('limit','min'));assert used.used.le(used.limit+.1).all() and dd.cash.ge(-.01).all() and np.isclose(ee.pnl.sum(),dd.equity.iloc[-1]-ex.B,atol=.1)
  rows=[];targets=[]
  cases=full[full.index_id.eq(idx)&full.year.gt(SPLIT[idx])&full.lead.eq(lead)]
  assert cases.event.nunique()==full.loc[full.index_id.eq(idx)&full.year.gt(SPLIT[idx]),'event'].nunique(),(idx,lead,'missing case')
  for cid,g in cases.groupby('case_id'):
   r=g.iloc[0];q=g[g.opportunity];base=set(q.loc[q.baseline_selected.eq(1),'code']);truth=set(g.loc[g.actual_addition.eq(1),'code'])
   p=pd.Series(np.nan,index=q.index)
   if family=='baseline':names=base
   elif family in ['ma20','shrunk_trend']:
    assert (cid,family) in det or not base,(cid,family,'missing deterministic forecast')
    names=det.get((cid,family),set())&set(q.code)
   else:
    tr=full[full.index_id.eq(idx)&full.year.le(SPLIT[idx])&full.effective_date.lt(r.information_date)&full.event.ne(r.event)&full.opportunity&full.actual_addition.notna()]
    assert tr.event.nunique()>=4 and tr.effective_date.max()<r.information_date
    key=(idx,family,tuple(sorted(tr.event.unique())))
    if key not in cache:
     if family=='calibration':
      tt=tr[tr.raw_frequency.gt(0)];w=1/tt.groupby('event').event.transform('size');w*=len(w)/w.sum()
      cache[key]=LogisticRegression(C=1,max_iter=500).fit(xfreq(tt.raw_frequency),tt.actual_addition.astype(int),sample_weight=w)
     else:cache[key]=fit(tr,ML[family])
    if family=='calibration':
     ok=q.raw_frequency.notna();p.loc[ok]=cache[key].predict_proba(xfreq(q.loc[ok,'raw_frequency']))[:,1];p.loc[q.raw_frequency.eq(0)]=0
    else:p[:]=cache[key].predict_proba(q[ML[family]])[:,1]
    names=set(q.assign(p=p).sort_values(['p','projected_cap','code'],ascending=[False,False,True]).head(len(base)).code)
    rt=tr[tr.exit_date.lt(r.information_date)&tr.edge_label.notna()&tr.raw_frequency.gt(0)]
    means={}
    for y in [0,1]:
     z=rt[rt.actual_addition.eq(y)].groupby('event').edge_label.mean();means[y]=z.mean()*len(z)/(len(z)+5) if len(z) else 0.
    audits.append(dict(index_id=idx,axis=axis,case_id=cid,train_events=tr.event.nunique(),train_end=SPLIT[idx],max_label_date=tr.effective_date.max(),max_return_date=rt.exit_date.max(),information_date=r.information_date))
   trade=names
   if policy=='attractiveness':
    edge=p*means[1]+(1-p)*means[0]-.005
    trade=set(q.loc[q.raw_frequency.gt(0)&p.ge(.6)&q.relative_momentum20.le(.30)&edge.gt(0),'code'])
   rows.append(dict(index_id=idx,axis=axis,phase='OOS',model=model,family=family,policy=policy,lead=lead,case_id=cid,event=r.event,year=r.year,information_date=r.information_date,warmup=False,return_fallback=False,entry_date=r.entry_date,exit_date=r.exit_date,predicted=len(names),hits=len(names&truth),actual=denom[cid],trade_count=len(trade),trade_hits=len(trade&truth)))
   targets.extend(dict(index_id=idx,axis=axis,case_id=cid,model=model,code=c,weight=1/len(trade)) for c in sorted(trade))
  fr=pd.DataFrame(rows);tt=pd.DataFrame(targets,columns=['index_id','axis','case_id','model','code','weight'])
  ss=fr[['case_id','index_id','event','lead','entry_date','exit_date']];aa=ss.assign(model=model,forecast_available=True)
  z,d,e,l=ex.simulate(idx,model,lead,pd.Timestamp(f'{SPLIT[idx]+1}-01-01'),pd.Timestamp('2026-06-30'),ss,aa,tt,px,tv,actions)
  h,n,a=fr[['hits','predicted','actual']].sum();th,tn=fr[['trade_hits','trade_count']].sum()
  info=dict(index_id=idx,axis=axis,phase='OOS',model=model,lead=lead,budget_eok=30)
  used=l[l.side.isin(['buy','sell'])].groupby(['date','code']).agg(used=('notional','sum'),limit=('limit','min'));assert used.used.le(used.limit+.1).all() and d.cash.ge(-.01).all() and np.isclose(e.pnl.sum(),d.equity.iloc[-1]-ex.B,atol=.1)
  metrics.append(dict(**z,**info,precision=h/n,recall=h/a,f1=2*h/(n+a),hits=h,predicted=n,actual=a,trade_targets=tn,trade_precision=th/tn if tn else np.nan,trade_recall=th/a))
  if d is not None:days.append(d.assign(**info));evs.append(e.assign(**info));ledgers.append(l.assign(**info))
  allfr+=rows;alltargets+=targets
  if lead in [20,40]:
   prior=pd.read_csv(SOURCE/'performance.csv');pr=prior[(prior.index_id==idx)&(prior['axis'].eq('prediction' if lead==20 else 'performance'))&(prior.phase=='OOS')&(prior.lead==lead)].iloc[0]
   for k in ['cagr','sharpe','mdd','mean_event_net']:assert np.isclose(z[k],pr[k],atol=1e-8),(lead,k,z[k],pr[k])
  print('OOS',idx,axis,model,lead,z,flush=True)
 print(pd.DataFrame(metrics)[['index_id','phase','lead','cagr','sharpe','mdd']].to_string(index=False),flush=True)
 save('performance',metrics);save('forecasts',allfr);save('targets',alltargets);save('fit_audit',audits)
 for n,parts in [('daily',days),('events',evs),('ledger',ledgers)]:pd.concat(parts,ignore_index=True).to_parquet(DEST/(n+'.parquet'),index=False)

if __name__=='__main__':main()
