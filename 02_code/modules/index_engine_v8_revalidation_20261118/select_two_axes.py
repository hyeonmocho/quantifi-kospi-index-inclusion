"""Two independently selected presentation axes; original standalone IS only."""
from select_standalone_balanced import ROOT,OUT,load,SPLIT,ML,xfreq,fit
from repair_partial_k100 import DEST
import pandas as pd,numpy as np,json,sys,hashlib
from datetime import datetime,timezone
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

OLD=ROOT/'04_results/standalone_balanced_selection_20261118'
def save(n,x):pd.DataFrame(x).to_csv(DEST/(n+'.csv'),index=False,encoding='utf-8-sig')
def dump(n,x):(DEST/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def main():
 threadpool_limits(1)
 choices=[]
 for idx in SPLIT:
  d=pd.read_csv(OLD/(idx+'_IS_candidates.csv'))
  for axis in ['prediction','performance']:
   q=d[d.model.str.endswith('__count')] if axis=='prediction' else d[d.status.eq('valid')&d.sharpe.notna()]
   q=q.sort_values(['f1','precision','features','lead','model'],ascending=[False,False,True,True,True]) if axis=='prediction' else q.sort_values(['sharpe','f1','features','lead','model'],ascending=[False,False,True,True,True])
   b=q.iloc[0];family,policy=b.model.split('__')
   choices.append(dict(index_id=idx,axis=axis,model=b.model,family=family,policy=policy,lead=int(b.lead),IS_F1=b.f1,IS_sharpe=b.sharpe,IS_mean_net=b.mean_event_net,IS_n=int(b.n_events)))
 save('choices',choices)
 dump('selection_freeze',dict(created_utc=datetime.now(timezone.utc).isoformat(),choices=choices,
  criterion_prediction='maximum standalone IS F1, then precision, fewer features, shorter lead',
  criterion_performance='maximum standalone IS daily account Sharpe over all valid candidates; NO F1 or mean-return floor; ties F1/simplicity/lead',
  original_standalone_IS_end=SPLIT,IS_unchanged=True,IS_results_source=str(OLD),
  OOS_not_used_for_this_selection=True,historical_OOS_previously_viewed=True,
  incomplete_roster='user permits selected 99 without fictional fill; fixed full simulation denominator; not an official full roster',
  broker='same information date refresh of IS-selected prediction family, complete named roster only; report date proxy, no verified public timestamp',
  live_order_allowed=False))
 full=pd.read_parquet(DEST/'model_panel.parquet')
 # All IS rows are byte-value equivalent to the preceding run. Only an OOS row group was restored.
 oldpanel=pd.read_parquet(OUT/'rule_margin_audit/model_panel.parquet')
 pd.testing.assert_frame_equal(full[full.case_id.isin(oldpanel.case_id)][oldpanel.columns].reset_index(drop=True),oldpanel,check_dtype=False)
 sys.path.append(str(ROOT/'02_code/modules/k100_long_20261117'))
 build=load('02_code/modules/k100_long_20261117/three_build.py','axis_builder');models,mm=build.old.setup()
 from dataclasses import replace
 models.config=replace(models.config,allow_underfilled_k100=True)
 def deterministic(obj,family):
  d=obj['panel'];r=d.iloc[0];f=obj['frame'];d=d.set_index('code',drop=False).reindex(f.index)
  selector,_=models._selector(r.index_id,r.event,r.information_date,f,obj['baseline'])
  logratio=d.log_ma20_ratio.fillna(0).clip(-.4,.4).to_numpy() if family=='ma20' else (.25*np.log1p(d.momentum20.clip(lower=-.99))*d.remaining/40).fillna(0).clip(-.4,.4).to_numpy()
  cap=(d.past_sum.to_numpy()+d.remaining.to_numpy()*d.current_cap.to_numpy()*np.exp(logratio))/d.full_days.to_numpy()
  cap=np.where(np.isfinite(cap)&(cap>0)&d.eligibility.eq('pass'),cap,d.projected_cap)
  return set(selector(cap))
 det={}
 for idx,path in {'KOSPI200':'index_two_axes_20261117/k200_long/targets.csv','KOSPI100':'k100_long_20261117/targets.csv','KRX300':'index_two_axes_20261117/three_indices/targets.csv'}.items():
  a=pd.read_csv(ROOT/'04_results'/path,low_memory=False)
  if 'train_end' in a:a=a[a.train_end.eq(SPLIT[idx])]
  for (cid,m),g in a[a.index_id.eq(idx)&a.model.isin(['ma20','shrunk_trend'])].groupby(['case_id','model']):det[(cid,m)]=set(g.code)
 repaired=pd.read_pickle(DEST/'frames/KOSPI100_2024-06_L20.pkl')
 for m in ['ma20','shrunk_trend']:det[('KOSPI100_2024-06_L20',m)]=deterministic(repaired,m)
 dump('repaired_deterministic_targets',{m:sorted(det[('KOSPI100_2024-06_L20',m)]) for m in ['ma20','shrunk_trend']})
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
 denom.update(pd.read_csv(DEST/'repaired_denominator.csv').set_index('case_id').actual.to_dict())
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','twoaxis_execution');ex.B=3e9
 metrics=[];allfr=[];alltargets=[];days=[];evs=[];ledgers=[];audits=[];cache={}
 oldperf=pd.read_csv(OLD/'performance.csv');oldfr=pd.read_csv(OLD/'forecasts.csv');olde=pd.read_parquet(OLD/'events.parquet');oldd=pd.read_parquet(OLD/'daily.parquet')
 for pick in choices:
  idx=pick['index_id'];axis=pick['axis'];family=pick['family'];model=pick['model'];lead=pick['lead'];policy=pick['policy']
  sel=lambda a:a[a.index_id.eq(idx)&a.model.eq(model)&a.lead.eq(lead)&a.phase.eq('IS')]
  metrics.extend(sel(oldperf).assign(axis=axis).to_dict('records'));days.append(sel(oldd).assign(axis=axis));evs.append(sel(olde).assign(axis=axis));allfr.extend(sel(oldfr).assign(axis=axis).to_dict('records'))
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
  info=dict(index_id=idx,axis=axis,phase='OOS',model=model,lead=lead)
  metrics.append(dict(**z,**info,precision=h/n,recall=h/a,f1=2*h/(n+a),hits=h,predicted=n,actual=a,trade_targets=tn,trade_precision=th/tn if tn else np.nan,trade_recall=th/a))
  if d is not None:days.append(d.assign(**info));evs.append(e.assign(**info));ledgers.append(l.assign(**info))
  allfr+=rows;alltargets+=targets
  print('OOS',idx,axis,model,lead,z,flush=True)
 save('performance',metrics);save('forecasts',allfr);save('targets',alltargets);save('fit_audit',audits)
 for n,parts in [('daily',days),('events',evs),('ledger',ledgers)]:pd.concat(parts,ignore_index=True).to_parquet(DEST/(n+'.parquet'),index=False)
 # Same-date broker forecasts for the selected prediction family. No other family wins on these reports.
 from candidates import enrich,dated_roster
 broot=ROOT/'04_results/index_two_axes_20261117/broker';bpanel=pd.read_parquet(broot/'panel.parquet');reports=pd.read_csv(broot/'reports.csv')
 rosters=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/comparison/target_rosters.csv')
 evidence=pd.read_parquet(ROOT/'04_results/index_financial_gap_repair_20261118/financial_receipt_evidence.parquet')
 br=[];ba=[]
 for report in reports.itertuples():
  cid='broker_'+report.report_id;path=broot/'frames'/(cid+'.pkl')
  if not path.exists():ba.append(dict(report_id=report.report_id,index_id=report.index_id,status='missing_same_date_frame'));continue
  obj=pd.read_pickle(path);g=obj['panel'];r=g.iloc[0];idx=r.index_id
  pick=next(x for x in choices if x['index_id']==idx and x['axis']=='prediction')
  if r.year<=SPLIT[idx]:ba.append(dict(report_id=report.report_id,index_id=idx,status='not_OOS'));continue
  assert pick['family'] in ['ma20','shrunk_trend'] # present matched frames only cover KOSPI200
  try:ns=deterministic(obj,pick['family'])
  except ValueError as err:ba.append(dict(report_id=report.report_id,index_id=idx,status=str(err)));continue
  f=obj['frame'];current=set(f.index[f.cur]);z=enrich(g,f,idx,r.information_date,current,r.information_date,len(current)>0,evidence)
  ns &= set(z.loc[z.monitor_candidate,'code'])
  bs=set(rosters.loc[rosters.report_id.eq(report.report_id)&rosters.who.eq('broker')&rosters.design_id.eq('available_date_refresh'),'code'])
  truth=set(models.engines.truth(idx,r.event));ba.append(dict(report_id=report.report_id,index_id=idx,status='paired',event=r.event))
  for who,names in [('engine',ns),('broker',bs)]:
   ss=pd.DataFrame([dict(index_id=idx,event=r.event,case_id=cid,lead=0,entry_date=r.entry_date,exit_date=r.exit_date)])
   tt=pd.DataFrame([dict(case_id=cid,model='broker_pair',code=c,weight=1/len(names)) for c in names],columns=['case_id','model','code','weight'])
   zz,dd,ee,ll=ex.simulate(idx,'broker_pair',0,r.entry_date,min(pd.Timestamp('2026-06-30'),r.exit_date+pd.Timedelta(days=120)),ss,ss.assign(model='broker_pair',forecast_available=True),tt,px,tv,actions)
   br.append(dict(index_id=idx,report_id=report.report_id,event=r.event,who=who,model=pick['model'] if who=='engine' else 'published_report',information_date=r.information_date,entry_date=r.entry_date,hits=len(names&truth),predicted=len(names),actual=len(truth),status=zz['status'],net_return=ee.net_return.sum() if ee is not None else np.nan,names=';'.join(sorted(names))))
 save('broker_pairs',br);save('broker_audit',ba)
 b=pd.DataFrame(br);s=[]
 for (idx,who),g in b.groupby(['index_id','who']):
  # Equal-review averages stop reviews with many reports dominating accuracy.
  h,n,a=g.groupby('event')[['hits','predicted','actual']].mean().sum()
  good=set(b.groupby('report_id').filter(lambda q:len(q)==2 and q.status.eq('valid').all()).report_id)
  x=g[g.report_id.isin(good)].groupby('event').net_return.mean();pos=x[x>0];neg=x[x<0]
  s.append(dict(index_id=idx,who=who,n_reports=len(g),n_events=g.event.nunique(),precision=h/n,recall=h/a,f1=2*h/(n+a),mean_net=x.mean(),win_rate=x.gt(0).mean(),payoff=pos.mean()/abs(neg.mean()) if len(pos) and len(neg) else np.nan))
 save('broker_summary',s)
 a=pd.DataFrame(audits);d=pd.concat(days);ll=pd.concat(ledgers);ll=ll[ll.side.isin(['buy','sell'])]
 keys=['index_id','axis','phase','model','lead'];ee=pd.concat(evs);pnl=ee.groupby(keys).pnl.sum();end=d.sort_values('date').groupby(keys).equity.last();by=ll.groupby(keys+['date','code'])
 checks=dict(IS_reused_unchanged=True,all_selected_OOS_valid=all(r['status']=='valid' for r in metrics),
  past_labels=bool((pd.to_datetime(a.max_label_date)<pd.to_datetime(a.information_date)).all()),
  training_capped=bool(pd.to_datetime(a.max_label_date).dt.year.le(a.train_end).all()),
  past_returns=bool((pd.to_datetime(a.max_return_date)<pd.to_datetime(a.information_date)).all()),
  nonnegative_cash=bool(d.cash.ge(-1e-4).all()),fees=bool(np.allclose(ll.fee,ll.notional*.0025)),
  daily_liquidity=bool((by.notional.sum()<=by.limit.first()+1e-4).all()),PNL_NAV=bool(np.allclose(end.reindex(pnl.index)-3e9,pnl,atol=.01)))
 dump('checks',checks);assert all(checks.values()),checks
 print(pd.DataFrame(s).to_string(index=False),flush=True)
if __name__=='__main__':main()
