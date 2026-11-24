"""One presentation candidate per index, selected on its original standalone IS.

Forecast quality is evaluated on the forecast roster; execution policy and its
smaller traded roster are separately disclosed. No OOS winner search.
"""
from research import ROOT,OUT,load,FEATURES
import os
from rule_margin_audit import CORE,RULE,fit
import pandas as pd,numpy as np,json,hashlib
from datetime import datetime,timezone
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

DEST=ROOT/'04_results/standalone_balanced_selection_20261118'
SPLIT={'KOSPI200':2016,'KOSPI100':2016,'KRX300':2022}
START={'KOSPI200':'2002-01-01','KOSPI100':'2006-01-01','KRX300':'2018-07-01'}
ML={'legacy31':FEATURES,'compact7':CORE+RULE,'augmented35':FEATURES+RULE}
FAMILIES=['baseline','ma20','shrunk_trend','calibration',*ML]
POLICIES={m:(['count','attractiveness'] if m in ['calibration',*ML] else ['count']) for m in FAMILIES}
def dump(n,o):(DEST/(n+'.json')).write_text(json.dumps(o,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def save(n,o):pd.DataFrame(o).to_csv(DEST/(n+'.csv'),index=False,encoding='utf-8-sig')
def xfreq(s):
 a=s.clip(.005,.995);return np.log(a/(1-a)).to_numpy().reshape(-1,1)

def main():
 if os.environ.get('ALLOW_LEGACY_INDEX_RULE_PANEL')!='1':
  raise RuntimeError('Frozen 2026-11-18 model_panel predates the dated-rule/TMI-parent correction. Rebuild candidates and scenario frequencies before refitting; set ALLOW_LEGACY_INDEX_RULE_PANEL=1 only to reproduce the archived result.')
 threadpool_limits(1);DEST.mkdir(exist_ok=True)
 full=pd.read_parquet(OUT/'rule_margin_audit/model_panel.parquet')
 original_cohort=pd.read_csv(ROOT/'04_results/index_two_axes_20261117/k200_long/cohort.csv')
 k100=json.loads((ROOT/'04_results/k100_long_20261117/cohort_audit.json').read_text(encoding='utf8'))[0]
 cohorts={'KOSPI200':sorted(original_cohort.loc[original_cohort.in_common_cohort & original_cohort.year.le(2016),'event']),
          'KOSPI100':sorted(e for e in k100['included'] if e[:4]<='2016'),
          'KRX300':sorted(full.loc[full.index_id.eq('KRX300')&full.year.le(2022),'event'].unique())}
 dump('protocol',dict(created_utc=datetime.now(timezone.utc).isoformat(),standalone_IS_end=SPLIT,IS_start=START,IS_events=cohorts,
      leads=[20,40,60,80],families=FAMILIES,policies=POLICIES,C=.1,calibration_C=1,min_training_events=4,
      primary_selection='engine forecast F1 >= same-lead basic rule F1 and mean net event return>0; maximize IS daily account Sharpe; ties F1, fewer raw features, shorter lead, model name',
      engine_vs_trade='engine top-N forecast metrics and actual filtered trade metrics both reported; never label trade precision as full-engine precision',
      warmup='baseline rule forecast and execution until four completed events; explicitly tagged, not evidence of learned-model skill',
      attractiveness='p>=0.6, raw scenario frequency>0, relative momentum20<=0.30, shrunk prior hit/miss sector-excess expected return minus 0.005>0; if <3 completed return events use baseline policy and tag fallback',
      capital=3e9,cost_each_side=.0025,entry_sessions=10,liquidity='min(10% lag ADV20,10% actual turnover), exits until fully liquidated',
      OOS='K200/K100 2017-01 onward; KRX300 2023-01 onward; through2026-06-30; repeatedly viewed historical period',
      missing='An unavailable selected OOS forecast remains missing, never cash. Original IS cohort held fixed; invalid execution disqualifies model, not event.',
      models_not_entered=['boosting per user simplification preference','unverified new parameter grids'],
      limitations=['historical K100 buffer and KRX BMI/TMI rule gaps persist','new retrospective selection, not untouched confirmation','not a claim to beat brokers'],
      source_hash=hashlib.sha256((OUT/'rule_margin_audit/model_panel.parquet').read_bytes()).hexdigest()))
 resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','balanced_resources');resource.write=lambda *a:None
 px,tv,actions=resource.resources()
 from corporate_actions import supplement
 px,tv,actions=supplement(px,tv,actions)
 # Return targets are made available only after the event exit AND effective date.
 # Same-period sector average is the requested comparator, not matched stocks.
 full['edge_label']=np.nan
 for cid,g in full.groupby('case_id'):
  r=g.iloc[0]
  if r.entry_date not in px.index or r.exit_date not in px.index:continue
  ret=px.loc[r.exit_date].reindex(g.code).to_numpy()/px.loc[r.entry_date].reindex(g.code).to_numpy()-1
  v=pd.Series(ret,index=g.index);full.loc[g.index,'edge_label']=v-v.groupby(g.group).transform('mean')
 # Previously computed deterministic MA/trend forecasts; no learned future coefficients.
 determ={}
 sources={'KOSPI200':'index_two_axes_20261117/k200_long/targets.csv','KOSPI100':'k100_long_20261117/targets.csv','KRX300':'index_two_axes_20261117/three_indices/targets.csv'}
 for idx,path in sources.items():
  a=pd.read_csv(ROOT/'04_results'/path,low_memory=False)
  if 'train_end' in a:a=a[a.train_end.eq(SPLIT[idx])]
  for (cid,m),g in a[a.index_id.eq(idx)&a.model.isin(['ma20','shrunk_trend'])].groupby(['case_id','model']):determ[(cid,m)]=set(g.code)
 denom=pd.read_csv(ROOT/'04_results/index_engine_v6_20261118/candidate_coverage.csv').set_index('case_id').actual.to_dict()
 forecast=[];targets=[];audits=[];probs=[];cache={}
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','balanced_execution');ex.B=3e9
 performance=[];events=[];daily=[];ledger=[];choices=[];missing=[]
 for phase in ['IS','OOS']:
  for idx in SPLIT:
   iset=full[full.index_id.eq(idx)]
   if phase=='IS':cases=iset[iset.event.isin(cohorts[idx])]
   else:
    pick=next(c for c in choices if c['index_id']==idx)
    if pick.get('status')!='selected':continue
    cases=iset[iset.year.gt(SPLIT[idx]) & iset.lead.eq(pick['lead'])]
    # Report missing selected-lead cases relative to all observed event labels.
    wanted=set(iset.loc[iset.year.gt(SPLIT[idx]),'event'])
    missing.extend(dict(index_id=idx,event=e,lead=pick['lead'],reason='selected_lead_forecast_unavailable') for e in sorted(wanted-set(cases.event)))
   for i,(cid,g) in enumerate(cases.groupby('case_id',sort=True)):
    r=g.iloc[0];q=g[g.opportunity].copy();base=set(q.loc[q.baseline_selected.eq(1),'code']);n=len(base)
    tr=full[full.index_id.eq(idx)&full.year.le(SPLIT[idx])&full.effective_date.lt(r.information_date)&full.event.ne(r.event)&full.opportunity&full.actual_addition.notna()]
    ready=tr.event.nunique()>=4 and tr.actual_addition.nunique()==2
    rt=tr[tr.exit_date.lt(r.information_date)&tr.edge_label.notna()&tr.raw_frequency.gt(0)]
    means={}
    for label in [0,1]:
     z=rt[rt.actual_addition.eq(label)].groupby('event').edge_label.mean();means[label]=z.mean()*len(z)/(len(z)+5) if len(z) else 0.
    er_ready=rt.event.nunique()>=3
    families=FAMILIES if phase=='IS' else [pick['family']]
    truth=set(g.loc[g.actual_addition.eq(1),'code'])
    for family in families:
     warmup=family not in ['baseline','ma20','shrunk_trend'] and not ready
     p=pd.Series(np.nan,index=q.index)
     if family=='baseline' or warmup:selected=base
     elif family in ['ma20','shrunk_trend']:
      if (cid,family) not in determ:
       # An empty deterministic target list is distinguished from missing only
       # where baseline count is zero; otherwise fail closed for this candidate.
       if n:missing.append(dict(index_id=idx,event=r.event,lead=r.lead,model=family,phase=phase,reason='deterministic_target_cache_missing'));continue
      selected=determ.get((cid,family),set()) & set(q.code)
     else:
      key=(idx,family,tuple(sorted(tr.event.unique())))
      if key not in cache:
       if family=='calibration':
        tt=tr[tr.raw_frequency.gt(0)].copy();w=1/tt.groupby('event').event.transform('size');w*=len(w)/w.sum()
        cache[key]=LogisticRegression(C=1,max_iter=500).fit(xfreq(tt.raw_frequency),tt.actual_addition.astype(int),sample_weight=w)
       else:cache[key]=fit(tr,ML[family])
      mdl=cache[key]
      if family=='calibration':
       ok=q.raw_frequency.notna();p.loc[ok]=mdl.predict_proba(xfreq(q.loc[ok,'raw_frequency']))[:,1];p.loc[q.raw_frequency.eq(0)]=0
      else:p[:]=mdl.predict_proba(q[ML[family]])[:,1]
      selected=set(q.assign(p=p).sort_values(['p','projected_cap','code'],ascending=[False,False,True]).head(n).code)
      audits.append(dict(index_id=idx,case_id=cid,family=family,phase=phase,train_events=tr.event.nunique(),train_end=SPLIT[idx],max_label_date=tr.effective_date.max(),information_date=r.information_date,return_max_exit=rt.exit_date.max()))
      probs.extend(dict(index_id=idx,case_id=cid,family=family,phase=phase,code=c,p=v,y=y) for c,v,y in zip(q.code,p,q.actual_addition))
     policies=POLICIES[family] if phase=='IS' else [pick['policy']]
     for policy in policies:
      model=family+'__'+policy;fallback=False
      if policy=='count':trade=selected
      elif warmup or not er_ready:trade=base;fallback=True
      else:
       edge=p*means[1]+(1-p)*means[0]-.005
       trade=set(q.loc[q.raw_frequency.gt(0)&p.ge(.6)&q.relative_momentum20.le(.30)&edge.gt(0),'code'])
      forecast.append(dict(index_id=idx,case_id=cid,event=r.event,year=r.year,lead=int(r.lead),phase=phase,model=model,family=family,policy=policy,
         predicted=len(selected),hits=len(selected&truth),actual=denom[cid],trade_count=len(trade),trade_hits=len(trade&truth),warmup=warmup,return_fallback=fallback,
         information_date=r.information_date,entry_date=r.entry_date,exit_date=r.exit_date))
      targets.extend(dict(case_id=cid,model=model,code=c,weight=1/len(trade)) for c in sorted(trade))
    if i%12==0:print('forecast',phase,idx,i,flush=True)
   fr=pd.DataFrame(forecast);tt=pd.DataFrame(targets)
   sub=fr[fr.index_id.eq(idx)&fr.phase.eq(phase)]
   for (model,lead),gg in sub.groupby(['model','lead']):
    expected=len(cohorts[idx]) if phase=='IS' else iset[iset.year.gt(SPLIT[idx])].event.nunique()
    if gg.event.nunique()!=expected:
     performance.append(dict(index_id=idx,phase=phase,model=model,lead=int(lead),status='incomplete_forecast_cohort',n_events=gg.event.nunique(),expected_events=expected));continue
    ss=gg[['case_id','index_id','event','lead','entry_date','exit_date']];aa=ss.assign(model=model,forecast_available=True)
    start=pd.Timestamp(START[idx]) if phase=='IS' else pd.Timestamp(f'{SPLIT[idx]+1}-01-01');end=pd.Timestamp(f'{SPLIT[idx]}-12-31') if phase=='IS' else pd.Timestamp('2026-06-30')
    z,d,e,l=ex.simulate(idx,model,int(lead),start,end,ss,aa,tt,px,tv,actions)
    h,n,a=gg[['hits','predicted','actual']].sum();th,tn=gg[['trade_hits','trade_count']].sum()
    info=dict(index_id=idx,phase=phase,model=model,lead=int(lead),budget_krw=3e9,precision=h/n if n else np.nan,recall=h/a,f1=2*h/(n+a),
      hits=h,predicted=n,actual=a,trade_precision=th/tn if tn else np.nan,trade_recall=th/a,trade_f1=2*th/(tn+a),
      trade_targets=tn,warmup_events=int(gg.warmup.sum()),return_fallback_events=int(gg.return_fallback.sum()))
    performance.append(dict(**z,**info))
    if d is not None:daily.append(d.assign(**{k:info[k] for k in ['index_id','phase','model','lead']}))
    if e is not None:events.append(e.assign(**{k:info[k] for k in ['index_id','phase','model','lead']}))
    if l is not None and len(l):ledger.append(l.assign(**{k:info[k] for k in ['index_id','phase','model','lead']}))
    print('execution',phase,idx,model,lead,z['status'],flush=True)
   if phase=='IS':
    per=pd.DataFrame(performance);per=per[per.index_id.eq(idx)&per.phase.eq('IS')&per.status.eq('valid')].copy()
    basef=per[per.model.eq('baseline__count')].set_index('lead').f1
    per['baseline_F1']=per.lead.map(basef);per['eligible_balanced']=per.f1.ge(per.baseline_F1-1e-12)&per.mean_event_net.gt(0)
    per['features']=per.model.str.split('__').str[0].map({**{k:len(v) for k,v in ML.items()},'calibration':1,'baseline':0,'ma20':0,'shrunk_trend':0})
    save(idx+'_IS_candidates',per)
    valid=per[per.eligible_balanced]
    if valid.empty:choices.append(dict(index_id=idx,status='no_qualifying_candidate'));continue
    b=valid.sort_values(['sharpe','f1','features','lead','model'],ascending=[False,False,True,True,True]).iloc[0]
    family,policy=b.model.split('__');choices.append(dict(index_id=idx,status='selected',model=b.model,family=family,policy=policy,lead=int(b.lead),IS_sharpe=b.sharpe,IS_F1=b.f1,IS_mean_net=b.mean_event_net,IS_n=int(b.n_events),baseline_F1=b.baseline_F1))
  if phase=='IS':dump('IS_selection_freeze',dict(created_utc=datetime.now(timezone.utc).isoformat(),choices=choices,OOS_scoring_not_started=True))
 save('performance',performance);save('forecasts',forecast);save('targets',targets);save('fit_audit',audits);save('missing',missing)
 pd.DataFrame(probs).to_parquet(DEST/'probabilities.parquet',index=False)
 for n,items in [('daily',daily),('events',events),('ledger',ledger)]:pd.concat(items,ignore_index=True).to_parquet(DEST/(n+'.parquet'),index=False)
 chosen=pd.DataFrame(choices);save('selected',chosen)
 per=pd.DataFrame(performance);selected=per.merge(chosen[['index_id','model','lead']],on=['index_id','model','lead']);save('selected_performance',selected)
 audits=pd.DataFrame(audits);l=pd.concat(ledger,ignore_index=True);l=l[l.side.isin(['buy','sell'])];d=pd.concat(daily,ignore_index=True);e=pd.concat(events,ignore_index=True)
 keys=['index_id','phase','model','lead'];by=l.groupby(keys+['date','code']);pnl=e.groupby(keys).pnl.sum();eq=d.sort_values('date').groupby(keys).equity.last()
 checks=dict(IS_selection_only=True,past_labels=bool((pd.to_datetime(audits.max_label_date)<pd.to_datetime(audits.information_date)).all()),
  labels_capped_by_own_IS=bool(pd.to_datetime(audits.max_label_date).dt.year.le(audits.train_end).all()),
  past_return_labels=bool((pd.to_datetime(audits.return_max_exit)<pd.to_datetime(audits.information_date)).all()),
  cash_nonnegative=bool(d.cash.ge(-1e-4).all()),fees=bool(np.allclose(l.fee,l.notional*.0025)),
  turnover_limit=bool((by.notional.sum()<=by.limit.first()+1e-4).all()),pnl_nav=bool(np.allclose(eq.reindex(pnl.index)-3e9,pnl,atol=.01)),
  source_unchanged=hashlib.sha256((OUT/'rule_margin_audit/model_panel.parquet').read_bytes()).hexdigest()==json.loads((DEST/'protocol.json').read_text(encoding='utf8'))['source_hash'])
 dump('checks',checks);assert all(checks.values()),checks
 print(selected.to_string(index=False),flush=True)

if __name__=='__main__':main()
