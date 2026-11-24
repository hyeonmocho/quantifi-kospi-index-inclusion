"""KRX300 full-roster warmup and partial-roster OOS comparison; K100 evidence audit."""
from research import ROOT,OUT,load
from select_standalone_balanced import fit
from rule_margin_audit import CORE,RULE,gap
import pandas as pd,numpy as np,json,sys
from threadpoolctl import threadpool_limits
DEST=ROOT/'04_results/two_axis_selection_20261118'
def save(n,x):pd.DataFrame(x).to_csv(DEST/(n+'.csv'),index=False,encoding='utf-8-sig')
def main():
 threadpool_limits(1)
 sys.path.append(str(ROOT/'02_code/modules/k100_long_20261117'))
 builder=load('02_code/modules/k100_long_20261117/three_build.py','broad_pair_builder');models,mm=builder.old.setup();e=models.engines
 records=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/source_registry.csv')
 cases=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/comparison/report_cases.csv')
 inputs=records[records.index_id.eq('KRX300') & records.roster_complete.eq(True)].copy()
 inputs=pd.concat([inputs,records[records.report_id.eq('SAMSUNG_20240119_KRX300')]],ignore_index=True)
 origfin=pd.read_parquet(ROOT/'04_results/index_financial_gap_repair_20261118/financial_receipt_evidence.parquet')
 fin=pd.read_parquet(OUT/'financial_evidence.parquet')
 from candidates import enrich,dated_roster
 from financial_parse import financial_asof
 full=pd.read_parquet(DEST/'model_panel.parquet');rows=[];audit=[];targets=[];days=[];ledgers=[];fitlog=[]
 resources=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','broad_pair_resources');resources.write=lambda *a:None
 px,tv,actions=resources.resources()
 from corporate_actions import supplement
 px,tv,actions=supplement(px,tv,actions)
 ex=load('02_code/modules/index_forecast_development_20261117/execution.py','broad_pair_execution');ex.B=3e9
 for r in inputs.itertuples():
  try:
   info=pd.Timestamp(r.assumed_available);cid='broker_'+r.report_id;event=r.event
   names,meta,f=e.forecast('KRX300',event,info)
   if names is None or f is None:raise ValueError(str(meta))
   cal=e.calendar('KRX300',event);eff=cal['effective'];cut=e.S[(e.S.year==eff.year)&(e.S.month==eff.month-2)][-1]
   proj=models._projection('KRX300',event,info,f);current,rd,known=dated_roster('KRX300',info,pd.DataFrame(),e.snaps)
   assert known
   d=pd.DataFrame({'code':f.index,'group':f.grp.to_numpy(),'projected_cap':f.mc.to_numpy(),'adv20':e.re.TVD.loc[:info].reindex(columns=f.index).tail(20).mean().to_numpy(),'eligibility':'pass',
    'baseline_selected':f.index.isin(names),'rank_selected':False,'raw_frequency':0.,'case_id':cid,'index_id':'KRX300','event':event,'year':int(event[:4]),'information_date':info,'review_cutoff':cut,'effective_date':eff,'remaining_fraction':proj['remaining']/proj['full_days']})
   z=enrich(d,f,'KRX300',info,current,rd,known,origfin).set_index('code',drop=False)
   fs,fq=financial_asof(z.index,info,cut,fin)
   # Replace earlier fiscal-year-only financial overlay using the current dated evidence.
   hard=z.candidate_reason.isin(['existing_candidate_or_not_eligible','monitor_only_financial_unknown','zero_frequency_not_hard_exclusion','financial_impairment'])
   z['opportunity']=hard&~fs.eq('observed_impairment')
   # Preserve every non-financial hard eligibility gate.
   assert not (z.opportunity & z.current_member).any()
   base=set(z.loc[z.opportunity&z.baseline_selected,'code'])
   tr=full[full.index_id.eq('KRX300')&full.year.le(2022)&full.effective_date.lt(info)&full.event.ne(event)&full.opportunity&full.actual_addition.notna()]
   warmup=tr.event.nunique()<4
   if warmup:ns=base;engine='baseline_warmup';scores=None
   else:
    result=models.forecast('KRX300',event,info);freq=pd.Series(result['path_frequency']['scores'])
    z['raw_frequency']=z.code.map(freq);pr=z.raw_frequency.clip(.005,.995);z['logit_frequency']=np.log(pr/(1-pr))
    vals=pd.DataFrame(np.nan,index=f.index,columns=RULE)
    pool=f[f.mc.gt(0)&f.tv.ge(0)&f.grp.notna()]
    for _,g in pool.groupby('grp'):
     g=g.assign(code_key=g.index).sort_values(['mc','code_key'],ascending=[False,True])
     vals.loc[g.index,'rule_coverage_margin']=.8-g.mc.cumsum().shift(fill_value=0)/g.mc.sum()
     vals.loc[g.index,'rule_liquidity_margin']=.8-g.tv.rank(ascending=False,method='min')/len(g)
     vals.loc[g.index,'rule_entry_cap_gap']=gap(g.mc,.3*len(g))
    vals.loc[pool.index,'rule_size_cap_gap']=gap(pool.mc,300)
    for c in RULE:z[c]=z.code.map(vals[c])
    mdl=fit(tr,CORE+RULE);q=z[z.opportunity].copy();q['score']=mdl.predict_proba(q[CORE+RULE])[:,1]
    q=q.assign(code_key=q.code).sort_values(['score','projected_cap','code_key'],ascending=[False,False,True])
    ns=set(q.head(len(base)).code);engine='compact7';scores=q
    fitlog.append(dict(report_id=r.report_id,train_events=tr.event.nunique(),max_label_date=tr.effective_date.max(),information_date=info))
   truth=e.truth('KRX300',event)
   assert truth is not None
   bs=set(str(r.primary_codes).split(';'));complete=bool(r.roster_complete)
   if not complete:
    # A named subset cannot define the broker's total recall. Compare precision@same K only.
    assert scores is not None
    ns=set(scores.head(len(bs)).code);scope='partial_same_count_OOS';full_engine_size=len(base)
   else:scope='full_roster_pretraining';full_engine_size=len(ns)
   entry=e.S[e.S>info][0];exit=cal['exit']
   for who,ss in [('engine',ns),('broker',bs)]:
    sched=pd.DataFrame([dict(case_id=cid,index_id='KRX300',event=event,lead=0,entry_date=entry,exit_date=exit)])
    tt=pd.DataFrame([dict(case_id=cid,model='pair',code=c,weight=1/len(ss)) for c in sorted(ss)],columns=['case_id','model','code','weight'])
    out,dd,ee,ll=ex.simulate('KRX300','pair',0,entry,min(pd.Timestamp('2026-06-30'),exit+pd.Timedelta(days=120)),sched,sched.assign(model='pair',forecast_available=True),tt,px,tv,actions)
    h=len(ss&truth);n=len(ss);a=len(truth)
    row=dict(index_id='KRX300',report_id=r.report_id,event=event,scope=scope,who=who,engine=engine if who=='engine' else 'published_report',warmup=warmup,
     information_date=info,entry_date=entry,exit_date=exit,complete_roster=complete,predicted=n,hits=h,actual=a,
     precision=h/n if n else np.nan,recall=h/a if complete else np.nan,f1=2*h/(n+a) if complete else np.nan,
     full_engine_count=full_engine_size,train_events=tr.event.nunique(),status=out['status'],net_return=ee.net_return.sum() if ee is not None else np.nan,names=';'.join(sorted(ss)),source_url=r.source_url)
    rows.append(row);targets.extend(dict(report_id=r.report_id,who=who,code=c) for c in sorted(ss))
    if dd is not None:days.append(dd.assign(report_id=r.report_id,who=who,scope=scope))
    if ll is not None and len(ll):ledgers.append(ll.assign(report_id=r.report_id,who=who,scope=scope))
   audit.append(dict(report_id=r.report_id,status='paired',scope=scope,train_events=tr.event.nunique(),financial_known=int(fs.isin(['observed_no_impairment','observed_impairment']).sum()),official_rule_chain_certified=False))
   print('PAIR',r.report_id,scope,engine,len(ns),len(bs),flush=True)
  except (ValueError,KeyError,AssertionError,IndexError) as err:
   audit.append(dict(report_id=r.report_id,status='unavailable',reason=repr(err)));print('UNAVAILABLE',r.report_id,repr(err),flush=True)
 save('broker_broad_pairs',rows);save('broker_broad_audit',audit);save('broker_broad_targets',targets);save('broker_broad_fit',fitlog)
 for n,parts in [('broker_broad_daily',days),('broker_broad_ledger',ledgers)]:
  if parts:pd.concat(parts,ignore_index=True).to_parquet(DEST/(n+'.parquet'),index=False)
 b=pd.DataFrame(rows);summary=[]
 if len(b):
  for (scope,who),g in b.groupby(['scope','who']):
   h,n,a=g.groupby('event')[['hits','predicted','actual']].mean().sum();complete=g.complete_roster.all()
   good=set(b.groupby('report_id').filter(lambda q:len(q)==2 and q.status.eq('valid').all()).report_id)
   rr=g[g.report_id.isin(good)].groupby('event').net_return.mean();positive=rr[rr>0];negative=rr[rr<0]
   summary.append(dict(index_id='KRX300',scope=scope,who=who,n_reports=len(g),n_events=g.event.nunique(),precision=h/n if n else np.nan,recall=h/a if complete else np.nan,f1=2*h/(n+a) if complete else np.nan,n_valid_return_events=len(rr),mean_net=rr.mean(),win_rate=rr.gt(0).mean() if len(rr) else np.nan,payoff=positive.mean()/abs(negative.mean()) if len(positive) and len(negative) else np.nan))
 save('broker_broad_summary',summary)
 # Exhaustive within the local curated inventory; no claim of global report nonexistence.
 save('broker_three_index_coverage',[
  dict(index_id='KOSPI200',full_reports=14,full_events=8,partial_reports=0,comparison='selected_prediction_family_OOS',reason='same-date replay completed'),
  dict(index_id='KOSPI100',full_reports=0,full_events=0,partial_reports=0,comparison='not_estimable',reason='no complete pre-announcement forecast roster located; KRX100 and size-largecap reports are different indices'),
  dict(index_id='KRX300',full_reports=int(inputs.roster_complete.sum()),full_events=inputs[inputs.roster_complete].event.nunique(),partial_reports=int((~inputs.roster_complete).sum()),comparison='warmup_full_rosters_plus_partial_OOS_separate',reason='2018-2019 full reports precede four training events; 2024 named subset cannot define recall')])
 checks={'no_future_model_labels':all(pd.Timestamp(v['max_label_date'])<pd.Timestamp(v['information_date']) and pd.Timestamp(v['max_label_date']).year<=2022 for v in fitlog),
  'partial_recall_not_reported':bool(b.loc[~b.complete_roster,['recall','f1']].isna().all().all()) if len(b) else True,
  'warmup_not_called_ML':bool(b.loc[b.warmup&b.who.eq('engine'),'engine'].eq('baseline_warmup').all()) if len(b) else True}
 if len(b) and days and ledgers:
  ll=pd.concat(ledgers);ll=ll[ll.side.isin(['buy','sell'])];dd=pd.concat(days);keys=['report_id','who'];gg=ll.groupby(keys+['date','code'])
  end=dd.sort_values('date').groupby(keys).equity.last();ret=b.set_index(keys).net_return
  checks.update(all_pairs_valid=bool(b.status.eq('valid').all()),cash=bool(dd.cash.ge(-1e-4).all()),
   fees=bool(np.allclose(ll.fee,ll.notional*.0025)),liquidity=bool((gg.notional.sum()<=gg.limit.first()+1e-4).all()),
   PNL_NAV=bool(np.allclose(end.reindex(ret.index)/3e9-1,ret)))
 (DEST/'broker_broad_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');assert all(checks.values())
 print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
