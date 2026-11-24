"""Additional broker originals, explicit regular-roster scope, same-date refresh.

No strategy selection or fitting uses the new reports or their outcomes.
"""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'04_results/presentation_revision_20261118'
sys.path.append(str(ROOT/'02_code/modules/k100_long_20261117'))
import three_build as build

CURATED={
 'MERITZ_20211006_K200':('A138040 A178920 A005850 A009900','A329180','A377300','p2,5: 현대중공업 특례 제외. 카카오페이 추가 상장 조건부.'),
 'EUGENE_20211026_K200':('A138040 A009900 A178920 A005850','A329180 A377300','','p2: 6종목 중 특례 2종목 분리.'),
 'MERITZ_20220330_K200':('A383220 A137310 A000060 A271940 A381970 A300720 A039130','','','p2 표1: 정기편입 7종목.'),
 'SAMSUNG_20240322_K200':('A066970 A454910 A457190 A003030 A002840','A042700','A002840','p3~4: 본문 표 5종목. 미원상사는 쌍용C&E 제외 조건부. 한미반도체 특례 분리.'),
 'SAMSUNG_20240426_K200':('A066970 A454910 A457190 A003030 A002840','A042700','A002840','p3: 1차 전망 5종목 유지 및 한미반도체 특례 추가. 조건부 미원상사 포함/제외 병기.'),
 'EUGENE_20241114_K200':('A298040 A278470 A034230 A002840 A456040','','','p1: 예측 5종목. 관심종목에서 제외한 OCI도 원래 예측명단에 유지.'),
 'KIS_20251107_K200':('A034230 A002030','','','공식 웹 본문: 파라다이스·아세아 정기편입. 특례 논의 별도.'),
 'EUGENE_20260323_K200':('A483650','','','p5: 달바글로벌 1종목.'),
 'SAMSUNG_20260327_K200':('A483650 A267270','','','p1,3: 달바글로벌 100%, HD건설기계 60% 전망. 두 종목 모두 기본 예상명단.'),
 'SAMSUNG_20260424_K200':('A483650 A267270','','','p3: 달바글로벌·HD건설기계 유지.')}

def save(n,d):pd.DataFrame(d).to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')

def main():
 threadpool_limits(1)
 registry=pd.read_csv(OUT/'source_download_audit.csv').fillna('')
 registry['primary_codes']=registry.report_id.map(lambda k:';'.join(CURATED[k][0].split()) if k in CURATED else '')
 registry['special_codes']=registry.report_id.map(lambda k:';'.join(CURATED[k][1].split()) if k in CURATED else '')
 registry['conditional_codes']=registry.report_id.map(lambda k:';'.join(CURATED[k][2].split()) if k in CURATED else '')
 registry['curation_notes']=registry.report_id.map(lambda k:CURATED[k][3] if k in CURATED else '부분 명단 또는 아직 결과가 없는 2026년 12월 전망')
 registry['comparison_status']=np.where(registry.report_id.isin(CURATED),'historical_same_date_refresh',np.where(registry.index_id.eq('KRX300'),'partial_single_name_not_full_recall','future_outcome_pending'))
 registry.loc[registry.report_id.eq('KIS_20251107_K200'),'status']='verified_official_html'
 registry.loc[registry.report_id.eq('HANA_20180320_KRX300'),'primary_codes']='A042660'
 save('new_reports_curated',registry)
 old=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/source_registry.csv')
 assert not registry.sha256.isin(old.sha256).any()
 save('all_source_registry',pd.concat([old.assign(corpus='previous_38'),registry.assign(corpus='added_14')],ignore_index=True))
 models,mm=build.old.setup();e=models.engines
 res=build.load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','revision_resources');res.write=lambda *a:None
 px,tv,actions=res.resources()
 from corporate_actions import supplement
 px,tv,actions=supplement(px,tv,actions)
 ex=build.load('02_code/modules/index_forecast_development_20261117/execution.py','revision_execution');ex.B=3e9
 pairs=[];audit=[];rosters=[]
 for r in registry[registry.report_id.isin(CURATED)].itertuples():
  print('REPORT',r.report_id,flush=True)
  info=pd.Timestamp(r.printed_date);info=e.S[e.S<=info][-1];cid='broker_'+r.report_id
  try:
   base,meta,f=e.forecast('KOSPI200',r.event,info)
   if f is None:raise ValueError(str(meta))
   pr=models._projection('KOSPI200',r.event,info,f);selector,_=models._selector('KOSPI200',r.event,info,f,base)
   h=e.px.loc[:info].reindex(columns=f.index).tail(61)
   momentum=h.iloc[-1]/h.iloc[-21]-1
   logratio=(.25*np.log1p(momentum.clip(lower=-.99))*pr['remaining']/40).fillna(0).clip(-.4,.4)
   cap=(pr['past_sum']+pr['remaining']*pr['last']*np.exp(logratio))/pr['full_days']
   cap=np.where(np.isfinite(cap)&(cap>0)&f.eligibility_state.eq('pass'),cap,f.mc)
   special=set(r.special_codes.split(';'))-{''}
   # Same hard regular-candidate checks as candidates.enrich; no financial rule applies to K200.
   ok=f.eligibility_state.eq('pass')&f.mc.gt(0)&f.tv.ge(0)&f.grp.notna()&~f.cur.astype(bool)
   if 'strategy_a_route' in f:ok &= f.strategy_a_route.fillna('unknown').isin(['regular_candidate','regular_candidate_proxy'])
   if 'a_exclusion_reason' in f:ok &= f.a_exclusion_reason.fillna('').eq('')
   for col in ['float_observation_date','sector_source_date']:
    if col in f:ok &= ~pd.to_datetime(f[col]).gt(info)
   ns=set(selector(cap))&set(f.index[ok]);ns-=special
   bs=set(r.primary_codes.split(';'))-{''};truth=set(e.truth('KOSPI200',r.event))-special
   cal=e.calendar('KOSPI200',r.event)
   entry=e.S[e.S>info][0];exit=pd.Timestamp(cal['exit'])
   assert info<pd.Timestamp(cal['announcement']),('post-announcement',r.report_id)
   for who,names in [('engine',ns),('broker',bs)]:
    ss=pd.DataFrame([dict(index_id='KOSPI200',event=r.event,case_id=cid,lead=0,entry_date=entry,exit_date=exit)])
    tt=pd.DataFrame([dict(case_id=cid,model='broker_pair',code=c,weight=1/len(names)) for c in sorted(names)],columns=['case_id','model','code','weight'])
    zz,dd,ee,ll=ex.simulate('KOSPI200','broker_pair',0,entry,min(pd.Timestamp('2026-06-30'),exit+pd.Timedelta(days=120)),ss,ss.assign(model='broker_pair',forecast_available=True),tt,px,tv,actions)
    pairs.append(dict(index_id='KOSPI200',report_id=r.report_id,event=r.event,who=who,model='shrunk_trend__count' if who=='engine' else 'published_report',information_date=info,entry_date=entry,hits=len(names&truth),predicted=len(names),actual=len(truth),status=zz['status'],net_return=ee.net_return.sum() if ee is not None else np.nan,names=';'.join(sorted(names))))
    for c in sorted(names):rosters.append(dict(report_id=r.report_id,event=r.event,who=who,code=c,actual_hit=c in truth,conditional=c in r.conditional_codes.split(';')))
   audit.append(dict(report_id=r.report_id,status='paired',event=r.event,information_date=info,entry_date=entry,exit_date=exit,truth=';'.join(sorted(truth)),regular_scope_special_excluded=';'.join(sorted(special))))
  except Exception as err:
   audit.append(dict(report_id=r.report_id,status='unavailable',reason=repr(err)));print('ERROR',repr(err),flush=True)
  save('new_broker_pairs',pairs);save('new_broker_audit',audit);save('new_broker_rosters',rosters)
  models._cache.clear();models._simulation_cache.clear();e.k200.cache_clear()
 oldpairs=pd.read_csv(ROOT/'04_results/two_axis_selection_20261118/broker_pairs.csv')
 b=pd.concat([oldpairs,pd.DataFrame(pairs)],ignore_index=True)
 assert not b.duplicated(['report_id','who']).any()
 save('broker_pairs',b)
 summaries=[]
 for scope,bb in [('all_report_revisions',b),('conditional_miwon_excluded',b.copy())]:
  if scope=='conditional_miwon_excluded':
   for rid in ['SAMSUNG_20240322_K200','SAMSUNG_20240426_K200']:
    mask=bb.report_id.eq(rid)&bb.who.eq('broker');truth=set(next(a['truth'].split(';') for a in audit if a['report_id']==rid and a['status']=='paired')) if any(a['report_id']==rid and a['status']=='paired' for a in audit) else set()
    bb.loc[mask,'predicted']-=1;bb.loc[mask,'hits']-=int('A002840' in truth)
  for (idx,who),g in bb.groupby(['index_id','who']):
   h,n,a=g.groupby('event')[['hits','predicted','actual']].mean().sum()
   good=set(bb.groupby('report_id').filter(lambda q:len(q)==2 and q.status.eq('valid').all()).report_id)
   rr=g[g.report_id.isin(good)].groupby('event').net_return.mean();pos=rr[rr>0];neg=rr[rr<0]
   summaries.append(dict(scope=scope,index_id=idx,who=who,n_reports=len(g),n_events=g.event.nunique(),precision=h/n,recall=h/a,f1=2*h/(n+a),n_return_events=len(rr),mean_net=rr.mean() if scope=='all_report_revisions' else np.nan,win_rate=rr.gt(0).mean() if scope=='all_report_revisions' else np.nan,payoff=pos.mean()/abs(neg.mean()) if len(pos) and len(neg) and scope=='all_report_revisions' else np.nan))
 save('broker_summary',summaries)
 (OUT/'broker_protocol.json').write_text(json.dumps(dict(information_date='printed report date close proxy, exact public timestamp unverified; entry next close',engine='long standalone IS selected prediction family shrunk_trend; same-date refresh',budget=3e9,fees_each_side=.0025,source_count=dict(KOSPI200=47,KOSPI100=0,KRX300=5),partial_KRX_added='HANA_20180320: one conditional company, not a complete roster, no full recall',future='Three 2026-12 reports have no observed outcomes',conditional='Samsung 2024 two documents retain their 5-name primary tables; Miwon excluded alternative reported for accuracy only',OOS='Previously viewed historical benchmark, not untouched proof; 2021 report review added',accuracy_weight='average report counts within event before summing across events'),ensure_ascii=False,indent=2),encoding='utf8')
 print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':main()
