"""User-authorized incomplete-roster research branch; no invented constituent."""
from research import ROOT,OUT,load,features
import sys,json,hashlib
from dataclasses import replace
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits

DEST=ROOT/'04_results/two_axis_selection_20261118'
def main():
 threadpool_limits(1);DEST.mkdir(exist_ok=True);(DEST/'frames').mkdir(exist_ok=True)
 sys.path.append(str(ROOT/'02_code/modules/k100_long_20261117'))
 builder=load('02_code/modules/k100_long_20261117/three_build.py','partial_builder')
 models,mm=builder.old.setup();models.config=replace(models.config,allow_underfilled_k100=True);e=models.engines
 schedule=pd.read_csv(ROOT/'04_results/k100_long_20261117/case_schedule.csv',parse_dates=['information_date','entry_date','review_cutoff','exit_date','effective_date'])
 r=schedule[schedule.case_id.eq('KOSPI100_2024-06_L20')].iloc[0];info=r.information_date
 baseline,meta,f=e.forecast(r.index_id,r.event,info)
 projection=models._projection(r.index_id,r.event,info,f);codes=f.index
 h=e.px.loc[:info].reindex(columns=codes).tail(61);m=e.re.MCD.loc[:info].reindex(columns=codes).tail(61)
 d=pd.DataFrame(index=codes);d['code']=codes;d['group']=f.grp.astype(str);d['current_cap']=projection['last']
 d['projected_cap']=f.mc;d['past_sum']=projection['past_sum'];d['full_days']=projection['full_days'];d['remaining']=projection['remaining']
 d['log_ma20_ratio']=np.log(m.tail(20).mean()/d.current_cap)
 d['momentum20']=h.iloc[-1]/h.iloc[-21]-1;d['momentum60']=h.iloc[-1]/h.iloc[0]-1
 d['relative_momentum20']=d.momentum20-d.momentum20.groupby(d.group).transform('mean')
 d['volatility60']=h.pct_change(fill_method=None).std()*np.sqrt(252)
 d['adv20']=e.re.TVD.loc[:info].reindex(columns=codes).tail(20).mean()
 dates=e.S[e.S<info][-20:]
 for actor,name in [('foreign','외국인'),('institution','기관')]:
  panel=pd.read_pickle(ROOT/f'01_data/processed/flow_{name}.pkl');z=panel.reindex(index=dates,columns=codes);s=z.sum(min_count=20)
  d[actor+'_cap20']=s/(d.current_cap*1e6)
  if actor=='foreign':
   d['foreign_adv20']=s/(20*d.adv20);d['foreign_persistence20']=z.gt(0).mean().where(z.notna().all())
   d['foreign_sector_relative20']=d.foreign_cap20-d.foreign_cap20.groupby(d.group).transform('mean')
 d['eligibility']=f.eligibility_state
 # Outcomes are attached for evaluation only; forecast generation below never sees them.
 truth=e.truth(r.index_id,r.event);d['actual_addition']=codes.isin(truth)
 cached=models.forecast(r.index_id,r.event,info)
 d['raw_frequency']=pd.Series(cached['path_frequency']['scores']).reindex(codes)
 d['baseline_selected']=codes.isin(baseline);d['rank_selected']=codes.isin(cached['anchored_rank_update']['targets'] or [])
 d['frequency_selected']=codes.isin(cached['path_frequency']['targets'] or [])
 for k in ['case_id','index_id','event','year','lead','information_date','review_cutoff','effective_date','entry_date','exit_date']:d[k]=r[k]
 assert cached['path_frequency']['metadata']['status']=='ok'
 assert d.raw_frequency.notna().all() and d.raw_frequency.between(0,1).all()
 original_panel=d.reset_index(drop=True)
 from candidates import enrich,dated_roster
 mem=e.k100;current,rd,known=dated_roster('KOSPI100',info,mem,pd.DataFrame())
 evidence=pd.read_parquet(ROOT/'04_results/index_financial_gap_repair_20261118/financial_receipt_evidence.parquet')
 z=enrich(original_panel,f,'KOSPI100',info,current,rd,known,evidence)
 listing=pd.read_csv(ROOT/'01_data/processed/kind_listing.csv',parse_dates=['list_date']);listing=listing[~listing.code.duplicated(False)].set_index('code')
 z['observed_listing_age_days']=(info-z.code.map(listing.list_date)).dt.days.where(z.code.map(listing.list_date).le(info))
 z['financial_current_known']=0.;z['prior_parent_member_feature']=np.nan
 history=pd.read_parquet(OUT/'panel_enriched.parquet');z=features(z,history)
 frame=dict(frame=f,panel=original_panel,baseline=baseline,metadata=meta,projection=projection)
 pd.to_pickle(frame,DEST/'frames'/f'{r.case_id}.pkl')
 # Rule-distance fields used only by the separate seven/35-variable challengers.
 eng=load('02_code/modules/signal_a_rules_v2/engine.py','partial_rule_distances')
 p=eng.profile_for(r.event,asof=info,cutoff_date=r.review_cutoff);pool=f[f.eligibility_state.eq('pass')&f.mc.gt(0)]
 from rule_margin_audit import RULE
 vals=pd.DataFrame(np.nan,index=f.index,columns=RULE)
 for _,g in pool.groupby('grp'):
  g=g.assign(code_key=g.index).sort_values(['mc','code_key'],ascending=[False,True])
  vals.loc[g.index,'rule_coverage_margin']=p.cumulative-g.mc.cumsum().shift(fill_value=0)/g.mc.sum()
  vals.loc[g.index,'rule_liquidity_margin']=p.liquidity_fraction-g.tv.rank(ascending=False,method='average')/len(g)
 parent=eng.prepare_selector(f,p).select(f.mc.to_numpy(),f.tv.to_numpy());caps=f.mc.where(parent).dropna().sort_values(ascending=False)
 for n,c in [(80,'rule_entry_cap_gap'),(100,'rule_size_cap_gap')]:vals[c]=np.log(f.mc.where(f.mc.gt(0))/caps.iloc[n-1])
 for c in RULE:z[c]=z.code.map(vals[c])
 z.to_parquet(DEST/'repaired_model_panel.parquet',index=False)
 original=pd.read_parquet(OUT/'rule_margin_audit/model_panel.parquet')
 assert not original.case_id.eq(r.case_id).any()
 pd.concat([original,z],ignore_index=True).to_parquet(DEST/'model_panel.parquet',index=False)
 pd.DataFrame([dict(case_id=r.case_id,actual=len(truth))]).to_csv(DEST/'repaired_denominator.csv',index=False)
 audit=dict(case_id=r.case_id,information_date=str(info),truth_for_evaluation_only=True,
  baseline_new_candidates=sorted(baseline),rank_new_candidates=cached['anchored_rank_update']['targets'],
  scenario_count=models.config.scenarios,scenarios_dropped=0,frequency_denominator=models.config.scenarios,
  raw_frequency_max=float(z.raw_frequency.max()),underfill_observations=models.underfill_audit,
  policy='keep selected partial constituent set; do not fill or discard underfilled scenarios; trade new candidates only',
  no_99_stock_replication=True,original_inputs_unchanged=True,production_certified=False)
 (DEST/'repair_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf8')
 (DEST/'repaired_forecast.json').write_text(json.dumps(builder.old.clean(cached),ensure_ascii=False,indent=2),encoding='utf8')
 print('REPAIRED',r.case_id,'underfill observations',len(models.underfill_audit),'new rank candidates',cached['anchored_rank_update']['targets'],flush=True)
if __name__=='__main__':main()
