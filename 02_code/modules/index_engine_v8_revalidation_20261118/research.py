"""IS-only forward-validation selection on widened, dated candidates; reused OOS audit."""
from context import ROOT,OUT,dump
import os,sys,json,importlib.util
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pandas as pd,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
def load(rel,name):
    sp=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
V5=load('02_code/modules/index_engine_v5_20261118/research.py','v5_reference_v7')
PRICE=V5.PRICE
EXTRA=V5.EXTRA+['industry_liquidity_percentile','entrant_rank_ratio','observed_cap_fraction','observed_tv_fraction','observed_listing_age_days','financial_current_known','prior_parent_member_feature']
FEATURES=PRICE+EXTRA
MODELS={'baseline':('baseline','count'),'logistic_count':('logistic','count'),'logistic_p50':('logistic','p50'),'boost_count':('boost','count'),'boost_p50':('boost','p50'),'strict_logistic_count':('logistic','strict_count')}
def save(n,x):pd.DataFrame(x).to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')
def features(g,history):
    dummy=pd.DataFrame({'assumed_available_date':pd.Series([],dtype='datetime64[ns]')})
    z=V5.feature_frame(g,history,dummy)
    for c in FEATURES:
        if c not in z:z[c]=np.nan
        z[c]=pd.to_numeric(z[c],errors='coerce')
    z[FEATURES]=z[FEATURES].replace([np.inf,-np.inf],np.nan)
    z['opportunity']=z.monitor_candidate
    return z
def fit(train,family):
    w=1/train.groupby('event').event.transform('size');w*=len(w)/w.sum()
    if family=='logistic':
        m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=600))
        m.fit(train[FEATURES],train.actual_addition.astype(int),logisticregression__sample_weight=w)
    else:
        m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=1811))
        m.fit(train[FEATURES],train.actual_addition.astype(int),histgradientboostingclassifier__sample_weight=w)
    return m
def choose(g,p,selector):
    z=g[g.opportunity].copy();z['prob']=p.reindex(z.index)
    if selector=='strict_count' and g.index_id.iloc[0]=='KRX300':z=z[z.strict_financial_candidate]
    z=z.sort_values(['prob','projected_cap','code'],ascending=[False,False,True])
    return set(z[z.prob.ge(.5)].code) if selector=='p50' else set(z.head(int(g.baseline_selected.sum())).code)
def summary(g):
    v=g.groupby('event')[['hits','predicted','actual']].mean().sum()
    return dict(n_events=g.event.nunique(),hits=v.hits,predicted=v.predicted,actual=v.actual,precision=v.hits/v.predicted if v.predicted else np.nan,recall=v.hits/v.actual if v.actual else np.nan,f1=2*v.hits/(v.predicted+v.actual) if v.predicted+v.actual else np.nan)
def main():
    threadpool_limits(1)
    raw=pd.read_parquet(OUT/'panel_enriched.parquet')
    full=pd.concat([features(g,raw) for _,g in raw.groupby('case_id')],ignore_index=True)
    full.to_parquet(OUT/'model_panel.parquet',index=False)
    coverage=pd.read_csv(ROOT/'04_results/index_engine_v6_20261118/candidate_coverage.csv');denom=coverage.set_index('case_id').actual.to_dict()
    protocol=dict(train_end=2022,OOS='2023-01 to 2026-06, repeatedly observed historical research',min_train_events=4,leads=[20,40,60,80],selection='IS forward-validation F1; tie precision, simpler model, larger lead',models=MODELS,features=FEATURES,financial_unknown='research monitoring allowed; strict challenger excludes unknown',no_outcome_fields_as_inputs=True,broker_rosters_as_inputs=False,live_order_allowed=False,capital_per_index=3e9,fee_each_side=.0025)
    dump('model_protocol',protocol)
    rows=[];targets=[];audit=[];prob=[];cache={};skipped=[]
    def evalcase(g,phase,report_id=None):
        r=g.iloc[0];tr=full[full.index_id.eq(r.index_id)&full.year.le(2022)&full.effective_date.lt(r.information_date)&full.event.ne(r.event)&full.opportunity&full.actual_addition.notna()]
        if tr.event.nunique()<4 or tr.actual_addition.nunique()<2:
            skipped.append(dict(case_id=r.case_id,phase=phase,reason='fewer_than_four_completed_IS_events_or_one_label_class'));return
        assert tr.effective_date.max()<r.information_date and tr.year.max()<=2022
        truth=set(g.loc[g.actual_addition.eq(True),'code']);base=set(g.loc[g.baseline_selected.eq(1)&g.opportunity,'code']);scores={}
        for family in ['logistic','boost']:
            key=(r.index_id,family,tuple(sorted(tr.event.unique())))
            if key not in cache:cache[key]=fit(tr,family)
            p=pd.Series(np.nan,index=g.index);mask=g.opportunity
            if mask.any():p.loc[mask]=cache[key].predict_proba(g.loc[mask,FEATURES])[:,1]
            scores[family]=p;audit.append(dict(case_id=r.case_id,index_id=r.index_id,phase=phase,family=family,train_events=tr.event.nunique(),train_rows=len(tr),max_label_date=tr.effective_date.max(),information_date=r.information_date))
            if phase!='IS_forward':
                prob.extend(dict(case_id=r.case_id,index_id=r.index_id,event=r.event,family=family,code=g.at[j,'code'],p=p[j],actual=bool(g.at[j,'actual_addition'])) for j in g.index[mask])
        for name,(family,selector) in MODELS.items():
            selected=base if family=='baseline' else choose(g,scores[family],selector)
            rows.append(dict(case_id=r.case_id,index_id=r.index_id,event=r.event,year=r.year,lead=r.lead,phase=phase,model=name,report_id=report_id,hits=len(selected&truth),predicted=len(selected),actual=denom[r.case_id],information_date=r.information_date,entry_date=r.entry_date,exit_date=r.exit_date))
            targets.extend(dict(case_id=r.case_id,index_id=r.index_id,event=r.event,model=name,code=c,weight=1/len(selected)) for c in sorted(selected))
    for i,(_,g) in enumerate(full[full.year.le(2022)].groupby('case_id'),1):
        evalcase(g,'IS_forward')
        if i%15==0:print('IS',i,'fits',len(cache),flush=True)
    fr=pd.DataFrame(rows);stats=[];common={}
    for idx,q in fr.groupby('index_id'):
        sets=[set(g.event) for _,g in q.groupby('lead')]
        common[idx]=sorted(set.intersection(*sets))
    dump('common_IS_events',common)
    for (idx,m,lead),g in fr.groupby(['index_id','model','lead']):stats.append(dict(index_id=idx,model=m,lead=int(lead),**summary(g)))
    stats=[dict(index_id=idx,model=m,lead=int(lead),**summary(g[g.event.isin(common[idx])])) for (idx,m,lead),g in fr.groupby(['index_id','model','lead'])]
    ss=pd.DataFrame(stats);save('IS_candidates',ss);picks=[]
    for idx,g in ss.groupby('index_id'):
        g=g.assign(simplicity=g.model.map({'baseline':0,'logistic_count':1,'logistic_p50':2,'boost_count':3,'boost_p50':4,'strict_logistic_count':5}))
        best=g.sort_values(['f1','precision','simplicity','lead'],ascending=[False,False,True,False]).iloc[0]
        picks.append(dict(index_id=idx,model=best.model,lead=int(best.lead),IS_f1=best.f1,IS_precision=best.precision,IS_recall=best.recall))
    save('frozen_choices',picks);dump('selection_freeze',dict(choices=picks,OOS_not_yet_scored=True,OOS_previously_seen_in_earlier_research=True))
    for i,(_,g) in enumerate(full[full.year.ge(2023)].groupby('case_id'),1):
        evalcase(g,'OOS')
        if i%12==0:print('OOS',i,flush=True)
    # Every existing same-date frame is audited. Missing frames are reported, never imputed.
    bpanel=pd.read_parquet(ROOT/'04_results/index_two_axes_20261117/broker/panel.parquet')
    pairs=pd.read_csv(ROOT/'04_results/index_two_axes_20261117/broker/pairs.csv');bden=pairs.groupby('report_id').actual.first()
    from candidates import enrich,dated_roster
    from financial_parse import financial_asof
    evidence=pd.read_parquet(OUT/'financial_evidence.parquet')
    oldfin=pd.read_parquet(ROOT/'04_results/index_financial_gap_repair_20261118/financial_receipt_evidence.parquet')
    listing=pd.read_csv(ROOT/'01_data/processed/kind_listing.csv',parse_dates=['list_date']);listing=listing[~listing.code.duplicated(False)].set_index('code')
    ba=[]
    for cid,g in bpanel.groupby('case_id'):
        r=g.iloc[0];rid=cid.removeprefix('broker_');path=ROOT/'04_results/index_two_axes_20261117/broker/frames'/(cid+'.pkl')
        if r.year<2023:ba.append(dict(report_id=rid,status='pre_OOS_report_not_in_primary_pair'));continue
        if not path.exists():ba.append(dict(report_id=rid,status='missing_same_date_frame'));continue
        f=pd.read_pickle(path)['frame'];info=pd.Timestamp(r.information_date);current=set(f.index[f.cur])
        z=enrich(g,f,r.index_id,info,current,info,len(current)>0,oldfin)
        z['observed_listing_age_days']=(info-z.code.map(listing.list_date)).dt.days.where(z.code.map(listing.list_date).le(info))
        z['financial_current_known']=0.;z['prior_parent_member_feature']=np.nan
        denom[cid]=bden[rid];evalcase(features(z,full),'BROKER_OOS',rid);ba.append(dict(report_id=rid,status='same_date_frame_evaluated'))
    save('broker_frame_audit',ba);save('forecasts',rows);save('targets',targets);save('fit_audit',audit);save('skipped_cases',skipped)
    pd.DataFrame(prob).to_parquet(OUT/'probabilities.parquet',index=False)
    result=[];fr=pd.DataFrame(rows);chosen={x['index_id']:(x['model'],x['lead']) for x in picks}
    for (idx,phase,m,lead),g in fr.groupby(['index_id','phase','model','lead']):
        if phase=='IS_forward':g=g[g.event.isin(common[idx])]
        result.append(dict(index_id=idx,phase=phase,model=m,lead=lead,selected=(m,int(lead))==chosen[idx],**summary(g)))
    save('accuracy_summary',result)
    print(pd.DataFrame(result).query('selected').to_string(index=False),flush=True)
if __name__=='__main__':main()
