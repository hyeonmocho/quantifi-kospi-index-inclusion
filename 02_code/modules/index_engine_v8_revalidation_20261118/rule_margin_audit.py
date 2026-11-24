"""Frozen, paired rule-distance logistic experiment; no production changes."""
from research import ROOT, OUT, load, FEATURES
import json, hashlib, itertools, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

DEST = OUT / 'rule_margin_audit'
CORE = ['logit_frequency', 'baseline_selected', 'remaining_fraction']
RULE = ['rule_coverage_margin', 'rule_liquidity_margin', 'rule_entry_cap_gap', 'rule_size_cap_gap']
MODELS = {'legacy': FEATURES, 'legacy_no_rank': [x for x in FEATURES if x != 'rank_selected'],
          'core': CORE, 'compact_rules': CORE + RULE, 'compact_rank': CORE + RULE + ['rank_selected'],
          'legacy_rules': FEATURES + RULE}
for col in RULE:
    MODELS['compact_without_' + col.removeprefix('rule_')] = CORE + [x for x in RULE if x != col]
DIRS = {'KOSPI200': 'index_two_axes_20261117/k200_long/frames',
        'KOSPI100': 'k100_long_20261117/frames', 'KRX300': 'index_two_axes_20261117/three_indices/frames'}
def save(name, value):
    pd.DataFrame(value).to_csv(DEST / (name + '.csv'), index=False, encoding='utf-8-sig')
def dump(name, value):
    (DEST / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')
def gap(cap, rank):
    """Reference level, not a claim that this is the final legal selection gate."""
    rank = int(np.floor(rank))
    ordered = cap[cap.gt(0)].sort_values(ascending=False)
    if rank < 1 or rank > len(ordered): return pd.Series(np.nan, index=cap.index)
    return np.log(cap.where(cap.gt(0)) / ordered.iloc[rank-1])

def build_features(full):
    eng = load('02_code/modules/signal_a_rules_v2/engine.py', 'rule_margin_frozen_selector')
    output, audit = [], []
    for cid, g in full.groupby('case_id', sort=True):
        r = g.iloc[0]
        path = ROOT / '04_results' / DIRS[r.index_id] / (cid + '.pkl')
        o = pd.read_pickle(path); f = o['frame'].copy()
        # Only dated input frames and publicly dated rule profiles are consulted.
        assert pd.Timestamp(o['panel'].information_date.iloc[0]) == r.information_date
        if r.index_id == 'KRX300':
            pool = f[f.mc.gt(0) & f.tv.ge(0) & f.grp.notna()].copy()
            cumulative, liquid, profile = .8, .8, 'frozen_BMI_proxy'
        else:
            p = eng.profile_for(r.event, asof=r.information_date, cutoff_date=r.review_cutoff)
            pool = f[f.eligibility_state.eq('pass') & f.mc.gt(0)].copy()
            cumulative, liquid, profile = p.cumulative, p.liquidity_fraction, p.name
        vals = pd.DataFrame(np.nan, index=f.index, columns=RULE)
        for sector, s in pool.groupby('grp', sort=True):
            s = s.assign(code_key=s.index).sort_values(['mc','code_key'], ascending=[False, True])
            before = s.mc.cumsum().shift(fill_value=0) / s.mc.sum()
            vals.loc[s.index, 'rule_coverage_margin'] = cumulative - before
            method = 'min' if r.index_id == 'KRX300' else 'average'
            vals.loc[s.index, 'rule_liquidity_margin'] = liquid - s.tv.rank(ascending=False, method=method) / len(s)
            if r.index_id == 'KRX300': threshold = .3 * len(s)
            else: threshold = p.entrant_buffer * int((f.cur & f.grp.eq(sector)).sum())
            vals.loc[s.index, 'rule_entry_cap_gap'] = gap(s.mc, threshold)
        if r.index_id == 'KOSPI100':
            selector = eng.prepare_selector(f, p)
            parent = selector.select(f.mc.to_numpy(), f.tv.to_numpy())
            caps = f.mc.where(parent).dropna().sort_values(ascending=False)
            for threshold, col in [(80,'rule_entry_cap_gap'), (100,'rule_size_cap_gap')]:
                vals[col] = np.log(f.mc.where(f.mc.gt(0)) / caps.iloc[threshold-1]) if len(caps)>=threshold else np.nan
        else:
            vals.loc[pool.index,'rule_size_cap_gap'] = gap(pool.mc, 200 if r.index_id=='KOSPI200' else 300)
        z = g.copy()
        for col in RULE: z[col] = z.code.map(vals[col])
        output.append(z)
        audit.append(dict(case_id=cid,index_id=r.index_id,event=r.event,profile=profile,
                          information_date=r.information_date,source=str(path.relative_to(ROOT)),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pool=len(pool),
                          **{x+'_missing_candidates':int(z.loc[z.opportunity,x].isna().sum()) for x in RULE}))
    save('feature_audit',audit)
    result=pd.concat(output,ignore_index=True)
    result.to_parquet(DEST/'model_panel.parquet',index=False)
    return result

def fit(train, cols):
    w=1/train.groupby('event').event.transform('size');w*=len(w)/w.sum()
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                        StandardScaler(),LogisticRegression(C=.1,max_iter=600))
    model.fit(train[cols],train.actual_addition.astype(int),logisticregression__sample_weight=w)
    return model

def summarize(fr):
    rows=[]
    for (idx,m,phase),q in fr.groupby(['index_id','model','phase']):
        h,n,a=q[['hits','predicted','actual']].sum()
        rows.append(dict(index_id=idx,model=m,phase=phase,n_events=len(q),hits=h,predicted=n,actual=a,
                         precision=h/n if n else np.nan,recall=h/a if a else np.nan,f1=2*h/(n+a) if n+a else np.nan,
                         first_event=q.event.min(),last_event=q.event.max()))
    return pd.DataFrame(rows)

def main():
    threadpool_limits(1);DEST.mkdir(exist_ok=True)
    old=pd.read_parquet(OUT/'model_panel.parquet')
    oldfr=pd.read_csv(OUT/'forecasts.csv',parse_dates=['information_date','entry_date','exit_date'])
    choices=pd.read_csv(OUT/'performance_choices.csv').set_index('index_id')
    common=json.loads((OUT/'common_IS_events.json').read_text(encoding='utf8'))
    dump('protocol',dict(created_utc=datetime.now(timezone.utc).isoformat(),features=MODELS,
         leads=choices.lead.astype(int).to_dict(),C=.1,min_train_events=4,training_end=2022,
         selector='same number as baseline, equal weights; no threshold or lead tuning',
         primary_contrasts=['legacy_rules minus legacy','compact_rules minus core'],
         selection='IS forward-validation F1, then precision, then fewer features; baseline and diagnostic ablations not selection candidates',
         selection_candidates=['legacy','core','compact_rules','compact_rank','legacy_rules'],
         OOS='2023-01-02 to 2026-06-30; repeatedly examined historical OOS, not fresh confirmation',
         rule_caveats=['KOSPI100 80/120 proxy not certified','KRX300 historical BMI proxy not certified TMI',
             'coverage margin alone omits replacement, manufacturing and industry protection overlays',
             'size rank gap is a reference, not exact final fill cutoff with incumbents',
             'candidate set, financial unknown policy and corporate actions unchanged'],
         features_use_outcomes=False,budget_krw=3e9,fee_each_side=.0025,max_entry_days=10,
         liquidity='min(10% lagged ADV20,10% actual daily turnover), cash constrained; sell until liquidated',
         hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['model_panel.parquet','forecasts.csv','targets.csv','performance_choices.csv']}))
    full=build_features(old)
    cases=oldfr[oldfr.model.eq('baseline') & oldfr.phase.isin(['IS_forward','OOS'])].copy()
    cases=cases[cases.lead.eq(cases.index_id.map(choices.lead))]
    cases=cases[[r.phase!='IS_forward' or r.event in common[r.index_id] for r in cases.itertuples()]]
    forecasts=[];targets=[];probs=[];fits=[];coefficients=[];cache={}
    for phase in ['IS_forward','OOS']:
        for k,row in enumerate(cases[cases.phase.eq(phase)].sort_values(['index_id','information_date']).itertuples()):
            g=full[full.case_id.eq(row.case_id)];q=g[g.opportunity].copy()
            tr=full[full.index_id.eq(row.index_id)&full.year.le(2022)&full.effective_date.lt(row.information_date)&full.event.ne(row.event)&full.opportunity&full.actual_addition.notna()]
            assert tr.event.nunique()>=4 and tr.actual_addition.nunique()==2
            truth=set(g.loc[g.actual_addition.eq(True),'code']);n=int(row.predicted)
            for name,cols in {'baseline':[],**MODELS}.items():
                if name=='baseline': selected=set(g.loc[g.baseline_selected.eq(1)&g.opportunity,'code'])
                else:
                    key=(row.index_id,name,tuple(sorted(tr.event.unique())))
                    if key not in cache: cache[key]=fit(tr,cols)
                    mdl=cache[key];p=mdl.predict_proba(q[cols])[:,1]
                    selected=set(q.assign(p=p).sort_values(['p','projected_cap','code'],ascending=[False,False,True]).head(n).code)
                    fits.append(dict(case_id=row.case_id,index_id=row.index_id,model=name,phase=phase,
                         max_label_date=tr.effective_date.max(),information_date=row.information_date,
                         train_events=tr.event.nunique(),train_rows=len(tr),features=len(cols)))
                    probs.extend(dict(index_id=row.index_id,case_id=row.case_id,event=row.event,phase=phase,model=name,code=c,y=y,p=v)
                                 for c,y,v in zip(q.code,q.actual_addition,p))
                    if phase=='OOS' and not any(x['index_id']==row.index_id and x['model']==name for x in coefficients):
                        names=mdl[0].get_feature_names_out(cols)
                        coefficients.extend(dict(index_id=row.index_id,model=name,feature=c,standardized_coefficient=float(v)) for c,v in zip(names,mdl[-1].coef_[0]))
                assert len(selected)==n
                forecasts.append(dict(case_id=row.case_id,index_id=row.index_id,event=row.event,year=row.year,lead=row.lead,
                     phase=phase,model=name,hits=len(selected&truth),predicted=len(selected),actual=row.actual,
                     information_date=row.information_date,entry_date=row.entry_date,exit_date=row.exit_date))
                targets.extend(dict(case_id=row.case_id,model=name,code=c,weight=1/len(selected)) for c in sorted(selected))
            if k%5==0: print(phase,k,row.index_id,row.event,flush=True)
        if phase=='IS_forward':
            acc=summarize(pd.DataFrame(forecasts));picks=[]
            for idx,z in acc[acc.model.isin(['legacy','core','compact_rules','compact_rank','legacy_rules'])].groupby('index_id'):
                z=z.assign(features=z.model.map({m:len(c) for m,c in MODELS.items()}))
                b=z.sort_values(['f1','precision','features','model'],ascending=[False,False,True,True]).iloc[0]
                picks.append(b.to_dict())
            dump('IS_selection_freeze',dict(created_utc=datetime.now(timezone.utc).isoformat(),choices=picks,OOS_scoring_not_started=True))
    fr=pd.DataFrame(forecasts);tt=pd.DataFrame(targets)
    save('forecasts',fr);save('targets',tt);save('fit_audit',fits);save('coefficients',coefficients);save('accuracy',summarize(fr))
    pr=pd.DataFrame(probs);pr.to_parquet(DEST/'probabilities.parquet',index=False)
    pr['brier']=(pr.p-pr.y)**2;pr['logloss']=-(pr.y*np.log(pr.p.clip(1e-8,1-1e-8))+(1-pr.y)*np.log(1-pr.p.clip(1e-8,1-1e-8)))
    save('probability_quality',pr.groupby(['index_id','model','phase','event'])[['brier','logloss']].mean().groupby(['index_id','model','phase']).mean().reset_index())
    resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','rule_margin_resource');resource.write=lambda *a:None
    px,tv,actions=resource.resources()
    from corporate_actions import supplement
    px,tv,actions=supplement(px,tv,actions)
    ex=load('02_code/modules/index_forecast_development_20261117/execution.py','rule_margin_execution');ex.B=3e9
    results=[];days=[];events=[];ledgers=[]
    for (idx,model,phase),q in fr.groupby(['index_id','model','phase']):
        lead=int(q.lead.iloc[0]);start=pd.Timestamp(f'{int(q.year.min())}-01-01') if phase=='IS_forward' else pd.Timestamp('2023-01-02')
        end=pd.Timestamp('2022-12-30') if phase=='IS_forward' else pd.Timestamp('2026-06-30')
        ss=q[['case_id','index_id','event','lead','entry_date','exit_date']];aa=ss.assign(model=model,forecast_available=True)
        z,d,g,l=ex.simulate(idx,model,lead,start,end,ss,aa,tt[tt.model.eq(model)&tt.case_id.isin(ss.case_id)],px,tv,actions)
        meta=dict(index_id=idx,model=model,lead=lead,phase=phase,budget_krw=3e9);results.append(dict(**z,**meta))
        if d is not None:days.append(d.assign(**meta))
        if g is not None:events.append(g.assign(**meta))
        if l is not None and len(l):ledgers.append(l.assign(**meta))
        print('execution',idx,model,phase,z['status'],flush=True)
    save('performance',results)
    for name,items in [('daily',days),('events',events),('ledger',ledgers)]:pd.concat(items,ignore_index=True).to_parquet(DEST/(name+'.parquet'),index=False)
    validate()

def validate():
    p=pd.read_csv(DEST/'performance.csv');d=pd.read_parquet(DEST/'daily.parquet');e=pd.read_parquet(DEST/'events.parquet')
    l=pd.read_parquet(DEST/'ledger.parquet');l=l[l.side.isin(['buy','sell'])];f=pd.read_csv(DEST/'fit_audit.csv')
    c=pd.read_csv(DEST/'forecasts.csv');t=pd.read_csv(DEST/'targets.csv');full=pd.read_parquet(DEST/'model_panel.parquet')
    keys=['index_id','model','phase'];s=l.groupby(keys+['date','code']);pnl=e.groupby(keys).pnl.sum();last=d.sort_values('date').groupby(keys).equity.last()
    joined=t.merge(full[['case_id','code','opportunity']],on=['case_id','code'],validate='many_to_one')
    proto=json.loads((DEST/'protocol.json').read_text(encoding='utf8'))
    old=pd.read_csv(OUT/'complexity_audit/performance.csv');old=old[old.model.isin(['baseline','logistic_count'])].copy();old['model']=old.model.replace({'logistic_count':'legacy'})
    cmp=p.merge(old,on=keys,suffixes=('_new','_old'))
    checks=dict(all_execution_valid=bool(p.status.eq('valid').all()),cash_nonnegative=bool(d.cash.ge(-1e-4).all()),
        correct_fees=bool(np.allclose(l.fee,l.notional*.0025)),liquidity_respected=bool((s.notional.sum()<=s.limit.first()+1e-4).all()),
        pnl_matches_nav=bool(np.allclose(last.reindex(pnl.index)-3e9,pnl,atol=.001)),
        past_only_labels=bool((pd.to_datetime(f.max_label_date)<pd.to_datetime(f.information_date)).all()),
        labels_never_after_2022=bool(pd.to_datetime(f.max_label_date).dt.year.le(2022).all()),
        all_eligible=bool(joined.opportunity.all()),same_count=bool(c.groupby('case_id').predicted.nunique().eq(1).all()),
        baseline_and_legacy_reproduced=bool(len(cmp)==12 and np.allclose(cmp.sharpe_new,cmp.sharpe_old) and np.allclose(cmp.mean_event_net_new,cmp.mean_event_net_old)),
        sources_unchanged=all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==h for n,h in proto['hashes'].items()))
    dump('checks',checks)
    assert all(checks.values()),checks
    rows=[]
    contrasts=[('legacy_rules','legacy'),('compact_rules','core'),('compact_rules','legacy'),('compact_rank','compact_rules'),('legacy','legacy_no_rank')]
    for col in RULE:contrasts.append(('compact_rules','compact_without_'+col.removeprefix('rule_')))
    for (idx,phase),q in e.groupby(['index_id','phase']):
        wide=q.pivot(index='event',columns='model',values='net_return')
        for a,b in contrasts:
            diff=wide[a]-wide[b];yr=diff.index.str[:4];centered=pd.Series(diff.to_numpy()-diff.mean(),index=yr).groupby(level=0).sum();k=len(centered);n=len(diff)
            se=np.sqrt(k/(k-1)*(centered**2).sum()/n**2) if k>1 else np.nan
            tv=diff.mean()/se if se>0 else np.nan
            if k<=18:
                sums=pd.Series(diff.to_numpy(),index=yr).groupby(level=0).sum().to_numpy()
                flips=np.array([np.dot(sums,sg)/n for sg in itertools.product([-1,1],repeat=k)])
                sp=float((np.abs(flips)>=abs(diff.mean())-1e-12).mean())
            else:sp=np.nan
            rows.append(dict(index_id=idx,phase=phase,model=a,reference=b,n_events=n,year_clusters=k,
                 mean_difference=diff.mean(),better_events=int(diff.gt(1e-12).sum()),worse_events=int(diff.lt(-1e-12).sum()),
                 cluster_t=tv,cluster_p=2*stats.t.sf(abs(tv),k-1),year_signflip_p=sp,
                 largest_gain=diff.max(),largest_loss=diff.min(),leave_best_event_out_mean=diff.drop(diff.idxmax()).mean()))
    save('paired_improvement',rows)
    print(p[p.phase.eq('OOS')][['index_id','model','mean_event_net','sharpe','mdd']].to_string(index=False))

if __name__=='__main__':main()
