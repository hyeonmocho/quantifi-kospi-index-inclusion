"""Replay V18 model families on independent standalone windows and a common OOS account."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from build_vintage_submission_20261119 import load, ROOT
from backtest_vintage_engine_20261119 import FRAME_ROOT
import index_vintage_engine_20261119 as vintage

MOD = Path(__file__).resolve().parent
sys.path.insert(0, str(MOD / 'index_engine_v8_revalidation_20261118'))
from select_standalone_balanced import SPLIT, START, fit, ML, xfreq
from corporate_actions import supplement
R = ROOT / '04_results'
OUT = R / 'v18_long_vintage_replay'
OUT.mkdir(exist_ok=True)
LEADS = (80,60,40,20)
MODELS = {'KOSPI200':'shrunk_trend__count','KOSPI100':'legacy31__attractiveness',
          'KRX300':'calibration__attractiveness'}
E = load('02_code/modules/signal_a_rules_v2/engine.py','long_v18_krx_rules')
P = load('02_code/modules/paper_adaptation_20261117/model_engine.py','long_v18_paths')

def krx_fast(f, parent, rule):
    eligible = (f.index.isin(parent) & f.eligibility_state.eq('pass') &
                f.tv.gt(0) & f.mc.gt(0) & f.grp.notna())
    group = [f.index.get_indexer(g.index) for _,g in f[eligible].groupby('grp')]
    liquid = np.zeros(len(f),bool)
    for ix in group:
        liquid[ix] = f.tv.iloc[ix].rank(ascending=False,method='min') <= len(ix)*rule.liquidity_fraction
    current = f.cur.to_numpy(bool)
    def select(cap):
        selected=np.zeros(len(f),bool)
        for ix in group:
            order=ix[np.argsort(-cap[ix],kind='stable')]
            size=cap[order]
            mask=(size.cumsum()-size < rule.cap_coverage*size.sum())
            mask[:int(np.ceil(.3*len(order)))]=True
            selected[order]=mask & liquid[order]
        ranked=np.flatnonzero(eligible)
        ranked=ranked[np.argsort(-cap[ranked],kind='stable')]
        if len(ranked)>330:
            selected |= eligible.to_numpy() & current & liquid & (cap>=cap[ranked[329]])
        else:
            selected |= eligible.to_numpy() & current & liquid
        keep=ranked[selected[ranked]][:300]
        extra=ranked[~selected[ranked]&liquid[ranked]][:300-len(keep)]
        return np.r_[keep,extra]
    reference=vintage.select_krx300_with_parent(f,rule,parent)[0]
    assert set(f.index[select(f.mc.to_numpy())])==reference
    return select

def forecasts(px, include_early_signals=False):
    panel=pd.read_parquet(R/'two_axis_literal_20261119/model_panel.parquet')
    cohorts=json.loads((R/'standalone_balanced_literal_20261119/protocol.json').read_text())['IS_events']
    parents=pd.read_parquet(R/'index_method_vintage_20261119/tmi_parent_forecast_positive.parquet')
    paths=P.PaperModels(engines=SimpleNamespace(px=px))
    archive={}
    for idx,folder in FRAME_ROOT.items():
        if MODELS[idx]!='shrunk_trend__count':continue
        archived=pd.read_csv(folder.parent/'targets.csv',low_memory=False)
        archived=archived[archived.index_id.eq(idx)&archived.model.eq('shrunk_trend')]
        if 'train_end' in archived:archived=archived[archived.train_end.eq(SPLIT[idx])]
        archive.update(archived.groupby('case_id').code.agg(set).to_dict())
    audits=[];trends={};updated=[]
    for cid,g in panel.groupby('case_id',sort=True):
        r=g.iloc[0]; idx=r.index_id
        rule=vintage.vintage(idx,r.event,r.information_date,r.effective_date)
        status='historical_research_proxy'
        q=g.set_index('code')
        is_trend=MODELS[idx]=='shrunk_trend__count'
        if is_trend:
            trends[cid]=archive.get(cid,set())
        use_known=rule.cap_coverage is not None if idx=='KOSPI200' else (
            rule.incumbent_buffer is not None if idx=='KOSPI100' else cid in set(parents.case_id))
        if use_known:
            fp=FRAME_ROOT[idx]/(cid+'.pkl')
            if cid=='KOSPI100_2024-06_L20' and not fp.exists():
                fp=R/'two_axis_selection_20261118/frames'/fp.name
            obj=pd.read_pickle(fp)
            f=obj['frame'].copy()
            z=q.reindex(f.index)
            if idx in ('KOSPI200','KOSPI100'):
                profile=E.profile_for(r.event,asof=r.information_date)
                prepared=E.prepare_selector(f,profile)
                tv=f.tv.to_numpy()
            cur=z.current_member.fillna(False).to_numpy(bool)
            if idx=='KOSPI200':
                def select(cap):
                    return np.flatnonzero(prepared.additions_mask(prepared.select(cap,tv)))
            else:
                if idx=='KOSPI100':
                    valid=(f.eligibility_state.eq('pass') & f.tv.gt(0)&f.grp.notna()).to_numpy()
                    def select(cap):
                        pool=np.flatnonzero(prepared.select(cap,tv)&valid&(cap>0))
                        order=pool[np.argsort(-cap[pool],kind='stable')]
                        inc=order[:120][cur[order[:120]]]
                        new=order[:80][~cur[order[:80]]]
                        chosen=np.r_[inc,new][:100]
                        rest=order[cur[order]&~np.isin(order,chosen)]
                        return np.r_[chosen,rest[:100-len(chosen)]]
                else:
                    f['cur']=cur
                    f['eligibility_state']=z.eligibility.fillna('unknown')
                    parent=set(parents.loc[parents.case_id.eq(cid)&parents.predicted_parent,'code'])
                    select=krx_fast(f,parent,rule)
            if is_trend:
                adjust=(.25*np.log1p(z.momentum20.clip(lower=-.99))*z.remaining/40).fillna(0).clip(-.4,.4)
                cap=(z.past_sum+z.remaining*z.current_cap*np.exp(adjust))/z.full_days
                cap=cap.where(np.isfinite(cap)&cap.gt(0)&z.eligibility.eq('pass'),f.mc).to_numpy()
                ix=select(cap)
                trends[cid]=set(f.index[ix] if idx=='KOSPI200' else f.index[ix[~cur[ix]]])
            else:
                # Preserve the original path generator and its seed; only change the selection rule.
                missing=f.index.difference(paths.engines.px.columns)
                if len(missing):
                    paths.engines.px=paths.engines.px.reindex(columns=paths.engines.px.columns.union(missing,sort=False))
                caps,_=paths._simulate(obj['frame'],r.information_date,obj['projection'],idx,r.event)
                counts=np.zeros(len(f));base=None
                for j,cap in enumerate(caps):
                    ix=select(cap);ix=ix[~cur[ix]]
                    counts[ix]+=1
                    if j==0:base=set(f.index[ix])
                freq=pd.Series(counts/len(caps),index=f.index)
                g['raw_frequency']=g.code.map(freq)
                g['baseline_selected']=g.code.isin(base)&g.opportunity
                ranked=g[g.raw_frequency.gt(0)&g.opportunity].sort_values(
                    ['raw_frequency','baseline_selected','projected_cap','code'],ascending=[False,False,False,True])
                g['frequency_selected']=g.code.isin(ranked.head(int(g.baseline_selected.sum())).code)
                paths._simulation_cache.clear()
            status='dated_rule_partial' if 'partial' in rule.status else 'dated_rule_research'
        audits.append(dict(index_id=idx,case_id=cid,event=r.event,lead=r.lead,status=status,rule=rule.status))
        rr=px.loc[r.exit_date].reindex(g.code).to_numpy()/px.loc[r.entry_date].reindex(g.code).to_numpy()-1
        v=pd.Series(rr,index=g.index);g['edge']=v-v.groupby(g.group).transform('mean')
        updated.append(g)
    panel=pd.concat(updated).sort_values(['index_id','event','information_date','code'])
    prior=panel.groupby(['index_id','event','code'])
    panel['prior_frequency']=prior.raw_frequency.shift()
    panel['prior_baseline']=prior.baseline_selected.shift()
    panel['frequency_change']=panel.raw_frequency-panel.prior_frequency
    prob=panel.raw_frequency.clip(.005,.995)
    panel['logit_frequency']=np.log(prob/(1-prob))
    forecasts=[];targets=[];cache={}
    for cid,g in panel.groupby('case_id'):
        r=g.iloc[0];idx=r.index_id;model=MODELS[idx]
        is_train=r.year<=SPLIT[idx]
        if is_train and r.event not in cohorts[idx] and not include_early_signals:continue
        q=g[g.opportunity]
        base=set(q.loc[q.baseline_selected.astype(bool),'code'])
        if model=='shrunk_trend__count':trade=trends[cid]&set(q.code)
        else:
            tr=panel[panel.index_id.eq(idx)&panel.year.le(SPLIT[idx])&panel.effective_date.lt(r.information_date)&
                     panel.event.ne(r.event)&panel.opportunity&panel.actual_addition.notna()]
            rt=tr[tr.exit_date.lt(r.information_date)&tr.edge.notna()&tr.raw_frequency.gt(0)]
            if tr.event.nunique()<4 or tr.actual_addition.nunique()<2 or rt.event.nunique()<3:
                trade=base
            else:
                key=(idx,tuple(tr.event.unique()))
                if key not in cache:
                    if model=='legacy31__attractiveness':cache[key]=fit(tr,ML['legacy31'])
                    else:
                        tt=tr[tr.raw_frequency.gt(0)];w=1/tt.groupby('event').event.transform('size')
                        cache[key]=LogisticRegression(C=1,max_iter=500).fit(xfreq(tt.raw_frequency),tt.actual_addition,sample_weight=w*len(w)/w.sum())
                if model=='legacy31__attractiveness':p=cache[key].predict_proba(q[ML['legacy31']])[:,1]
                else:p=cache[key].predict_proba(xfreq(q.raw_frequency.fillna(0)))[:,1]
                means=[]
                for label in (0,1):
                    z=rt[rt.actual_addition.eq(label)].groupby('event').edge.mean()
                    means.append(z.sum()/(len(z)+5))
                trade=set(q.loc[(p>=.6)&q.raw_frequency.gt(0)&q.relative_momentum20.le(.30)&
                                (p*means[1]+(1-p)*means[0]>.005),'code'])
        forecasts.append(dict(index_id=idx,case_id=cid,model=model,event=r.event,lead=r.lead,phase='IS' if is_train else 'OOS',
                              entry_date=r.entry_date,exit_date=r.exit_date,forecast_available=True))
        targets.extend(dict(case_id=cid,model=model,code=c,weight=1/len(trade)) for c in sorted(trade))
    for name,data in [('forecasts',forecasts),('targets',targets),('rule_coverage',audits)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)

def main():
    threadpool_limits(1)
    resources=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','v18_prices')
    resources.write=lambda *a:None
    px,tv,actions=supplement(*resources.resources())
    if '--orders-only' not in sys.argv:forecasts(px)
    f=pd.read_csv(OUT/'forecasts.csv',parse_dates=['entry_date','exit_date'])
    t=pd.read_csv(OUT/'targets.csv')
    en=load('02_code/modules/index_three_portfolio_20261117/engine_selected_checkpoint.py','v18_order_engine')
    en.BUDGET=1e10
    def sim(q,mapping,start,end):
        ch=pd.DataFrame([dict(index_id=i,model=MODELS[i]) for i in mapping])
        return en.simulate(ch,q,f,t,px,tv,actions,start=start,end=end,lead_map=mapping)
    rows=[];days=[];chosen={};gaps=[]
    for idx in MODELS:
        for phase in ['IS','OOS','OOS7']:
            q=f[f.index_id.eq(idx)&f.phase.eq('IS' if phase=='IS' else 'OOS')]
            if phase=='OOS7':q=q[q.event.ge('2023-01')]
            start=START[idx] if phase=='IS' else ('2023-01-01' if phase=='OOS7' else f'{SPLIT[idx]+1}-01-01')
            end=f'{SPLIT[idx]}-12-31' if phase=='IS' else '2026-06-30'
            for n in range(1,5):
                leads=LEADS[:n];s=q[q.lead.isin(leads)]
                m,d,g,l,h,checks=sim(s,{idx:leads},start,end)
                assert checks['all_events_closed'] and len(g)==len(s)
                for ev,z in s.groupby('event'):
                    if set(z.lead)!=set(leads):gaps.append((idx,phase,n,ev,sorted(set(leads)-set(z.lead))))
                buys=l[l.side.eq('buy')].groupby('event').code.nunique().reindex(s.event.unique(),fill_value=0)
                row=dict(m.iloc[0],phase=phase,n_splits=n,reviews=s.event.nunique(),mean_names=buys.mean(),min_names=buys.min(),max_names=buys.max())
                rows.append(row);days.append(d[d.index_id.eq(idx)].assign(phase=phase,n_splits=n))
                print(idx,phase,n,row['reviews'],row['sharpe'],flush=True)
            if phase=='IS':chosen[idx]=max(rows[-4:],key=lambda r:r['sharpe'])['n_splits']
    pd.DataFrame(rows).to_csv(OUT/'performance.csv',index=False)
    pd.concat(days).to_parquet(OUT/'daily.parquet',index=False)
    pd.DataFrame(gaps,columns=['index_id','phase','n','event','missing_leads']).to_csv(OUT/'missing_signals.csv',index=False)
    (OUT/'selection.json').write_text(json.dumps(chosen))
    m,d,g,l,h,ch=sim(f[f.event.ge('2023-01')],{i:LEADS[:n] for i,n in chosen.items()},'2023-01-01','2026-06-30')
    assert ch['all_events_closed'] and ch['profit_matches_event_cashflows']
    assert g.groupby('index_id').event.nunique().eq(7).all()
    m.to_csv(OUT/'portfolio.csv',index=False)
    for name,data in [('daily',d),('holdings',h),('orders',l),('tranches',g)]:
        data.to_parquet(OUT/f'portfolio_{name}.parquet',index=False)
    verify()

def verify():
    f=pd.read_csv(OUT/'performance.csv')
    chosen=json.loads((OUT/'selection.json').read_text())
    p=pd.read_csv(OUT/'portfolio.csv').set_index('index_id')
    l=pd.read_parquet(OUT/'portfolio_orders.parquet')
    d=pd.read_parquet(OUT/'portfolio_daily.parquet')
    g=pd.read_parquet(OUT/'portfolio_tranches.parquet')
    by_day=l[l.side.isin(['buy','sell'])].groupby(['code','date']).agg(used=('notional','sum'),limit=('limit','min'))
    assert by_day.used.le(by_day.limit+.1).all() and d.cash.min()>-.1
    assert np.isclose(g.pnl.sum(),p.loc['COMBINED','profit_krw'],atol=.02,rtol=0)
    assert g.actual_exit.notna().all() and g.groupby('index_id').event.nunique().eq(7).all()
    for idx,counts in {'KOSPI200':(11,16),'KOSPI100':(10,16),'KRX300':(9,7)}.items():
        for phase,expected in zip(('IS','OOS','OOS7'),(*counts,7)):
            z=f[f.index_id.eq(idx)&f.phase.eq(phase)]
            assert len(z)==4 and z.reviews.eq(expected).all()
        z=f[f.index_id.eq(idx)&f.phase.eq('IS')]
        assert z.loc[z.sharpe.idxmax(),'n_splits']==chosen[idx]
    protocol={'IS_start':START,'IS_end_year':SPLIT,'standalone_OOS_start_year':{i:y+1 for i,y in SPLIT.items()},
              'combined_OOS':['2023-01-02','2026-06-30'],'models':MODELS,'selected_splits':chosen,
              'lead_sequence':LEADS,'fee_per_side':.0025,'capital_per_sleeve_krw':10000000000,
              'selection':'maximum daily account Sharpe on original index-specific IS cohorts',
              'sharpe_basis':'lagged annual KRW CD91, ACT/365, sample SD of net daily excess returns, sqrt(252); first interval one day; cash interest zero',
              'coverage':'early archived research proxies; use dated rules at checkpoints where then available',
              'missing_signal':'unallocated tranche remains cash; disclosed in missing_signals.csv',
              'historical_rank_feature':'KOSPI100 auxiliary rank-change indicator retained from historical inputs',
              'order_checks':{'closed_tranches':len(g),'max_limit_excess_krw':max(0.,(by_day.used-by_day.limit).max()),
                              'minimum_cash_krw':d.cash.min(),'ledger_profit_reconciled':True}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,default=str,indent=2),encoding='utf-8')

if __name__=='__main__':
    if '--verify-only' in sys.argv:verify()
    else:main()
