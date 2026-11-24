"""Compare frozen IS Sharpe/F1/F0.5 choices on equal-principal execution.
No beta/model/timing re-selection on OOS and no change to registered choices.
"""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DEV=ROOT/'04_results/index_forecast_development_20261117'
OUT=ROOT/'04_results/index_three_portfolio_20261117/selection_comparison'


def load(path,name):
    s=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def main():
    OUT.mkdir(exist_ok=True)
    ex=load('02_code/modules/index_forecast_development_20261117/execution.py','selection_execution')
    base=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','selection_resources')
    base.write=lambda n,d:pd.DataFrame(d).to_csv(OUT/f'{n}.csv',index=False,encoding='utf-8-sig')
    sources=[DEV/'selected_IS_only.csv',DEV/'IS_grid.csv',DEV/'capital_30eok/comparison.csv',
             ROOT/'04_results/index_engine_completion_20261117/selection_financial_diagnostic/IS_Fbeta_selection_diagnostic.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    choices=pd.read_csv(sources[0]);choices=choices[(choices.train_end==2022)&choices.index_id.isin(['KOSPI200','KOSPI100'])&choices.role.isin(['IS_F1_selected','IS_sharpe_selected'])].copy()
    f=pd.read_csv(sources[3]);f=f[(f.beta==.5)&f.index_id.isin(['KOSPI200','KOSPI100'])]
    f=f[['train_end','index_id','model','lead']].assign(role='IS_F05_selected')
    choices=pd.concat([choices,f],ignore_index=True)
    choices.to_csv(OUT/'frozen_choices.csv',index=False,encoding='utf-8-sig')
    protocol=dict(original_selection_budget=3e8,rerun_budgets=[3e8,3e9],selection_refit=False,
                  IS=['2018-01-01','2022-12-31'],OOS=['2023-01-01','2026-06-30'],
                  role='standalone sleeves, not the cross-strategy shared-liquidity portfolio',
                  labels='precision=actual additions/predicted additions; recall=hits/all actual regular additions',
                  weights='unchanged stored research target weights; no new 1/3 cap',
                  fees_per_side=.0025,entry_window=10,exit='until filled',
                  multiple_comparison='already-observed OOS diagnostic, not new untouched OOS',source_hashes=hashes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    px,tv,actions=base.resources()
    schedule=pd.read_csv(ROOT/'04_results/index_forecast_is_oos_20261117/case_schedule.csv',parse_dates=['entry_date','exit_date','information_date'])
    acc=pd.read_csv(DEV/'accuracy.csv',low_memory=False);acc=acc[acc.train_end.eq(2022)]
    targets=pd.read_parquet(DEV/'execution_targets.parquet');targets=targets[targets.train_end.eq(2022)]
    isref=pd.read_csv(DEV/'IS_grid.csv');oosref=pd.read_csv(DEV/'capital_30eok/comparison.csv')
    rows=[];daily=[];events=[];checks=[];cache={}
    for r in choices.itertuples():
        for phase,start,end in [('IS','2018-01-01','2022-12-31'),('OOS','2023-01-01','2026-06-30')]:
            for budget in [3e8,3e9]:
                key=(r.index_id,r.model,int(r.lead),phase,budget)
                if key not in cache:
                    ex.B=budget
                    m,d,g,l=ex.simulate(r.index_id,r.model,int(r.lead),pd.Timestamp(start),pd.Timestamp(end),schedule,acc,targets,px,tv,actions)
                    assert m['status']=='valid', (key,m)
                    ac=base.accuracy(r.index_id,r.model,int(r.lead),pd.Timestamp(start),pd.Timestamp(end),schedule,acc)
                    assert ac['n_labelled_forecasts']==m['n_events']
                    old=isref[(isref.train_end==2022)&(isref.index_id==r.index_id)&(isref.model==r.model)&(isref.lead==r.lead)] if phase=='IS' and budget==3e8 else oosref[(oosref.index_id==r.index_id)&(oosref.model==r.model)&(oosref.lead==r.lead)] if phase=='OOS' and budget==3e9 else pd.DataFrame()
                    if len(old):
                        for col in ['cagr','sharpe','volatility','mdd','total_return']:assert np.isclose(m[col],old.iloc[0][col],atol=1e-10),(key,col)
                    lim=l[l.side.isin(['buy','sell'])].groupby(['date','code']).agg(notional=('notional','sum'),limit=('limit','min'))
                    assert (lim.notional<=lim.limit+.02).all() and (d.cash>=-.001).all()
                    assert abs(g.pnl.sum()-(d.equity.iloc[-1]-budget))<.02
                    checks.append(dict(index_id=r.index_id,model=r.model,lead=r.lead,phase=phase,budget_krw=budget,reference_reproduced=bool(len(old)),ledger_passed=True))
                    cache[key]=(m,d,g,ac)
                    print(key,'Sharpe',round(m['sharpe'],4),flush=True)
                m,d,g,ac=cache[key]
                ident=dict(index_id=r.index_id,model=r.model,lead=r.lead,role=r.role,phase=phase,budget_krw=budget)
                rows.append(dict(ident,**m,**{'prediction_'+k:v for k,v in ac.items()},profit_krw=d.equity.iloc[-1]-budget))
                daily.append(d.assign(**ident));events.append(g.assign(**ident))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'performance_and_accuracy.csv',index=False,encoding='utf-8-sig')
    pd.concat(daily,ignore_index=True).to_parquet(OUT/'daily.parquet',index=False)
    pd.concat(events,ignore_index=True).to_csv(OUT/'events.csv',index=False,encoding='utf-8-sig')
    for path,digest in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    (OUT/'validation.json').write_text(json.dumps(dict(all_passed=True,source_files_unchanged=True,unique_runs=len(cache),checks=checks),ensure_ascii=False,indent=2),encoding='utf8')
    print(result[result.budget_krw.eq(3e9)][['index_id','role','phase','prediction_precision','prediction_recall','cagr','sharpe','mdd','win_rate','payoff']].to_string(index=False))


if __name__=='__main__':main()
