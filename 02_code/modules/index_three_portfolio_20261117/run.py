from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import engine

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT/'04_results/index_forecast_development_20261117'
OUT = ROOT/'04_results/index_three_portfolio_20261117'
LABELS = {'sharpe_selected': 'IS 샤프 선택', 'f1_selected': 'IS F1 선택',
          'f1_cap_third': 'IS F1 선택 + 종목당 1/3 상한'}


def unit_check():
    old = engine.BUDGET
    engine.BUDGET = 100.
    dates = pd.bdate_range('2020-01-01', periods=40)
    px = pd.DataFrame({'A000001': 1.}, index=dates)
    tv = px*1000
    choices = pd.DataFrame([dict(index_id=i, model='fixed', lead=40) for i in ['X','Y']])
    schedule = pd.DataFrame([dict(index_id=i, lead=40, case_id=i, event='2020-06',
                                 entry_date=dates[21], exit_date=dates[25]) for i in ['X','Y']])
    acc = schedule.assign(model='fixed', forecast_available=True)
    targets = schedule.assign(model='fixed', code='A000001', weight=1.)
    _, d, _, l, _, checks = engine.simulate(choices,schedule,acc,targets,px,tv,[],
                                           start=dates[21],end=dates[29])
    first = l[(l.date == dates[21]) & l.side.eq('buy')]
    assert len(first) == 2 and np.allclose(first.notional, 50.)
    assert l.groupby(['date','code']).notional.sum().max() <= 100+1e-9
    assert np.isclose(d[d.index_id.eq('COMBINED')].equity.iloc[-1], 200/(1+engine.FEE)*(1-engine.FEE))
    engine.BUDGET = old
    return dict(shared_limit=True, proportional_allocations=True, roundtrip_cost=True, **checks)


def make_plots(daily):
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False, 'figure.dpi':130})
    colors = {'KOSPI200':'#3164b1','KOSPI100':'#cd8730','KRX300':'#27846b','COMBINED':'#252a37'}
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for row, (scenario, label) in enumerate(LABELS.items()):
        for name, g in daily[daily.scenario.eq(scenario)].groupby('index_id'):
            g = g.sort_values('date'); b = 9e9 if name == 'COMBINED' else 3e9
            nav = g.equity/b
            dd = nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
            for ax, y in [(axes[row,0], nav), (axes[row,1], dd*100)]:
                ax.plot(g.date,y,label='통합 90억원' if name=='COMBINED' else name,
                        color=colors[name],lw=2.4 if name=='COMBINED' else 1.15,
                        alpha=1 if name=='COMBINED' else .8)
        axes[row,0].set_title(label+' | 원금 대비 평가자산')
        axes[row,1].set_title(label+' | 고점 대비 낙폭')
        axes[row,0].set_ylabel('NAV (시작=1)');axes[row,1].set_ylabel('낙폭 (%)')
        axes[row,0].legend(fontsize=8,ncol=2)
        for ax in axes[row]:ax.grid(alpha=.2)
    fig.suptitle('예측 편입주 롱온리 3전략 · 각 30억원 / 통합 90억원\nOOS 2023-01-02~2026-06-30 · 고정 투자원금, 이익 재투자 없음',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'NAV_drawdown.png');plt.close(fig)
    fig, axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
    for ax,(scenario,label) in zip(axes,LABELS.items()):
        g = daily[daily.scenario.eq(scenario)&~daily.index_id.eq('COMBINED')]
        p = g.pivot(index='date',columns='index_id',values='held_value')/9e9*100
        p = p[['KOSPI200','KOSPI100','KRX300']]
        ax.stackplot(p.index,p.T.to_numpy(),labels=p.columns,colors=[colors[x] for x in p.columns],alpha=.8)
        ax.set_title(label);ax.set_ylabel('주식 평가액 / 최초 원금 (%)');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,loc='upper left')
    fig.suptitle('실제 보유 포지션: 미체결·대기 현금은 주식 노출에 포함하지 않음',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'exposure.png');plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    protected = [DEV/'selected_IS_only.csv', DEV/'IS_grid.csv', DEV/'OOS_results.csv',
                 DEV/'capital_30eok/comparison.csv',DEV/'capital_30eok/daily.parquet',
                 DEV/'capital_30eok/ledger.parquet', DEV/'execution_targets.parquet',
                 ROOT/'04_results/index_engine_completion_20261117/forward/protocol.json']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    protocol = dict(budget_per_strategy_krw=3e9,total_principal_krw=9e9,
                    IS='2018-2022; KRX300 eligible evaluation begins 2018-12',
                    OOS=['2023-01-02','2026-06-30'],choice_source='Previously frozen IS selections',
                    scenarios=LABELS,allocation='30eok each; segregated cash; no reallocation or new OOS optimization',
                    capital='Fixed entry principal ceiling; profits retained in cash, losses reduce available cash',
                    cost_per_side=engine.FEE,participation='min(10% lagged ADV20,10% observed same-day turnover)',
                    shared_limit='All sleeves buy+sell same code/day share one limit; exits first, pro-rata within side',
                    entry_window=10,exit_window='Until liquidated; do not drop unfinished positions',
                    sharpe='Daily account returns, annualized sqrt(252), risk-free=0 for prior comparability',
                    cagr='Geometric annualized endpoint growth of account NAV; does not imply profit reinvestment',
                    forward_cap='f1_cap_third only: initial security target <=1/3 of own strategy budget, not entire portfolio',
                    limits=['KRX300 financial eligibility and latest full rules remain uncertified',
                            'Observed OOS has already been examined in prior research',
                            'Observed-price fills; nonlinear impact, order-book queue and integer-share conversion not newly modeled',
                            'Historical adjusted-price unit accounting and corporate-action assumptions inherited',
                            'No live orders and no forward registration modification'],source_hashes=hashes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    checks = {'synthetic':unit_check(),'replay':[]}
    spec=importlib.util.spec_from_file_location('portfolio_resources',ROOT/'02_code/modules/index_forecast_is_oos_20261117/evaluate.py')
    resources=importlib.util.module_from_spec(spec);sys.modules[spec.name]=resources;spec.loader.exec_module(resources)
    resources.write=lambda name, data:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    print('Loading frozen price / turnover / corporate actions',flush=True)
    px,tv,actions=resources.resources()
    schedule=pd.read_csv(ROOT/'04_results/index_forecast_is_oos_20261117/case_schedule.csv',parse_dates=['entry_date','exit_date','information_date'])
    acc=pd.read_csv(DEV/'accuracy.csv',low_memory=False);acc=acc[acc.train_end.eq(2022)]
    targets=pd.read_parquet(DEV/'execution_targets.parquet');targets=targets[targets.train_end.eq(2022)]
    choices=pd.read_csv(DEV/'selected_IS_only.csv');choices=choices[choices.train_end.eq(2022)]
    ref=pd.read_csv(DEV/'capital_30eok/comparison.csv')
    rd=pd.read_parquet(DEV/'capital_30eok/daily.parquet')
    rl=pd.read_parquet(DEV/'capital_30eok/ledger.parquet')
    summary=[];daily=[];events=[];ledgers=[];holdings=[];collisions=[];independent=[]
    for scenario,role,cap in [('sharpe_selected','IS_sharpe_selected',None),
                              ('f1_selected','IS_F1_selected',None),
                              ('f1_cap_third','IS_F1_selected',1/3)]:
        c=choices[choices.role.eq(role)].copy()
        assert len(c)==3
        c.assign(scenario=scenario,weight_cap=cap).to_csv(OUT/f'{scenario}_choices.csv',index=False)
        if cap is None:
            print('Replaying standalone accounts',scenario,flush=True)
            m,d,*_=engine.simulate(c,schedule,acc,targets,px,tv,actions,shared=False)
            for index in c.index_id:
                actual=d[d.index_id.eq(index)].set_index('date').equity
                prior=rd[(rd.index_id==index)&(rd.role==role)].set_index('date').equity
                assert actual.index.equals(prior.index)
                gap=float((actual-prior).abs().max());assert gap<.02,(index,scenario,gap)
                mm=m[m.index_id.eq(index)].iloc[0];rr=ref[(ref.index_id==index)&(ref.role==role)].iloc[0]
                for col in ['cagr','volatility','sharpe','mdd','total_return']:
                    assert np.isclose(mm[col],rr[col],atol=1e-10),(col,mm[col],rr[col])
                checks['replay'].append(dict(scenario=scenario,index_id=index,max_daily_equity_difference_krw=gap))
            independent.append(m.assign(scenario=scenario))
            trade=rl[(rl.role==role)&rl.side.isin(['buy','sell'])]
            col=trade.groupby(['date','code']).agg(notional=('notional','sum'),limit=('limit','min'),strategies=('index_id','nunique')).reset_index()
            col['excess']=(col.notional-col.limit).clip(lower=0)
            collisions.append(col.assign(scenario=scenario))
        print('Shared-liquidity portfolio',scenario,flush=True)
        m,d,g,l,h,check=engine.simulate(c,schedule,acc,targets,px,tv,actions,shared=True,weight_cap=cap)
        summary.append(m.assign(scenario=scenario));daily.append(d.assign(scenario=scenario))
        events.append(g.assign(scenario=scenario));ledgers.append(l.assign(scenario=scenario));holdings.append(h.assign(scenario=scenario))
        checks[scenario]=check
    s=pd.concat(summary,ignore_index=True);d=pd.concat(daily,ignore_index=True)
    g=pd.concat(events,ignore_index=True);l=pd.concat(ledgers,ignore_index=True);h=pd.concat(holdings,ignore_index=True)
    s.to_csv(OUT/'performance.csv',index=False,encoding='utf-8-sig');d.to_parquet(OUT/'daily.parquet',index=False)
    g.to_csv(OUT/'events.csv',index=False,encoding='utf-8-sig');l.to_parquet(OUT/'ledger.parquet',index=False)
    h.to_parquet(OUT/'holdings.parquet',index=False)
    pd.concat(independent).to_csv(OUT/'independent_reference.csv',index=False,encoding='utf-8-sig')
    pd.concat(collisions).to_csv(OUT/'independent_liquidity_audit.csv',index=False,encoding='utf-8-sig')
    monthly=[];annual=[];concentration=[]
    for (sc,index),q in d.groupby(['scenario','index_id']):
        q=q.sort_values('date').set_index('date');budget=9e9 if index=='COMBINED' else 3e9
        for freq,container in [('ME',monthly),('YE',annual)]:
            v=q.equity.resample(freq).last();opening=v.shift(1).fillna(budget)
            out=pd.DataFrame({'date':v.index,'profit_krw':(v-opening).to_numpy(),
                              'return_on_start_equity':(v/opening-1).to_numpy(),
                              'return_on_fixed_principal':((v-opening)/budget).to_numpy()})
            out['scenario']=sc;out['index_id']=index;out['partial_2026']=out.date.dt.year.eq(2026)
            container.append(out)
        if index=='COMBINED':
            hh=h[h.scenario.eq(sc)].groupby(['date','code']).agg(value=('market_value','sum'),strategies=('index_id','nunique')).reset_index()
            q.index=pd.to_datetime(q.index)
            hh['weight_in_nav']=hh.value/hh.date.map(q.equity)
            concentration.append(dict(scenario=sc,max_security_nav_weight=hh.weight_in_nav.max(),
                                      shared_holding_security_days=int(hh.strategies.gt(1).sum()),
                                      max_simultaneous_strategies=int(hh.strategies.max())))
            r=d[(d.scenario==sc)&(d.index_id!='COMBINED')].pivot(index='date',columns='index_id',values='return_daily')
            r.corr().to_csv(OUT/f'{sc}_return_correlation.csv')
    pd.concat(monthly).to_csv(OUT/'monthly.csv',index=False,encoding='utf-8-sig')
    pd.concat(annual).to_csv(OUT/'annual.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(concentration).to_csv(OUT/'concentration.csv',index=False)
    # Trace the four apparently conflicting Sharpe figures to their exact source rows.
    isg=pd.read_csv(DEV/'IS_grid.csv');isg=isg[(isg.train_end==2022)&(isg.index_id=='KRX300')]
    audit=[]
    for model,lead,beta in [('attractiveness',80,'IS Sharpe'),('joint_flow_variable',40,'IS F0.5'),('calibrated_same_count',40,'IS F1')]:
        r=isg[(isg.model==model)&(isg.lead==lead)].iloc[0].to_dict()
        r.update(phase='IS',budget_krw=3e8,selection_basis=beta);audit.append(r)
    for role in ['IS_sharpe_selected','IS_F1_selected']:
        r=ref[(ref.index_id=='KRX300')&(ref.role==role)].iloc[0].to_dict();r['phase']='OOS';audit.append(r)
        if role=='IS_sharpe_selected':
            a=r.copy();a['budget_krw']=3e8
            for col in ['cagr','volatility','sharpe','mdd','total_return']:a[col]=r[col+'_3eok']
            audit.append(a)
    pd.DataFrame(audit).to_csv(OUT/'KRX300_sharpe_source_audit.csv',index=False,encoding='utf-8-sig')
    for key,value in hashes.items():assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==value,key
    checks['source_files_unchanged']=True;checks['no_OOS_reselection']=True
    checks['all_checks_passed']=True
    (OUT/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    make_plots(d)
    print(s[['scenario','index_id','cagr','volatility','sharpe','mdd','profit_krw','position_days']].to_string(index=False),flush=True)


if __name__=='__main__':main()
