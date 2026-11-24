"""Fixed 30eok, equal execution, IS-only portfolio choice and paired broker audit."""
from research import *
from scipy import stats
import itertools
def event_stats(g):
    x=g.net_return.to_numpy(float);p=x[x>1e-12];n=x[x< -1e-12]
    t,pv=stats.ttest_1samp(x,0) if len(x)>1 and np.std(x)>0 else (np.nan,np.nan)
    year=g.event.str[:4];res=pd.Series(x-x.mean()).groupby(year.to_numpy()).sum();k=len(res)
    se=np.sqrt(k/(k-1)*np.square(res).sum()/len(x)**2) if k>1 else np.nan
    ct=x.mean()/se if se>0 else np.nan
    return dict(n_events=len(x),mean_net=x.mean(),median_net=np.median(x),win_rate=(x>1e-12).mean(),payoff=p.mean()/abs(n.mean()) if len(p) and len(n) else np.nan,t=t,p=pv,year_clusters=k,cluster_t=ct,cluster_p=2*stats.t.sf(abs(ct),k-1) if k>1 else np.nan)
def main():
    fr=pd.read_csv(OUT/'forecasts.csv',parse_dates=['information_date','entry_date','exit_date']);tr=pd.read_csv(OUT/'targets.csv')
    common=json.loads((OUT/'common_IS_events.json').read_text(encoding='utf8'))
    fr=fr[~fr.phase.eq('IS_forward')|fr.apply(lambda r:r.event in common[r.index_id],axis=1)].copy()
    picks=pd.read_csv(OUT/'frozen_choices.csv');iss=pd.read_csv(OUT/'IS_candidates.csv')
    resource=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','v7_resource');resource.write=lambda *a:None
    px,tv,actions=resource.resources()
    from corporate_actions import supplement
    px,tv,actions=supplement(px,tv,actions)
    ex=load('02_code/modules/index_forecast_development_20261117/execution.py','v7_execution');ex.B=3e9
    rows=[];daily=[];events=[];ledgers=[];cache={}
    def run(index,model,lead,phase):
        key=(index,model,int(lead),phase)
        if key in cache:return cache[key]
        f=fr[fr.index_id.eq(index)&fr.model.eq(model)&fr.lead.eq(lead)&fr.phase.eq(phase)].copy()
        if f.empty:return dict(status='no_forecasts')
        start=pd.Timestamp(f'{int(f.year.min())}-01-01') if phase=='IS_forward' else pd.Timestamp('2023-01-02')
        end=pd.Timestamp('2022-12-30') if phase=='IS_forward' else pd.Timestamp('2026-06-30')
        ss=f[['case_id','index_id','event','lead','entry_date','exit_date']];aa=ss.assign(model=model,forecast_available=True)
        tt=tr[tr.model.eq(model)&tr.case_id.isin(ss.case_id)]
        result,d,g,l=ex.simulate(index,model,int(lead),start,end,ss,aa,tt,px,tv,actions)
        result.update(index_id=index,model=model,lead=int(lead),phase=phase,budget_krw=3e9)
        rows.append(result);cache[key]=result
        if d is not None:daily.append(d.assign(index_id=index,model=model,lead=int(lead),phase=phase))
        if g is not None:events.append(g.assign(phase=phase))
        if l is not None and len(l):ledgers.append(l.assign(phase=phase))
        print('EXEC',phase,index,model,lead,result['status'],flush=True);return result
    # Performance candidate set is fixed: IS F1 leader per lead plus baseline per lead.
    menu=[]
    for (idx,lead),g in iss.groupby(['index_id','lead']):
        best=g.sort_values(['f1','precision','model'],ascending=[False,False,True]).iloc[0]
        menu.extend([(idx,'baseline',int(lead)),(idx,best.model,int(lead))])
    for idx,m,l in sorted(set(menu)):run(idx,m,l,'IS_forward')
    isresults=pd.DataFrame(rows);save('execution_IS_candidates',isresults)
    perf=[]
    for idx,g in isresults.groupby('index_id'):
        valid=g[g.status.eq('valid')&g.mean_event_net.gt(0)&g.sharpe.notna()]
        if valid.empty:perf.append(dict(index_id=idx,status='no_valid_positive_IS_portfolio'));continue
        b=valid.sort_values(['sharpe','model','lead'],ascending=[False,True,False]).iloc[0]
        perf.append(dict(index_id=idx,status='selected',model=b.model,lead=int(b.lead),IS_sharpe=b.sharpe,IS_mean_event_net=b.mean_event_net,valid_positive_candidates=len(valid),invalid_candidates=int(g.status.ne('valid').sum()),selection_scope='valid_positive_IS_candidates_only'))
    save('performance_choices',perf);dump('performance_selection_freeze',dict(candidates='IS F1 leader per lead plus baseline',selection='positive IS mean event net, then maximum calendar Sharpe',OOS_used_to_select=False,choices=perf))
    roles=[]
    for r in picks.itertuples():
        for role,m in [('prediction_selected',r.model),('baseline_same_lead','baseline')]:
            for phase in ['IS_forward','OOS']:
                z=run(r.index_id,m,int(r.lead),phase);roles.append(dict(role=role,**z))
    for p in perf:
        if p['status']=='selected':
            for phase in ['IS_forward','OOS']:roles.append(dict(role='IS_performance_selected',**run(p['index_id'],p['model'],p['lead'],phase)))
    save('portfolio_roles',roles);save('execution_all',rows)
    if daily:pd.concat(daily,ignore_index=True).to_parquet(OUT/'daily_accounts.parquet',index=False)
    if events:pd.concat(events,ignore_index=True).to_parquet(OUT/'execution_events.parquet',index=False)
    if ledgers:pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'execution_ledger.parquet',index=False)
    # Same-date broker pair, equal budget and fills. Multiple reports in one review cluster.
    roster=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/comparison/target_rosters.csv')
    bp=pd.read_csv(ROOT/'04_results/index_two_axes_20261117/broker/pairs.csv');bp=bp[bp.train_end.eq(2022)&bp.who.eq('broker')].set_index('report_id')
    chosen=picks.set_index('index_id').model.to_dict();br=[];brokercount=[]
    for r in fr[fr.phase.eq('BROKER_OOS')&fr.apply(lambda r:r.model==chosen[r.index_id],axis=1)].itertuples():
        ns=set(tr.loc[tr.case_id.eq(r.case_id)&tr.model.eq(r.model),'code']);bs=set(roster.loc[roster.report_id.eq(r.report_id)&roster.who.eq('broker')&roster.design_id.eq('available_date_refresh'),'code'])
        for who,names,hits in [('V8',ns,r.hits),('broker',bs,bp.loc[r.report_id,'hits'])]:
            ss=pd.DataFrame([dict(case_id='pair',index_id=r.index_id,event=r.event,lead=0,entry_date=r.entry_date,exit_date=r.exit_date)])
            aa=ss.assign(model='pair',forecast_available=True);tt=pd.DataFrame([dict(case_id='pair',model='pair',code=c,weight=1/len(names)) for c in names],columns=['case_id','model','code','weight'])
            result,d,g,l=ex.simulate(r.index_id,'pair',0,r.entry_date,min(pd.Timestamp('2026-06-30'),r.exit_date+pd.Timedelta(days=120)),ss,aa,tt,px,tv,actions)
            row=dict(report_id=r.report_id,index_id=r.index_id,event=r.event,who=who,predicted=len(names),hits=hits,actual=r.actual,status=result['status'],net_return=float(g.net_return.sum()) if g is not None else np.nan,entry_date=r.entry_date,exit_date=r.exit_date,budget_krw=3e9)
            br.append(row);brokercount.append({k:row[k] for k in ['report_id','index_id','event','who','predicted','hits','actual']})
    save('broker_pairs',br);bsum=[];infer=[];b=pd.DataFrame(br)
    for (idx,who),g in b.groupby(['index_id','who']):
        counts=summary(g);good=set(b.groupby('report_id').filter(lambda z:len(z)==2 and z.status.eq('valid').all()).report_id)
        vals=g[g.report_id.isin(good)].groupby('event',as_index=False).net_return.mean()
        bsum.append(dict(index_id=idx,who=who,n_reports=len(g),**counts,**({'mean_net':vals.net_return.mean(),'win_rate':(vals.net_return>0).mean(),'payoff':vals.loc[vals.net_return>0,'net_return'].mean()/abs(vals.loc[vals.net_return<0,'net_return'].mean()),'n_performance_events':len(vals)} if len(vals) else {})))
    for idx,g in b.groupby('index_id'):
        good=g.groupby('report_id').filter(lambda z:len(z)==2 and z.status.eq('valid').all())
        w=good.groupby(['event','who']).net_return.mean().unstack();delta=(w.V8-w.broker).dropna().to_numpy()
        if len(delta):
            perms=[np.mean(delta*np.array(sign)) for sign in itertools.product([-1,1],repeat=len(delta))]
            infer.append(dict(index_id=idx,n_review_clusters=len(delta),mean_net_difference=delta.mean(),paired_signflip_p=float((np.abs(perms)>=abs(delta.mean())-1e-15).mean()),assumption='symmetric paired review differences; reused historical OOS'))
    save('broker_summary',bsum);save('broker_paired_inference',infer)
    # Registry coverage is separate from statistical comparisons.
    registry=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261117_expand/comparison/report_cases.csv')
    registry['v8_comparison']=registry.report_id.isin(b.report_id)
    registry['v8_exclusion_reason']=np.select([registry.v8_comparison,~registry.status.eq('comparison_attempted'),~registry.roster_complete.eq(True),registry.event.lt('2023-01')|registry.event.gt('2026-06')],['included','source_or_timing_exclusion','incomplete_named_subset','outside_frozen_OOS'],default='missing_same_date_frame')
    registry.to_csv(OUT/'broker_registry_audit.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(roles).to_string(index=False),flush=True);print(pd.DataFrame(bsum).to_string(index=False),flush=True)
if __name__=='__main__':main()
