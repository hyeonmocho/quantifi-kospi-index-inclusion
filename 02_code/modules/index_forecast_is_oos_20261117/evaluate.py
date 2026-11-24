from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'04_results/index_forecast_is_oos_20261117'
B=300000000.;FEE=.0025
def write(n,d):pd.DataFrame(d).to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')
def resources():
 c=pd.read_pickle(ROOT/'01_data/processed/signal_a_cache.pkl');px=c['px'].astype(float);tv=c['tv'].astype(float)
 actions=[dict(old='A079440',new='A055550',halt='2020-01-22',effective='2020-01-28',delivery='2020-02-14',ratio=.6601483,old_ref='2020-01-21'),
          dict(old='A000060',new='A138040',halt='2023-01-30',effective='2023-02-01',delivery='2023-02-21',ratio=1.2657378,old_ref='2023-01-27'),
          dict(old='A119860',new=None,halt='2024-09-04',effective='2024-09-06',delivery='2024-09-25',cash=18000.,old_ref='2024-09-03',cash_fee_rate=.0035),
          dict(old='A085370',new=None,halt='2023-10-11',effective='2023-10-13',delivery='2023-10-31',cash=36700.,old_ref='2023-10-10',cash_fee_rate=.0035,source='DART 20231024000203; scheduled payment, not broker cash ledger')]
 raw=pd.read_parquet(ROOT/'01_data/processed/index_data_extension_20261117/raw_quotes_with_turnover.parquet')
 rr=[]
 for f in (OUT/'corporate_raw').glob('*.json'):
  for r in json.loads(f.read_text(encoding='utf-8'))['OutBlock_1']:
   rr.append(dict(date=pd.Timestamp(r['BAS_DD']),code='A'+str(r['ISU_CD']).zfill(6),close=float(str(r['TDD_CLSPRC']).replace(',',''))))
 raw=pd.concat([raw[['date','code','close']],pd.DataFrame(rr)]).drop_duplicates(['date','code'],keep='last').set_index(['date','code'])
 for a in actions:
  for k in ['halt','effective','delivery','old_ref']:a[k]=pd.Timestamp(a[k])
  old_raw=raw.loc[(a['old_ref'],a['old']),'close'];old_factor=px.loc[a['old_ref'],a['old']]/old_raw
  a['old_adjusted_to_raw_factor']=old_factor
  if a['new']:
   new_raw=raw.loc[(a['delivery'],a['new']),'close'];new_factor=px.loc[a['delivery'],a['new']]/new_raw
   a['conversion']=old_factor*a['ratio']/new_factor
  else:a['cash_per_adjusted_unit']=a['cash']*old_factor
  assert old_factor>0
  # Original security cannot trade from its documented permanent suspension onward.
  tv.loc[tv.index>=a['halt'],a['old']]=0.
 write('corporate_action_reconciliation',actions)
 return px,tv,actions
def simulate_v1_single_event(index,model,lead,start,end,schedule,acc,targets,px,tv,actions):
 ss=schedule[(schedule.index_id==index)&(schedule.lead==lead)&schedule.entry_date.between(start,end)].copy()
 if index=='KRX300':ss=ss[ss.event>='2018-12']
 aa=acc[(acc.index_id==index)&(acc.model==model)&(acc.lead==lead)&acc.case_id.isin(ss.case_id)]
 if len(ss)!=len(aa) or not aa.forecast_available.all():return dict(status='unavailable_forecast',n_events=len(ss),missing_cases=';'.join(aa.loc[~aa.forecast_available,'case_id'])),None,None,None
 tr=targets[(targets.model==model)&targets.case_id.isin(ss.case_id)]
 codes=sorted(set(tr.code)|{a['new'] for a in actions if a['new']})
 P=px.reindex(columns=codes);T=tv.reindex(columns=codes);A=T.rolling(20,min_periods=20).mean().shift(1)
 days=px.index[(px.index>=start)&(px.index<=end)]
 ci={c:i for i,c in enumerate(codes)};positions=np.zeros(len(codes));remaining=np.zeros(len(codes));cash=B
 current=None;exitday=None;deadline=None;entry_equity=None;entry_budget=0;paid=0.;bought=0.;events=[];daily=[];ledger=[];prev=B
 last=P.loc[P.index<start].ffill().iloc[-1].to_numpy() if (P.index<start).any() else np.full(len(codes),np.nan)
 startmap={r.entry_date:r for r in ss.itertuples()};live=[a for a in actions if a['old'] in ci]
 def eq(p):return cash+np.nansum(positions*p)
 for date in days:
  prices=P.loc[date].to_numpy().copy();turn=T.loc[date].to_numpy();adv=A.loc[date].to_numpy()
  for a in live:
   j=ci[a['old']]
   if a['halt']<=date<a['effective']:prices[j]=last[j]
   if a['effective']<=date<a['delivery']:
    prices[j]=prices[ci[a['new']]]*a['conversion'] if a['new'] else a['cash_per_adjusted_unit']
   if date==a['delivery'] and positions[j]>0:
    amount=positions[j]
    if a['new']:positions[ci[a['new']]]+=amount*a['conversion']
    else:cash+=amount*a['cash_per_adjusted_unit']
    positions[j]=0.;ledger.append(dict(date=date,index_id=index,model=model,lead=lead,event=current,code=a['old'],side='corporate_exchange_or_cash_delivery',notional=np.nan,fee=0.,scenario_cash_delivery_date=a['new'] is None))
  if ((positions>1e-10)&(~np.isfinite(prices)|(prices<=0))).any():
   return dict(status='missing_held_price',missing_cases=str(date),n_events=len(ss)),None,None,None
  if date in startmap:
   r=startmap[date]
   if positions.sum()>1e-8:return dict(status='unliquidated_previous_event',n_events=len(ss),missing_cases=str(date),held_codes=';'.join(codes[j] for j in np.flatnonzero(positions>1e-10)),previous_event=current),None,None,None
   current=r.event;exitday=r.exit_date;deadline=px.index[min(px.index.get_loc(date)+9,len(px)-1)]
   entry_equity=cash;entry_budget=min(B,max(cash,0.));remaining[:]=0.;paid=0.;bought=0.
   names=tr.loc[tr.case_id.eq(r.case_id),'code'].tolist()
   if names:remaining[[ci[c] for c in names]]=entry_budget/(1+FEE)/len(names)
  valid=np.isfinite(prices)&(prices>0)&np.isfinite(adv)&(adv>0)&np.isfinite(turn)&(turn>0)
  limit=np.where(valid,np.minimum(.1*adv,.1*turn),0.)
  if current is not None:
   side='sell' if date>=exitday else 'buy'
   demand=np.nan_to_num(positions*prices,nan=0.) if side=='sell' else remaining if date<=deadline else np.zeros(len(codes))
   notion=np.minimum(demand,limit)
   if side=='buy':
    need=notion.sum()*(1+FEE)
    if need>0 and need>max(cash,0.):notion*=max(cash,0.)/need
   units=notion/np.where(valid,prices,1.);fees=notion*FEE;paid+=fees.sum()
   if side=='buy':positions+=units;cash-=(notion+fees).sum();remaining=np.maximum(0.,remaining-notion);bought+=notion.sum()
   else:positions=np.maximum(0.,positions-units);cash+=(notion-fees).sum()
   assert cash>=-1e-4 and (notion<=limit+1e-5).all()
   if -1e-4<cash<0:cash=0.  # sub-won floating-point roundoff only
   for j in np.flatnonzero(notion>1e-5):ledger.append(dict(date=date,index_id=index,model=model,lead=lead,event=current,code=codes[j],side=side,notional=notion[j],fee=fees[j],limit=limit[j]))
   if date>=exitday and positions.sum()<1e-8:
    events.append(dict(index_id=index,model=model,lead=lead,event=current,net_return=(cash-entry_equity)/B,pnl=cash-entry_equity,
      starting_equity=entry_equity,fixed_budget=B,allocation=entry_budget,entry_notional=bought,fees=paid,actual_exit=date,planned_exit=exitday))
    current=None;remaining[:]=0.
  equity=eq(prices);daily.append(dict(date=date,equity=equity,return_daily=equity/prev-1,exposure=np.nansum(positions*prices)/B,any_position=positions.sum()>1e-8,cash=cash));prev=equity
  last=np.where(np.isfinite(prices)&(prices>0),prices,last)
 if positions.sum()>1e-8 or len(events)!=len(ss):return dict(status='unliquidated_period_end',n_events=len(ss)),None,None,None
 d=pd.DataFrame(daily);nav=d.equity/B;ret=d.return_daily;years=(d.date.iloc[-1]-d.date.iloc[0]).days/365.25
 x=np.array([x['net_return'] for x in events]);pos=x[x>1e-12];neg=x[x< -1e-12]
 G=pd.DataFrame(events);groups=G.event.str[:4];res=x-x.mean();sums=pd.Series(res).groupby(groups.to_numpy()).sum();g=len(sums)
 se=np.sqrt(g/(g-1)*(sums**2).sum()/len(x)**2) if g>1 else np.nan
 t=x.mean()/se if se>0 else np.nan;p=2*stats.t.sf(abs(t),g-1) if g>1 else np.nan
 info=dict(status='valid',n_events=len(x),n_year_clusters=g,period_start=d.date.iloc[0],period_end=d.date.iloc[-1],total_return=nav.iloc[-1]-1,
  cagr=nav.iloc[-1]**(1/years)-1,sharpe=ret.mean()/ret.std(ddof=1)*np.sqrt(252),volatility=ret.std(ddof=1)*np.sqrt(252),
  mdd=(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1).min(),mean_event_net=x.mean(),median_event_net=np.median(x),win_rate=(x>1e-12).mean(),
  payoff=pos.mean()/abs(neg.mean()) if len(pos) and len(neg) else np.nan,year_cluster_t=t,year_cluster_p=p,
  mean_exposure=d.exposure.mean(),position_days=int(d.any_position.sum()),n_sessions=len(d),total_fees=sum(z['fees'] for z in events))
 return info,d,pd.DataFrame(events),pd.DataFrame(ledger)

def simulate(index,model,lead,start,end,schedule,acc,targets,px,tv,actions):
 """Cash-constrained overlapping event lots; share the daily limit by security."""
 ss=schedule[(schedule.index_id==index)&(schedule.lead==lead)&schedule.entry_date.between(start,end)].copy()
 if index=='KRX300':ss=ss[ss.event>='2018-12']
 aa=acc[(acc.index_id==index)&(acc.model==model)&(acc.lead==lead)&acc.case_id.isin(ss.case_id)]
 if len(ss)!=len(aa) or not aa.forecast_available.all():return dict(status='unavailable_forecast',n_events=len(ss),missing_cases=';'.join(aa.loc[~aa.forecast_available,'case_id'])),None,None,None
 tr=targets[(targets.model==model)&targets.case_id.isin(ss.case_id)]
 codes=sorted(set(tr.code)|{a['new'] for a in actions if a['new']});ci={c:i for i,c in enumerate(codes)}
 P=px.reindex(columns=codes);T=tv.reindex(columns=codes);A=T.rolling(20,min_periods=20).mean().shift(1)
 days=px.index[(px.index>=start)&(px.index<=end)];cash=B;prev=B;lots=[];events=[];daily=[];ledger=[];overlap_entries=0
 last=P.loc[P.index<start].ffill().iloc[-1].to_numpy() if (P.index<start).any() else np.full(len(codes),np.nan)
 startmap={r.entry_date:r for r in ss.itertuples()};live=[a for a in actions if a['old'] in ci]
 def total_positions():return sum((z['positions'] for z in lots),np.zeros(len(codes)))
 for date in days:
  prices=P.loc[date].to_numpy().copy();turn=T.loc[date].to_numpy();adv=A.loc[date].to_numpy()
  for a in live:
   j=ci[a['old']]
   if a['halt']<=date<a['effective']:prices[j]=last[j]
   if a['effective']<=date<a['delivery']:prices[j]=prices[ci[a['new']]]*a['conversion'] if a['new'] else a['cash_per_adjusted_unit']
   if date==a['delivery']:
    for z in lots:
     amount=z['positions'][j]
     if amount<=0:continue
     delivered=0.;cash_fee=0.
     if a['new']:z['positions'][ci[a['new']]]+=amount*a['conversion']
     else:
      delivered=amount*a['cash_per_adjusted_unit'];cash_fee=delivered*a.get('cash_fee_rate',0.)
      cash+=delivered-cash_fee;z['net_cashflow']+=delivered-cash_fee;z['fees']+=cash_fee
     z['positions'][j]=0.
     ledger.append(dict(date=date,index_id=index,model=model,lead=lead,event=z['event'],code=a['old'],side='corporate_exchange_or_cash_delivery',notional=np.nan,fee=cash_fee,cash_delivery=delivered,scenario_cash_delivery_date=a['new'] is None))
  pos=total_positions()
  if ((pos>1e-10)&(~np.isfinite(prices)|(prices<=0))).any():return dict(status='missing_held_price',missing_cases=str(date),n_events=len(ss),held_codes=';'.join(codes[j] for j in np.flatnonzero((pos>1e-10)&~np.isfinite(prices)))),None,None,None
  if date in startmap:
   r=startmap[date];held_value=np.nansum(pos*prices)
   if held_value>1e-4:overlap_entries+=1
   budget=min(max(cash,0.),max(B-held_value,0.))
   z=dict(event=r.event,exit_date=r.exit_date,deadline=px.index[min(px.index.get_loc(date)+9,len(px)-1)],positions=np.zeros(len(codes)),remaining=np.zeros(len(codes)),starting_equity=cash+held_value,allocation=budget,entry_notional=0.,fees=0.,net_cashflow=0.,overlap_held_value=held_value)
   names=tr.loc[tr.case_id.eq(r.case_id),'code'].tolist()
   if names:z['remaining'][[ci[c] for c in names]]=budget/(1+FEE)/len(names)
   lots.append(z)
  valid=np.isfinite(prices)&(prices>0)&np.isfinite(adv)&(adv>0)&np.isfinite(turn)&(turn>0)
  limit=np.where(valid,np.minimum(.1*adv,.1*turn),0.);used=np.zeros(len(codes))
  # Exits have priority; combined buy+sell amounts across lots share one limit.
  for side in ['sell','buy']:
   for z in lots:
    if side=='sell' and date<z['exit_date']:continue
    if side=='buy' and (date>=z['exit_date'] or date>z['deadline']):continue
    demand=np.nan_to_num(z['positions']*prices,nan=0.) if side=='sell' else z['remaining']
    notion=np.minimum(demand,np.maximum(0.,limit-used))
    if side=='buy':
     need=notion.sum()*(1+FEE)
     if need>0 and need>max(cash,0.):notion*=max(cash,0.)/need
    used+=notion;units=notion/np.where(valid,prices,1.);fees=notion*FEE;z['fees']+=fees.sum()
    if side=='buy':
     z['positions']+=units;cash-=(notion+fees).sum();z['net_cashflow']-=(notion+fees).sum();z['remaining']=np.maximum(0.,z['remaining']-notion);z['entry_notional']+=notion.sum()
    else:z['positions']=np.maximum(0.,z['positions']-units);cash+=(notion-fees).sum();z['net_cashflow']+=(notion-fees).sum()
    assert cash>=-1e-4 and (used<=limit+1e-5).all()
    if -1e-4<cash<0:cash=0.
    for j in np.flatnonzero(notion>1e-5):ledger.append(dict(date=date,index_id=index,model=model,lead=lead,event=z['event'],code=codes[j],side=side,notional=notion[j],fee=fees[j],limit=limit[j]))
  remaining_lots=[]
  for z in lots:
   if date>=z['exit_date'] and z['positions'].sum()<1e-8:
    events.append(dict(index_id=index,model=model,lead=lead,event=z['event'],net_return=z['net_cashflow']/B,pnl=z['net_cashflow'],starting_equity=z['starting_equity'],fixed_budget=B,allocation=z['allocation'],entry_notional=z['entry_notional'],fees=z['fees'],actual_exit=date,planned_exit=z['exit_date'],overlap_held_value=z['overlap_held_value']))
   else:remaining_lots.append(z)
  lots=remaining_lots;pos=total_positions();held_value=np.nansum(pos*prices);equity=cash+held_value
  daily.append(dict(date=date,equity=equity,return_daily=equity/prev-1,exposure=held_value/B,any_position=pos.sum()>1e-8,cash=cash,active_event_lots=len(lots)));prev=equity
  last=np.where(np.isfinite(prices)&(prices>0),prices,last)
 if lots or len(events)!=len(ss):return dict(status='unliquidated_period_end',n_events=len(ss),held_codes=';'.join(codes[j] for j in np.flatnonzero(total_positions()>1e-10))),None,None,None
 d=pd.DataFrame(daily);nav=d.equity/B;ret=d.return_daily;years=(d.date.iloc[-1]-d.date.iloc[0]).days/365.25
 G=pd.DataFrame(events);x=G.net_return.to_numpy();pos=x[x>1e-12];neg=x[x< -1e-12]
 groups=G.event.str[:4];res=x-x.mean();sums=pd.Series(res).groupby(groups.to_numpy()).sum();g=len(sums)
 se=np.sqrt(g/(g-1)*(sums**2).sum()/len(x)**2) if g>1 else np.nan;t=x.mean()/se if se>0 else np.nan;p=2*stats.t.sf(abs(t),g-1) if g>1 else np.nan
 sd=ret.std(ddof=1)
 info=dict(status='valid',n_events=len(x),n_year_clusters=g,period_start=d.date.iloc[0],period_end=d.date.iloc[-1],total_return=nav.iloc[-1]-1,cagr=nav.iloc[-1]**(1/years)-1,sharpe=ret.mean()/sd*np.sqrt(252) if sd>0 else np.nan,volatility=sd*np.sqrt(252),mdd=(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1).min(),mean_event_net=x.mean(),median_event_net=np.median(x),win_rate=(x>1e-12).mean(),payoff=pos.mean()/abs(neg.mean()) if len(pos) and len(neg) else np.nan,year_cluster_t=t,year_cluster_p=p,mean_exposure=d.exposure.mean(),position_days=int(d.any_position.sum()),n_sessions=len(d),total_fees=G.fees.sum(),overlap_entries=overlap_entries)
 traded_events=G[G.entry_notional>1e-4]
 info.update(n_traded_events=len(traded_events),win_rate_traded=float((traded_events.net_return>1e-12).mean()) if len(traded_events) else np.nan)
 return info,d,G,pd.DataFrame(ledger)
def accuracy(index,model,lead,start,end,schedule,acc):
 s=schedule[(schedule.index_id==index)&(schedule.lead==lead)&schedule.entry_date.between(start,end)]
 if index=='KRX300':s=s[s.event>='2018-12']
 a=acc[(acc.index_id==index)&(acc.model==model)&(acc.lead==lead)&acc.case_id.isin(s.case_id)]
 good=a[a.forecast_available&a.label_available];hits=good.hits.sum();pred=good.predicted.sum();act=good.actual.sum()
 return dict(n_scheduled=len(a),n_forecasts=int(a.forecast_available.sum()),n_labelled_forecasts=len(good),predicted=pred,actual=act,hits=hits,
  precision=hits/pred if pred else np.nan,recall=hits/act if act else np.nan,f1=2*hits/(pred+act) if pred+act else np.nan)
def main():
 spec=json.loads((OUT/'protocol_frozen_before_selection.json').read_text(encoding='utf-8'))
 acc=pd.read_csv(OUT/'prediction_accuracy.csv');targets=pd.read_csv(OUT/'predicted_targets.csv');schedule=pd.read_csv(OUT/'case_schedule.csv',parse_dates=['entry_date','exit_date','information_date'])
 expected=len(schedule)*3
 assert len(acc)==expected, f'Forecast run incomplete: {len(acc)} vs {expected}'
 px,tv,actions=resources();results=[];selected=[];testrows=[];accuracyrows=[];daily=[];events=[];ledgers=[];annual=[]
 for sp in spec['splits']:
  for index in spec['indices']:
   start=pd.Timestamp('2018-07-01' if index=='KRX300' else '2018-01-01');end=pd.Timestamp(sp['train_end'],12,31)
   local=[]
   for model in spec['models']:
    for lead in spec['leads']:
     print(sp['id'],index,model,lead,'IS',flush=True)
     metrics,*_=simulate(index,model,lead,start,end,schedule,acc,targets,px,tv,actions)
     ac=accuracy(index,model,lead,start,end,schedule,acc)
     row=dict(split_id=sp['id'],index_id=index,model=model,lead=lead,**metrics,**{'prediction_'+k:v for k,v in ac.items()})
     results.append(row);local.append(row)
   z=pd.DataFrame(local);z=z[z.status.eq('valid')].sort_values(['sharpe','prediction_f1','lead','model'],ascending=[False,False,True,True])
   if z.empty:continue
   best=z.iloc[0];selected.append(best.to_dict());write('selected_IS_only',selected)
   # Fixed L60 carry control is declared before OOS and never selects a parameter.
   for role,model,lead in [('IS_selected',best.model,int(best.lead)),('fixed_L60_carry_control','baseline_carry',60)]:
    a=pd.Timestamp(sp['test_start'],1,1);b=pd.Timestamp('2026-06-30')
    metrics,d,ev,ledger=simulate(index,model,lead,a,b,schedule,acc,targets,px,tv,actions)
    ident=dict(split_id=sp['id'],role=role,index_id=index,model=model,lead=lead)
    testrows.append(dict(ident,**metrics));ac=accuracy(index,model,lead,a,b,schedule,acc);accuracyrows.append(dict(ident,**ac))
    if d is not None:
     daily.append(d.assign(**ident));events.append(ev.assign(**ident));ledgers.append(ledger.assign(**ident))
     previous=B
     for y,g in d.groupby(d.date.dt.year):
      annual.append(dict(ident,year=y,return_on_prior_equity=g.equity.iloc[-1]/previous-1,pnl_on_fixed_budget=(g.equity.iloc[-1]-previous)/B,position_days=int(g.any_position.sum()),year_end=g.date.iloc[-1],partial_year=y==2026));previous=g.equity.iloc[-1]
   write('IS_grid',results);write('OOS_selected_and_control',testrows);write('OOS_prediction_accuracy',accuracyrows);write('OOS_annual',annual)
 pd.concat(daily,ignore_index=True).to_parquet(OUT/'OOS_daily_accounts.parquet',index=False)
 pd.concat(events,ignore_index=True).to_csv(OUT/'OOS_event_returns.csv',index=False,encoding='utf-8-sig')
 pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'OOS_execution_ledger.parquet',index=False)
 print('EVALUATION COMPLETE',flush=True)
if __name__=='__main__':main()
