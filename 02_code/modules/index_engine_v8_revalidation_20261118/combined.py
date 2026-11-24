"""Combine IS-selected sleeves, separately funded with a shared security/day cap."""
from research import load,ROOT,OUT,save,dump
from corporate_actions import supplement
import pandas as pd,numpy as np
from scipy import stats
import statsmodels.api as sm

def main():
 choices=pd.read_csv(OUT/'performance_choices.csv');assert choices.status.eq('selected').all()
 f=pd.read_csv(OUT/'forecasts.csv',parse_dates=['information_date','entry_date','exit_date'])
 t=pd.read_csv(OUT/'targets.csv');parts=[]
 for r in choices.itertuples():parts.append(f[f.phase.eq('OOS')&f.index_id.eq(r.index_id)&f.model.eq(r.model)&f.lead.eq(r.lead)])
 a=pd.concat(parts,ignore_index=True);schedule=a[['case_id','index_id','event','lead','entry_date','exit_date','information_date']].copy()
 universe=sorted(f[f.phase.eq('OOS')].event.unique())
 coverage=pd.DataFrame([dict(event=e,**{idx:bool(((a.index_id==idx)&(a.event==e)).any()) for idx in choices.index_id}) for e in universe]);coverage['all_available']=coverage[list(choices.index_id)].all(axis=1)
 save('integration_coverage',coverage)
 assert coverage.all_available.all(),'Selected sleeves have missing OOS forecasts; do not silently count missing as no signal.'
 a['forecast_available']=True
 rs=load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','v8_joint_resources');rs.write=lambda *a:None
 px,tv,actions=rs.resources();px,tv,actions=supplement(px,tv,actions)
 engine=load('02_code/modules/index_three_portfolio_20261117/engine.py','v8_joint_engine')
 perf,d,g,l,h,ck=engine.simulate(choices,schedule,a,t,px,tv,actions,shared=True,start='2023-01-02',end='2026-06-30')
 save('integration_performance',perf);save('integration_events',g)
 for n,frame in [('integration_daily',d),('integration_ledger',l),('integration_holdings',h)]:frame.to_parquet(OUT/(n+'.parquet'),index=False)
 used=l[l.side.isin(['buy','sell'])].groupby(['date','code']).agg(notional=('notional','sum'),limit=('limit','first'));assert (used.notional<=used.limit+.01).all()
 ck['checked_shared_order_ledger']=True
 dump('integration_checks',ck)
 events=g.groupby('event').pnl.sum()/9e9;x=events.to_numpy();year=events.index.str[:4];k=year.nunique();s=pd.Series(x-x.mean()).groupby(year.to_numpy()).sum();se=np.sqrt(k/(k-1)*(s*s).sum()/len(x)**2);ct=x.mean()/se
 save('integration_event_inference',[dict(n_events=len(x),year_clusters=k,mean_event_net=x.mean(),win_rate=(x>0).mean(),payoff=x[x>0].mean()/abs(x[x<0].mean()) if (x<0).any() else np.nan,year_cluster_t=ct,year_cluster_p=2*stats.t.sf(abs(ct),k-1))])
 q=d[d.index_id.eq('COMBINED')].set_index('date').return_daily
 month=(1+q).resample('ME').prod()-1
 rows=[]
 for lag in [1,3,6]:
  fit=sm.OLS(month.to_numpy(),np.ones((len(month),1))).fit(cov_type='HAC',cov_kwds={'maxlags':lag},use_t=True)
  rows.append(dict(months=len(month),HAC_lags=lag,mean_monthly=month.mean(),t=fit.tvalues[0],p=fit.pvalues[0],win_rate=(month>0).mean()))
 save('integration_monthly_inference',rows)
 annual=[]
 for idx,z in d.groupby('index_id'):
  opening=9e9 if idx=='COMBINED' else 3e9
  for year,v in z.sort_values('date').groupby(z.sort_values('date').date.dt.year):
   annual.append(dict(index_id=idx,year=year,opening_equity=opening,ending_equity=v.equity.iloc[-1],pnl=v.equity.iloc[-1]-opening,return_year=v.equity.iloc[-1]/opening-1));opening=v.equity.iloc[-1]
 save('integration_annual',annual)
 print(perf.to_string(index=False),flush=True)
if __name__=='__main__':main()
