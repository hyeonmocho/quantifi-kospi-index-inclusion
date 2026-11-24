"""Validate Nov-20 FnGuide updates, replay H2 and update the submission package."""
from pathlib import Path
import sys, json, hashlib, csv, shutil
import numpy as np
import pandas as pd
import openpyxl

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'01_data/processed/kospi_pair_fnguide_update_20261122'
OUT=ROOT/'04_results/kospi_pair_nov20_update_20261122'
PKG=ROOT/'최종'
PRICE_PANELS=None

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare():
    DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    sources=[('float_ratio','유동주식비율_1120.csv','유동주식비율_1120.xlsx'),
             ('KOSPI100','코스피100_구성종목여부_1120.csv','코스피100_구성종목여부_1120.xlsx'),
             ('KOSPI200','코스피200_구성종목여부_1120.csv','코스피200_구성종목여부1120.xlsx')]
    audits=[]
    for key,cn,xn in sources:
        cf=ROOT/'01_data/processed/fnguide'/cn;xf=ROOT/'01_data/raw/fnguide'/xn
        wb=openpyxl.load_workbook(xf,read_only=True,data_only=True)
        rows=list(wb.worksheets[0].iter_rows(max_row=40,values_only=True));wb.close()
        pos=next(i for i,r in enumerate(rows) if r[0]=='코드')
        head=rows[pos:pos+6]
        assert head[0][0]=='코드' and head[1][0]=='코드명'
        codes=list(head[0][1:]);names=list(head[1][1:])
        with cf.open(encoding='utf-8-sig') as f:csv_names=next(csv.reader(f))[1:]
        assert names==csv_names and len(codes)==len(set(codes))
        blocks=[]
        for c in pd.read_csv(cf,header=None,skiprows=1,chunksize=300,dtype=str):
            dates=pd.to_datetime(c.iloc[:,0],errors='raise')
            a=c.iloc[:,1:].apply(lambda s:s.str.strip())
            if key=='float_ratio':
                a=a.apply(pd.to_numeric,errors='coerce')
                assert ((a>=0)&(a<=100)|a.isna()).all().all()
                a=(a/100).astype('float32')
            else:
                assert a.isin(['Y','N']).all().all()
                a=a.eq('Y')
            a.columns=codes;a.index=pd.DatetimeIndex(dates,name='date');blocks.append(a)
        panel=pd.concat(blocks)
        assert panel.index.is_unique and panel.index.is_monotonic_increasing
        assert panel.index.max()==pd.Timestamp('2026-11-20')
        panel.to_parquet(DATA/(key+'.parquet'))
        pd.DataFrame({'code':codes,'name':names}).to_csv(DATA/(key+'_names.csv'),index=False,encoding='utf-8-sig')
        audit=dict(field=key,file=str(cf.relative_to(ROOT)),xlsx=str(xf.relative_to(ROOT)),
            csv_sha256=digest(cf),xlsx_sha256=digest(xf),rows=len(panel),codes=len(codes),
            first=str(panel.index.min().date()),last=str(panel.index.max().date()),
            latest_valid=int(panel.iloc[-1].notna().sum()),latest_members=int(panel.iloc[-1].sum()) if key!='float_ratio' else None)
        if key=='float_ratio':
            old=pd.read_parquet(ROOT/'04_results/signal_a_rules_v2_20261107/inputs/float_ratio_zero_preserved.parquet')
            dates=old.index.intersection(panel.index);cols=old.columns.intersection(panel.columns)
            # Existing eligibility threshold is 10%; summarize both value and pass changes.
            x=old.loc[dates,cols];y=panel.loc[dates,cols]
            valid=x.notna()&y.notna()
            audit['overlap_numeric_differences_gt_1e_6']=int(((x-y).abs().gt(1e-6)&valid).sum().sum())
            audit['overlap_10pct_eligibility_changes']=int((x.ge(.1).ne(y.ge(.1))&valid).sum().sum())
            audit['old_missing_now_present']=int((x.isna()&y.notna()).sum().sum())
            audit['old_present_now_missing']=int((x.notna()&y.isna()).sum().sum())
            changes=(x.ge(.1).ne(y.ge(.1))&valid).stack()
            changes[changes].reset_index().to_csv(OUT/'float_eligibility_changed_dates.csv',index=False)
        else:
            official=pd.read_csv(ROOT/'01_data/processed/kospi_pair_nonfnguide_20261122/official_membership_snapshots.csv',dtype=str)
            q=official[official.index_id.eq(key)].copy();q['date']=pd.to_datetime(q.requested_date,format='%Y%m%d');q['code']='A'+q.isu_cd
            differences=[]
            for dt,g in q.groupby('date'):
                truth=set(g.code);v=set(panel.columns[panel.loc[dt]])
                differences.append(dict(date=str(dt.date()),official=len(truth),vendor=len(v),
                                        only_official=';'.join(sorted(truth-v)),only_vendor=';'.join(sorted(v-truth))))
            pd.DataFrame(differences).to_csv(DATA/(key+'_official_reconciliation.csv'),index=False)
            audit['official_snapshot_mismatches']=sum(bool(x['only_official'] or x['only_vendor']) for x in differences)
        audits.append(audit);print(json.dumps(audit,ensure_ascii=False),flush=True)
        (OUT/'input_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'input_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')

def replay():
    import kospi_pair_oct28_extension_20261122 as run
    run.OUT=OUT;run.END=pd.Timestamp('2026-11-20')
    run.FLOAT_OVERRIDE=DATA/'float_ratio.parquet'
    run.MEMBERS_OVERRIDE={idx:DATA/(idx+'.parquet') for idx in ['KOSPI200','KOSPI100']}
    run.PRICE_PANELS=PRICE_PANELS
    run.main();run.evaluate()

def full_accounts(cost_only=False, oos_start=None, reselect=False, historical_targets=None):
    """Replay every standalone entry-count and the shared account, frozen IS choices."""
    import kospi_pair_policy_20261121 as policy
    import kospi_pair_policy_summary_20261121 as summaries
    from compare_krx300_engine_swap import holding_period_sharpes
    from index_three_portfolio_20261117 import engine_selected_checkpoint as engine
    import corporate_actions
    corporate_actions.dump=lambda *args:None
    base=ROOT/'04_results/kospi_pair_policy_20261121'
    for p in base.iterdir():
        if p.is_file() and not (OUT/p.name).exists():shutil.copy2(p,OUT/p.name)
    if historical_targets is None and (base/'broker_comparison').exists():
        shutil.copytree(base/'broker_comparison',OUT/'broker_comparison',dirs_exist_ok=True)
    forecasts,targets=policy.windows.restore_early_signals(
        pd.read_csv(policy.SRC/'forecasts.csv',parse_dates=['entry_date','exit_date']),pd.read_csv(policy.SRC/'targets.csv'),policy.MODEL)
    forecasts=forecasts[forecasts.index_id.isin(policy.IDS)&forecasts.model.eq(policy.MODEL)]
    targets=targets[targets.case_id.isin(forecasts.case_id)&targets.model.eq(policy.MODEL)]
    if historical_targets is not None:
        assert historical_targets.case_id.isin(forecasts.case_id).all()
        targets=historical_targets.copy()
    resources=policy.replay.load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py','nov20_resources')
    resources.write=lambda *args:None
    px,tv,actions=policy.replay.supplement(*resources.resources())
    exits,evidence=policy.announcement_exit_map(forecasts,targets,px.index)
    evidence.to_csv(OUT/'announcement_decisions.csv',index=False)
    forecasts=pd.concat([forecasts,pd.read_csv(OUT/'forecasts.csv',parse_dates=['entry_date','exit_date'])],ignore_index=True)
    targets=pd.concat([targets,pd.read_csv(OUT/'targets.csv')],ignore_index=True)
    forecasts.to_csv(OUT/'all_forecasts.csv',index=False);targets.to_csv(OUT/'all_targets.csv',index=False)
    raw=pd.read_parquet(ROOT/'01_data/processed/kospi_pair_nonfnguide_20261122/KRX_raw_market_observations.parquet')
    raw['date']=pd.to_datetime(raw.BAS_DD,format='%Y%m%d');raw['code']='A'+raw.ISU_CD.astype(str).str.zfill(6)
    for col in ['TDD_CLSPRC','ACC_TRDVAL']:raw[col]=pd.to_numeric(raw[col].astype(str).str.replace(',',''),errors='raise')
    for field,old in [('TDD_CLSPRC',px),('ACC_TRDVAL',tv)]:
        tail=raw.pivot(index='date',columns='code',values=field).loc['2026-07-01':'2026-11-20']
        if PRICE_PANELS is not None:
            panel=PRICE_PANELS['close' if field=='TDD_CLSPRC' else 'turnover']
            tail=panel.reindex_like(tail).where(tail.notna())
        combined=pd.concat([old.loc[:'2026-06-30'],tail]).sort_index()
        if field=='TDD_CLSPRC':px=combined
        else:tv=combined
    dec=pd.read_csv(OUT/'H2_announcement_decisions.csv')
    for r in dec[dec.outcome.eq('confirmed_out')].itertuples():exits[(r.case_id,r.code)]=pd.Timestamp(r.first_sale)
    rates=pd.read_csv(ROOT/'01_data/processed/fnguide/CD_91.csv')
    rates=pd.Series(pd.to_numeric(rates.iloc[:,1]).to_numpy()/100,index=pd.to_datetime(rates.iloc[:,0])).sort_index()
    rates=rates.reindex(pd.date_range('1999-12-01','2026-11-20'),method='ffill')
    # Earliest evaluation is 2004; no use of rates before their first observation.
    engine.cd91_rates=lambda:rates
    engine.BUDGET=1e10
    comparisons=[];main=[];audits=[]
    def simulate(ids,phase,s,e,n,prefix=None):
        counts=n if isinstance(n,dict) else dict.fromkeys(ids,n)
        choices=pd.DataFrame({'index_id':ids,'model':[policy.MODEL]*len(ids)})
        m,d,l,o,h,a=engine.simulate(choices,forecasts,forecasts,targets,px,tv,actions,
            start=s,end=e,lead_map={idx:policy.LEADS[counts[idx]] for idx in ids},mark_open=True,early_exit_dates=exits)
        assert a['max_limit_excess']<=.001 and a['min_cash']>=-.001 and a['profit_matches_event_cashflows']
        m['review_events']=[l.event.nunique() if idx=='COMBINED' else l[l.index_id.eq(idx)].event.nunique() for idx in m.index_id]
        a.update(ids=ids,phase=phase,n_splits=n);audits.append(a)
        if prefix:
            for suffix,frame in [('daily',d),('lots',l),('orders',o),('holdings',h)]:frame.to_parquet(OUT/f'{prefix}_{suffix}.parquet',index=False)
            risks=[holding_period_sharpes(g,idx,phase,m.loc[m.index_id.eq(idx),'sharpe'].item()) for idx,g in d.groupby('index_id')]
            pd.DataFrame(risks).to_csv(OUT/f'{prefix}_sharpe.csv',index=False)
            # Closed events only: do not report an unfinished December review as a win/loss.
            if phase=='OOS':policy.event_results(o[o.event.ne('2026-12')],d).to_csv(OUT/f'{prefix}_events.csv',index=False)
            if oos_start is None and historical_targets is None:
                old=pd.read_parquet(base/f'{prefix}_daily.parquet')
                q=d[d.date.le(old.date.max())].sort_values(['index_id','date']).reset_index(drop=True)
                old=old.sort_values(['index_id','date']).reset_index(drop=True)
                np.testing.assert_allclose(q.equity,old.equity,rtol=1e-11,atol=.01)
        print('replayed',ids,phase,n,flush=True)
        return m
    if reselect:
        assert oos_start is not None
        is_end=(pd.Timestamp(oos_start)-pd.Timedelta(days=1)).date().isoformat()
        oos_end='2026-11-20'
        table=[]; picked={}
        for idx,start in [('KOSPI200','2004-01-01'),('KOSPI100','2006-01-01')]:
            train=[]
            for n in policy.LEADS:
                m=simulate([idx],'IS',start,is_end,n)
                train.append(m[m.index_id.eq(idx)].assign(phase='IS',n_splits=n))
            train=pd.concat(train)
            picked[idx]=int(train.sort_values(['sharpe','mdd','n_splits'],ascending=[False,False,True]).iloc[0].n_splits)
            print('IS selection',idx,picked[idx],flush=True)
            table.append(train)
            simulate([idx],'IS',start,is_end,picked[idx],f'{idx}_IS')
            for n in policy.LEADS:
                m=simulate([idx],'OOS',oos_start,oos_end,n,f'{idx}_OOS' if n==picked[idx] else None)
                table.append(m[m.index_id.eq(idx)].assign(phase='OOS',n_splits=n))
        for phase,s,e in [('IS','2006-01-01',is_end),('OOS',oos_start,oos_end)]:
            main.append(simulate(list(policy.IDS),phase,s,e,picked,phase).assign(phase=phase))
        pd.concat(table).to_csv(OUT/'standalone_splits.csv',index=False)
        pd.concat(main).to_csv(OUT/'metrics.csv',index=False)
        (OUT/'selected_splits.json').write_text(json.dumps(picked),encoding='utf-8')
        costs=[]
        try:
            for bp in [40,50,60]:
                engine.FEE=bp/20000
                for ids in [[idx] for idx in policy.IDS]+[list(policy.IDS)]:
                    m=simulate(ids,'OOS',oos_start,oos_end,picked)
                    key=ids[0] if len(ids)==1 else 'COMBINED'
                    costs.append(m[m.index_id.eq(key)].assign(roundtrip_bp=bp))
        finally:
            engine.FEE=.0025
        pd.concat(costs).to_csv(OUT/'cost_sensitivity_oos.csv',index=False)
        (OUT/'full_replay_audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        summaries.OUT=OUT
        summaries.monthly_profiles();summaries.snapshots();summaries.summary()
        return
    if oos_start is not None:
        results=[]
        for ids in [[idx] for idx in policy.IDS]+[list(policy.IDS)]:
            idx=ids[0] if len(ids)==1 else 'COMBINED'
            prefix=f'{idx}_OOS' if len(ids)==1 else 'OOS'
            m=simulate(ids,'OOS',oos_start,'2026-11-20',4,prefix)
            results.append(m[m.index_id.eq(idx)].assign(phase='OOS'))
        pd.concat(results).to_csv(OUT/'oos_start_sensitivity.csv',index=False)
        (OUT/'oos_start_audits.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
        return
    if cost_only:
        rows=[]
        try:
            for cost in [40,50,60]:
                engine.FEE=cost/2/10000
                for ids in [['KOSPI200'],['KOSPI100'],list(policy.IDS)]:
                    m=simulate(ids,'OOS','2023-01-01','2026-11-20',4)
                    idx='COMBINED' if len(ids)==2 else ids[0]
                    rows.append(m[m.index_id.eq(idx)].assign(roundtrip_bp=cost))
        finally:engine.FEE=.0025
        pd.concat(rows).to_csv(OUT/'cost_sensitivity_oos.csv',index=False)
        return
    for idx,start in [('KOSPI200','2004-01-01'),('KOSPI100','2006-01-01')]:
        for phase,s,e in [('IS',start,'2022-12-31'),('OOS','2023-01-01','2026-11-20')]:
            for n in policy.LEADS:
                m=simulate([idx],phase,s,e,n,f'{idx}_{phase}' if n==4 else None)
                comparisons.append(m[m.index_id.eq(idx)].assign(phase=phase,n_splits=n))
    for phase,s,e in [('IS','2006-01-01','2022-12-31'),('OOS','2023-01-01','2026-11-20')]:
        m=simulate(list(policy.IDS),phase,s,e,4,phase);main.append(m.assign(phase=phase))
    pd.concat(comparisons).to_csv(OUT/'standalone_splits.csv',index=False)
    pd.concat(main).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'full_replay_audits.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    summaries.OUT=OUT;summaries.monthly_profiles();summaries.snapshots();summaries.summary()
    print(pd.concat(main)[['index_id','phase','ending_equity_krw','sharpe','mdd']].to_string(index=False),flush=True)

def historical_audit():
    import kospi_pair_policy_20261121 as policy
    from backtest_vintage_engine_20261119 import FRAME_ROOT
    rule=policy.replay.E
    ff=pd.read_parquet(DATA/'float_ratio.parquet')
    members={idx:pd.read_parquet(DATA/(idx+'.parquet')) for idx in policy.IDS}
    fc,tr=policy.windows.restore_early_signals(pd.read_csv(policy.SRC/'forecasts.csv',parse_dates=['entry_date','exit_date']),pd.read_csv(policy.SRC/'targets.csv'),policy.MODEL)
    records=[];changes=[]
    for r in fc[fc.index_id.isin(policy.IDS)&fc.model.eq(policy.MODEL)].itertuples():
        path=FRAME_ROOT[r.index_id]/(r.case_id+'.pkl')
        if not path.exists():path=ROOT/'04_results/two_axis_selection_20261118/frames'/path.name
        if not path.exists():
            records.append(dict(case_id=r.case_id,status='frame_missing_frozen_forecast'));continue
        obj=pd.read_pickle(path);f=obj['frame'].copy();q=obj['panel'].set_index('code')
        info=pd.Timestamp(obj['panel'].information_date.iloc[0])
        ref=ff.index[ff.index<info][-1]
        fresh=ff.loc[ref].reindex(f.index)
        changed=f.float_ratio.ge(.1).ne(fresh.ge(.1))&fresh.notna()&f.float_ratio.notna()
        cur=members['KOSPI200'].loc[ref].reindex(f.index,fill_value=False)
        mc=cur.ne(f.cur)
        rec=dict(case_id=r.case_id,information_date=info,float_pass_changes=int(changed.sum()),
                 float_changed_codes=';'.join(f.index[changed]),parent_membership_changes=int(mc.sum()),status='audited')
        if changed.any():
            profile=rule.profile_for(r.event,asof=info)
            z=q.reindex(f.index)
            adj=(.25*np.log1p(z.momentum20.clip(lower=-.99))*z.remaining/40).fillna(0).clip(-.4,.4)
            cap=(z.past_sum+z.remaining*z.current_cap*np.exp(adj))/z.full_days
            cap=cap.where(np.isfinite(cap)&cap.gt(0)&z.eligibility.eq('pass'),f.mc).to_numpy()
            old_sel=rule.prepare_selector(f,profile).select(cap,f.tv.to_numpy())
            f.loc[changed,'float_ratio']=fresh[changed]
            # Revisions in this input vintage only make a formerly passing name fail.
            assert fresh[changed].lt(.1).all(), 'Eligibility restoration needs full prerequisite reconstruction'
            f.loc[changed,'eligibility_state']='fail';f.loc[changed,'eligibility_reason']='float_below_10pct'
            new_sel=rule.prepare_selector(f,profile).select(cap,f.tv.to_numpy())
            diff=set(f.index[old_sel])^set(f.index[new_sel])
            rec['selected_parent_changes']=';'.join(sorted(diff))
            changes.append(rec.copy())
        records.append(rec)
    pd.DataFrame(records).to_csv(OUT/'historical_input_signal_audit.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'historical_float_selection_changes.csv',index=False)
    print(pd.DataFrame(changes).to_string(index=False),flush=True)
    print('membership discrepancies',sum(r.get('parent_membership_changes',0) for r in records),flush=True)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--replay' in sys.argv:replay()
    elif '--accounts' in sys.argv:full_accounts()
    elif '--costs' in sys.argv:full_accounts(cost_only=True)
    elif '--history-audit' in sys.argv:historical_audit()
    else:raise SystemExit('Use --prepare or --replay')
