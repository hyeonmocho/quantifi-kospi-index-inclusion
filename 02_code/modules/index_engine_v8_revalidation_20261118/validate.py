"""Independent result checks, uncertainty diagnostics, executed research notebook."""
from context import ROOT,OUT,dump
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,sys,os
from scipy import stats
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

def validate():
    p=pd.read_parquet(OUT/'panel_enriched.parquet');e=pd.read_parquet(OUT/'financial_evidence.parquet');o=e[e.original_priority.ge(1)]
    fits=pd.read_csv(OUT/'fit_audit.csv',parse_dates=['max_label_date','information_date'])
    f=pd.read_csv(OUT/'forecasts.csv',parse_dates=['information_date','entry_date']);t=pd.read_csv(OUT/'targets.csv')
    protocol=json.loads((OUT/'model_protocol.json').read_text(encoding='utf8'))
    # Align displayed IS summaries to the exact common events used in selection.
    a=pd.read_csv(OUT/'accuracy_summary.csv');common=json.loads((OUT/'common_IS_events.json').read_text(encoding='utf8'))
    for j,r in a[a.phase.eq('IS_forward')].iterrows():
        z=f[f.index_id.eq(r.index_id)&f.phase.eq(r.phase)&f.model.eq(r.model)&f.lead.eq(r.lead)&f.event.isin(common[r.index_id])]
        v=z.groupby('event')[['hits','predicted','actual']].mean().sum()
        for c,val in dict(n_events=z.event.nunique(),hits=v.hits,predicted=v.predicted,actual=v.actual,precision=v.hits/v.predicted if v.predicted else np.nan,recall=v.hits/v.actual if v.actual else np.nan,f1=2*v.hits/(v.predicted+v.actual) if v.predicted+v.actual else np.nan).items():a.loc[j,c]=val
    a.loc[a.phase.eq('BROKER_OOS'),'selected']=False
    a.to_csv(OUT/'accuracy_summary.csv',index=False,encoding='utf-8-sig')
    checks=[]
    def check(n,b):checks.append(dict(check=n,passed=bool(b)))
    check('original_OFS_requires_explicit_no_CFS',o.loc[o.fs_div.eq('OFS'),'original_no_CFS_statement'].eq(True).all())
    check('non_December_period_end_preserved',o.loc[o.fiscal_month.ne(12),'period_end'].dt.month.eq(o.loc[o.fiscal_month.ne(12),'fiscal_month']).all())
    check('no_future_financial_receipt_in_candidates',(p.financial_known_from.isna()|p.financial_known_from.le(p.information_date)).all())
    check('training_labels_strictly_older',(fits.max_label_date<fits.information_date).all())
    check('all_training_labels_IS_only',fits.max_label_date.dt.year.le(2022).all())
    check('no_outcome_or_future_features',all(not any(w in c for w in ['actual','label','future','target_log']) for c in protocol['features']))
    check('next_session_or_later_entry',(f.information_date<f.entry_date).all())
    check('prediction_counts_valid',(f.hits.le(f.predicted)&f.hits.le(f.actual)).all())
    weights=t.groupby(['case_id','model']).weight.sum();check('long_only_weights_sum_one',t.weight.ge(0).all()&np.isclose(weights,1).all())
    core=t[~t.case_id.str.startswith('broker_')].merge(p[['case_id','code','monitor_candidate']],on=['case_id','code'],how='left',validate='many_to_one')
    check('all_V8_targets_pass_updated_candidate_gate',core.monitor_candidate.eq(True).all())
    manifest=json.loads((OUT/'pre_run_freeze.json').read_text(encoding='utf8'))
    check('frozen_inputs_and_model_code_unchanged',all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest['hashes'].items()))
    d=pd.read_parquet(OUT/'daily_accounts.parquet');check('nonnegative_cash',d.cash.ge(-1e-4).all())
    l=pd.read_parquet(OUT/'execution_ledger.parquet');trade=l[l.side.isin(['buy','sell'])]
    used=trade.groupby(['index_id','model','lead','phase','date','code']).agg(notional=('notional','sum'),limit=('limit','first'))
    check('shared_ADV_limit_within_strategy',(used.notional<=used.limit+1e-3).all())
    check('fees_25bp_each_side',np.allclose(trade.fee,trade.notional*.0025))
    actions=json.loads((OUT/'corporate_supplement_actions.json').read_text(encoding='utf8'))
    check('no_old_security_trade_after_verified_halt',all(trade[trade.code.eq(x['old'])&trade.date.ge(pd.Timestamp(x['halt']))].empty for x in actions))
    accounts=pd.read_csv(OUT/'execution_all.csv');ev=pd.read_parquet(OUT/'execution_events.parquet')
    reconcile=[]
    for r in accounts[accounts.status.eq('valid')].itertuples():
        z=ev[ev.index_id.eq(r.index_id)&ev.model.eq(r.model)&ev.lead.eq(r.lead)&ev.phase.eq(r.phase)]
        reconcile.append(np.isclose(z.net_return.sum(),r.total_return,atol=1e-8))
    check('closed_event_PNL_reconciles_to_account_NAV',all(reconcile))
    check('no_live_certification',~p.production_certified.any())
    # Calendar guard can be checked without using actual index inclusion labels.
    from financial_parse import financial_asof
    synthetic=pd.DataFrame([dict(code='TEST',period_end=pd.Timestamp('2025-03-31'),known_from=pd.Timestamp('2025-06-13'),source_receipt='TEST',original_priority=1,evidence_usable=True,impaired=False)])
    before,_=financial_asof(['TEST'],pd.Timestamp('2025-06-12'),pd.Timestamp('2025-10-31'),synthetic)
    after,_=financial_asof(['TEST'],pd.Timestamp('2025-06-13'),pd.Timestamp('2025-10-31'),synthetic)
    check('future_filing_cannot_repair_earlier_forecast',before.iloc[0]=='unknown_no_receipt_evidence' and after.iloc[0]=='observed_no_impairment')
    dump('checks',checks);assert all(c['passed'] for c in checks),[c for c in checks if not c['passed']]
    # Paired review bootstrap of prediction differences, including zero-prediction events.
    rng=np.random.default_rng(18112026);choices=pd.read_csv(OUT/'frozen_choices.csv');ci=[]
    for r in choices.itertuples():
        z=f[f.index_id.eq(r.index_id)&f.phase.eq('OOS')&f.lead.eq(r.lead)]
        a=z[z.model.eq(r.model)].set_index('event');b=z[z.model.eq('baseline')].set_index('event');events=sorted(set(a.index)&set(b.index));n=len(events)
        sample=rng.integers(0,n,size=(10000,n))
        vals=[]
        for q in [a,b]:
            v=q.loc[events,['hits','predicted','actual']].to_numpy();s=v[sample].sum(axis=1);vals.append(2*s[:,0]/np.maximum(s[:,1]+s[:,2],1))
        delta=vals[0]-vals[1];lo,hi=np.quantile(delta,[.025,.975]);ci.append(dict(index_id=r.index_id,lead=r.lead,model=r.model,n_review_clusters=n,F1_difference_ci_low=lo,F1_difference_ci_high=hi,method='paired review bootstrap; exploratory reused OOS'))
    pd.DataFrame(ci).to_csv(OUT/'prediction_paired_uncertainty.csv',index=False,encoding='utf-8-sig')
    availability=[]
    for r in choices.itertuples():
        all_actual=f[f.index_id.eq(r.index_id)&f.phase.eq('OOS')].groupby('event').actual.first()
        selected=f[f.index_id.eq(r.index_id)&f.phase.eq('OOS')&f.model.eq(r.model)&f.lead.eq(r.lead)]
        hits=selected.hits.sum();predicted=selected.predicted.sum();actual=all_actual.sum()
        availability.append(dict(index_id=r.index_id,model=r.model,lead=r.lead,total_reviews=len(all_actual),available_reviews=selected.event.nunique(),unavailable_reviews=';'.join(sorted(set(all_actual.index)-set(selected.event))),precision=hits/predicted if predicted else np.nan,recall_including_unavailable=hits/actual,f1_including_unavailable=2*hits/(predicted+actual)))
    pd.DataFrame(availability).to_csv(OUT/'forecast_availability.csv',index=False,encoding='utf-8-sig')
    # Selection sensitivity, not a fresh set of independently validated strategies.
    sensitivity=f[f.phase.eq('OOS')].groupby(['index_id','model','lead'])[['hits','predicted','actual']].sum().reset_index()
    sensitivity['precision']=sensitivity.hits/sensitivity.predicted.replace(0,np.nan);sensitivity['recall']=sensitivity.hits/sensitivity.actual;sensitivity['f1']=2*sensitivity.hits/(sensitivity.predicted+sensitivity.actual)
    sensitivity.to_csv(OUT/'OOS_sensitivity_exploratory.csv',index=False,encoding='utf-8-sig')
    stages=[dict(priority=1,status='implemented_with_remaining_document_gaps',result='original XML, no-CFS declaration, fiscal calendars, dated gate implemented'),dict(priority=2,status='partial_external_data_gap',result='2025 five TMI anchors linked; full historical BMI and 2026 TMI still unavailable'),dict(priority=3,status='partial_rule_certification_gap',result='dated rules audited; early K100 and issuer-specific merger review decisions incomplete'),dict(priority=4,status='implemented_research_overlay_not_full_PIT_certification',result='listing records, dated sectors/float and parent-anchor flags linked; proxy labels preserved'),dict(priority=5,status='research_validation_executed',result='IS forward-only selection, frozen models, reused historical OOS evaluated'),dict(priority=6,status='execution_and_broker_comparison_executed',result='30eok per index, common costs and fills; full broker OOS only KOSPI200')]
    dump('six_priority_status',stages)
    dump('artifact_manifest',[dict(path=str(x.relative_to(ROOT)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in list(Path(__file__).parent.glob('*.py'))+[OUT/'financial_evidence.parquet',OUT/'panel_enriched.parquet',OUT/'frozen_choices.csv',OUT/'performance_choices.csv']])

if __name__=="__main__":validate()
