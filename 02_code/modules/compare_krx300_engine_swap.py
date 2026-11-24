"""Isolate the effect of replacing only the KRX300 engine in the locked 4/4/1 account."""
import json

import numpy as np
import pandas as pd

import v18_long_replay as replay
from index_three_portfolio_20261117 import engine_selected_checkpoint as engine


def holding_period_sharpes(daily, index_id, phase, reference):
    d = daily.sort_values('date')
    r = d.return_daily.to_numpy()
    rf = engine.cd91_intervals(d.date)
    # These accounts start in cash and orders execute at the close. Last
    # night's exposure determines the interest benchmark through today's close.
    held = d.any_position.shift(1, fill_value=False).to_numpy()
    weight = (d.held_value / d.equity).shift(1, fill_value=0).to_numpy()
    execution_or_holding = held | d.any_position.to_numpy()
    np.testing.assert_allclose(r[~execution_or_holding], 0, atol=1e-12)
    assert np.all((weight >= 0) & (weight <= 1+1e-10))
    def sharpe(x):
        return x.mean()/x.std(ddof=1)*np.sqrt(252)
    full_cd = sharpe(r-rf)
    np.testing.assert_allclose(full_cd, reference, atol=1e-10)
    x = r - rf*held
    return dict(index_id=index_id, phase=phase, start=d.date.iloc[0], end=d.date.iloc[-1],
                sessions=len(r), exposed_intervals=int(held.sum()), cash_intervals=int((~held).sum()),
                active_sessions=int(execution_or_holding.sum()),
                full_cd=full_cd, holding_period_cd=sharpe(x), delta=sharpe(x)-full_cd,
                exposure_weighted_cd=sharpe(r-rf*weight), active_sample_sharpe=sharpe(x[execution_or_holding]),
                zero_rf=sharpe(r))


def main():
    trend = replay.R / 'common_index_engine_20261119'
    selected = trend / 'long_IS_compromise'
    scenario = replay.R / 'v18_long_vintage_replay'
    dest = selected / 'krx300_engine_comparison'
    dest.mkdir(exist_ok=True)
    rows = []
    risk = []
    for label, folder in [('trend', selected), ('scenario', scenario)]:
        table = pd.read_csv(folder / 'performance.csv')
        daily = pd.read_parquet(folder / 'daily.parquet')
        for phase in ('IS', 'OOS'):
            query = "index_id=='KRX300' and n_splits==1 and phase==@phase"
            row = table.query(query).iloc[0].to_dict()
            row.update(engine.metrics(daily.query(query), 1e10))
            rows.append(dict(row, variant=label))
            if label=='scenario':
                risk.append(holding_period_sharpes(daily.query(query),'KRX300',phase,row['sharpe']))
        if label=='trend':
            for idx in ['KOSPI200','KOSPI100']:
                for phase in ('IS','OOS'):
                    query="index_id==@idx and phase==@phase and n_splits==4"
                    risk.append(holding_period_sharpes(daily.query(query),idx,phase,table.query(query).sharpe.item()))

    base = pd.read_csv(trend / 'forecasts.csv', parse_dates=['entry_date', 'exit_date'])
    alt = pd.read_csv(scenario / 'forecasts.csv', parse_dates=['entry_date', 'exit_date'])
    keys = ['case_id','lead','entry_date','exit_date']
    krx = "index_id=='KRX300' and event >= '2018-07' and event <= '2026-06'"
    pd.testing.assert_frame_equal(base.query(krx)[keys].sort_values('case_id').reset_index(drop=True),
                                  alt.query(krx)[keys].sort_values('case_id').reset_index(drop=True))
    schedules = pd.concat([base[base.index_id.ne('KRX300')], alt[alt.index_id.eq('KRX300')]])
    schedules = schedules[schedules.event.between('2018-07','2026-06')]
    baseline_targets = pd.read_csv(trend/'targets.csv')
    alt_targets = pd.read_csv(scenario/'targets.csv')
    targets = pd.concat([baseline_targets[baseline_targets.case_id.isin(base.loc[base.index_id.ne('KRX300'),'case_id'])],
                         alt_targets[alt_targets.case_id.isin(alt.loc[alt.index_id.eq('KRX300'),'case_id'])]])
    sleeves = schedules[['index_id','model']].drop_duplicates()
    resources = replay.load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py', 'krx_swap_inputs')
    resources.write = lambda *args: None
    prices, turnover, actions = replay.supplement(*resources.resources())
    counts = json.loads((selected/'selection.json').read_text())
    engine.BUDGET = 1e10
    mapping = {idx: replay.LEADS[:n] for idx, n in counts.items()}
    mixed = []
    for phase, start, end, daily_file in [('IS','2018-07-01','2022-12-31','portfolio_IS_daily.parquet'),
                                         ('OOS','2023-01-01','2026-06-30','portfolio_daily.parquet')]:
        schedule = schedules[schedules.event.between(start[:7],end[:7])]
        metrics, path, lots, orders, holdings, checks = engine.simulate(
            sleeves, schedule, schedules, targets, prices, turnover, actions, start=start, end=end,
            lead_map=mapping, mark_open=phase=='IS')
        assert checks['profit_matches_event_cashflows'] and (phase=='IS' or checks['all_events_closed'])
        used = orders[orders.side.isin(['buy','sell'])].groupby(['date','code']).agg(
            gross=('notional','sum'), cap=('limit','min'))
        assert (used.gross <= used.cap + .01).all() and path.cash.min() >= -.01
        hist = pd.read_parquet(selected/daily_file).query("index_id=='COMBINED'")
        joint = path[path.index_id.eq('COMBINED')]
        risk.append(holding_period_sharpes(joint,'COMBINED',phase,metrics.query("index_id=='COMBINED'").sharpe.item()))
        pd.testing.assert_index_equal(pd.Index(joint.date), pd.Index(hist.date))
        equity_sum = path[path.index_id.ne('COMBINED')].groupby('date').equity.sum()
        np.testing.assert_allclose(equity_sum, joint.equity, atol=.02, rtol=0)
        for label, frame in [('trend',hist),('mixed',joint)]:
            mixed.append(dict(phase=phase,variant=label,reviews=lots.event.nunique(),
                              end_stock_value=frame.held_value.iloc[-1],**engine.metrics(frame,3e10)))
        if phase=='IS':
            # Cross-check the reference series against a replay with the same schedules.
            trend_schedule=base[base.event.between(start[:7],end[:7])]
            _, reference, _, _, _, _=engine.simulate(
                trend_schedule[['index_id','model']].drop_duplicates(), trend_schedule, base, baseline_targets,
                prices,turnover,actions,start=start,end=end,lead_map=mapping,mark_open=True)
            np.testing.assert_allclose(reference.loc[reference.index_id.eq('COMBINED'),'equity'],hist.equity,
                                       atol=.02,rtol=0)
        else:
            for label,frame in [('orders',orders),('holdings',holdings),('daily',path),('tranches',lots)]:
                frame.to_parquet(dest/f'mixed_OOS_{label}.parquet',index=False)
    pd.DataFrame(mixed).to_csv(dest/'combined_IS_OOS.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(risk).to_csv(dest/'holding_period_sharpes.csv',index=False,encoding='utf-8-sig')
    for label, table in [('trend', pd.read_csv(selected/'portfolio.csv')), ('scenario', metrics)]:
        rows.append(dict(table.query("index_id=='COMBINED'").iloc[0].to_dict(), phase='OOS', variant=label))
    result = pd.DataFrame(rows)
    result.to_csv(dest/'performance.csv', index=False, encoding='utf-8-sig')
    old = pd.read_parquet(selected/'portfolio_tranches.parquet')
    pnl = pd.concat([t.groupby(['event','index_id']).pnl.sum().rename(label)
                     for label,t in [('trend',old),('scenario',lots)]],axis=1)
    pnl['change'] = pnl.scenario-pnl.trend
    np.testing.assert_allclose(pnl.change.sum(), metrics.query("index_id=='COMBINED'").profit_krw.item()
                               -rows[-2]['profit_krw'], rtol=0, atol=.02)
    pnl.to_csv(dest/'event_pnl.csv', encoding='utf-8-sig')
    print(result[['index_id','phase','variant','total_return','cagr','sharpe','sharpe_zero_rf','volatility','mdd','mean_names','initial_fill_ratio']].round(5).to_string(index=False))
    print((pnl.xs('KRX300',level='index_id')/1e8).round(2))


if __name__ == '__main__':
    main()
