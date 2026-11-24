"""Fixed-principal long-only sleeves sharing a security/day participation cap.

Model choices and target weights are read from frozen IS selections. No fitting.
Observed prices/corporate actions follow the existing audited single-sleeve model.
"""
import numpy as np
import pandas as pd

BUDGET = 3e9
FEE = .0025
PARTICIPATION = .1


def metrics(d, budget):
    d = d.sort_values('date')
    nav = d.equity / budget
    ret = d.return_daily
    years = (d.date.iloc[-1] - d.date.iloc[0]).days / 365.25
    sd = ret.std(ddof=1)
    return dict(period_start=str(d.date.iloc[0].date()), period_end=str(d.date.iloc[-1].date()),
                budget_krw=budget, ending_equity_krw=d.equity.iloc[-1],
                profit_krw=d.equity.iloc[-1]-budget, total_return=nav.iloc[-1]-1,
                cagr=nav.iloc[-1]**(1/years)-1, volatility=sd*np.sqrt(252),
                sharpe=ret.mean()/sd*np.sqrt(252) if sd > 0 else np.nan,
                mdd=(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1).min(),
                mean_exposure=d.held_value.mean()/budget,
                position_days=int(d.any_position.sum()), n_sessions=len(d))


def simulate(choices, schedule, accuracy, targets, px, tv, actions,
             shared=True, weight_cap=None, start='2023-01-01', end='2026-06-30', leads=(80,60,40,20)):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    accounts = choices.index_id.tolist()
    ai = {name: i for i, name in enumerate(accounts)}
    ns = len(accounts)
    cases, selected = [], []
    for r in choices.itertuples():
        ss = schedule[(schedule.index_id == r.index_id) & (schedule.lead.isin(leads)) &
                      schedule.entry_date.between(start, end)].copy()
        if r.index_id == 'KRX300':
            ss = ss[ss.event >= '2018-12']
        aa = accuracy[(accuracy.index_id == r.index_id) & (accuracy.model == r.model) &
                      accuracy.case_id.isin(ss.case_id)]
        assert len(ss) == len(aa) and aa.forecast_available.all(), r
        ss['model'] = r.model
        cases.append(ss)
        tr = targets[(targets.model == r.model) & targets.case_id.isin(ss.case_id)].copy()
        assert not tr.duplicated(['case_id', 'code']).any()
        if weight_cap is not None:
            tr['weight'] = tr.weight.clip(upper=weight_cap)
        selected.append(tr)
    ss, tr = pd.concat(cases), pd.concat(selected)
    codes = sorted(set(tr.code) | {a['new'] for a in actions if a['new']})
    ci = {c: i for i, c in enumerate(codes)}
    nc = len(codes)
    P, T = px.reindex(columns=codes), tv.reindex(columns=codes)
    A = T.rolling(20, min_periods=20).mean().shift(1)
    days = px.index[(px.index >= start) & (px.index <= end)]
    startmap = {date: g for date, g in ss.groupby('entry_date')}
    live = [a for a in actions if a['old'] in ci]
    cash = np.full(ns, BUDGET)
    prev = cash.copy()
    lots, events, daily, ledger, holdings = [], [], [], [], []
    last = P.loc[P.index < start].ffill().iloc[-1].to_numpy()
    min_cash = BUDGET
    max_limit_excess = 0.
    target_lookup = {cid: g.set_index('code').weight for cid, g in tr.groupby('case_id')}

    def positions():
        out = np.zeros((ns, nc))
        for z in lots:
            out[z['account']] += z['positions']
        return out

    for date in days:
        prices, turn, adv = P.loc[date].to_numpy().copy(), T.loc[date].to_numpy(), A.loc[date].to_numpy()
        for a in live:
            j = ci[a['old']]
            if a['halt'] <= date < a['effective']:
                prices[j] = last[j]
            if a['effective'] <= date < a['delivery']:
                prices[j] = prices[ci[a['new']]]*a['conversion'] if a['new'] else a['cash_per_adjusted_unit']
            if date == a['delivery']:
                for z in lots:
                    amount = z['positions'][j]
                    if amount <= 0:
                        continue
                    delivered, fee = 0., 0.
                    if a['new']:
                        z['positions'][ci[a['new']]] += amount*a['conversion']
                    else:
                        delivered = amount*a['cash_per_adjusted_unit']
                        fee = delivered*a.get('cash_fee_rate', 0.)
                        cash[z['account']] += delivered-fee
                        z['net_cashflow'] += delivered-fee
                        z['fees'] += fee
                    z['positions'][j] = 0.
                    ledger.append(dict(date=date, index_id=z['index_id'], event=z['event'], code=a['old'],
                                       side='corporate_exchange_or_cash_delivery', notional=np.nan,
                                       fee=fee, cash_delivery=delivered, limit=np.nan))
        pos = positions()
        held = pos.sum(axis=0) > 1e-10
        if (held & (~np.isfinite(prices) | (prices <= 0))).any():
            raise ValueError(f'Unobservable held price {date}: {np.array(codes)[held & ~np.isfinite(prices)]}')
        for r in startmap.get(date, pd.DataFrame()).itertuples():
            k = ai[r.index_id]
            held_value = np.nansum(pos[k]*prices)
            budget = min(max(cash[k], 0.), max(BUDGET-held_value, 0.), BUDGET/len(leads))
            z = dict(account=k, index_id=r.index_id, model=r.model, lead=r.lead,
                     case_id=r.case_id, event=r.event, exit_date=r.exit_date,
                     deadline=px.index[min(px.index.get_loc(date)+9, len(px)-1)],
                     positions=np.zeros(nc), remaining=np.zeros(nc), allocation=budget,
                     entry_notional=0., fees=0., net_cashflow=0.)
            w = target_lookup.get(r.case_id, pd.Series(dtype=float))
            assert np.isfinite(w).all() and (w >= 0).all() and w.sum() <= 1+1e-9
            for code, weight in w.items():
                z['remaining'][ci[code]] = budget/(1+FEE)*weight
            z['initial_target_notional'] = z['remaining'].sum()
            lots.append(z)
        valid = np.isfinite(prices) & (prices > 0) & np.isfinite(adv) & (adv > 0) & np.isfinite(turn) & (turn > 0)
        limit = np.where(valid, np.minimum(PARTICIPATION*adv, PARTICIPATION*turn), 0.)
        used = np.zeros((1 if shared else ns, nc))
        # Aggregate sell+buy participation, sell first. No internal crossing or free netting.
        for side in ['sell', 'buy']:
            active = [z for z in lots if (date >= z['exit_date'] if side == 'sell'
                      else date < z['exit_date'] and date <= z['deadline'])]
            if not active:
                continue
            demand = np.stack([np.nan_to_num(z['positions']*prices, nan=0.) if side == 'sell'
                               else z['remaining'] for z in active])
            allocation = np.zeros_like(demand)
            pools = np.array([0 if shared else z['account'] for z in active])
            for pool in np.unique(pools):
                ix = np.flatnonzero(pools == pool)
                summed = demand[ix].sum(axis=0)
                fraction = np.minimum(1., np.divide(np.maximum(0., limit-used[pool]), summed,
                                                    out=np.zeros(nc), where=summed > 0))
                allocation[ix] = demand[ix]*fraction
            if side == 'buy':
                for k in range(ns):
                    ix = [j for j, z in enumerate(active) if z['account'] == k]
                    if not ix:
                        continue
                    need = allocation[ix].sum()*(1+FEE)
                    if need > max(cash[k], 0.):
                        allocation[ix] *= max(cash[k], 0.)/need
            for j, z in enumerate(active):
                k = z['account']
                notion = allocation[j]
                used[0 if shared else k] += notion
                units = notion/np.where(valid, prices, 1.)
                fee = notion*FEE
                z['fees'] += fee.sum()
                if side == 'buy':
                    z['positions'] += units
                    cash[k] -= (notion+fee).sum()
                    z['net_cashflow'] -= (notion+fee).sum()
                    z['remaining'] = np.maximum(0., z['remaining']-notion)
                    z['entry_notional'] += notion.sum()
                else:
                    z['positions'] = np.maximum(0., z['positions']-units)
                    cash[k] += (notion-fee).sum()
                    z['net_cashflow'] += (notion-fee).sum()
                for c in np.flatnonzero(notion > 1e-5):
                    ledger.append(dict(date=date, index_id=z['index_id'], event=z['event'],
                                       code=codes[c], side=side, notional=notion[c],
                                       fee=fee[c], limit=limit[c]))
            min_cash = min(min_cash, cash.min())
            max_limit_excess = max(max_limit_excess, (used-limit).max())
            assert cash.min() >= -.001 and (used <= limit+.001).all()
            cash[(cash < 0) & (cash > -.001)] = 0.
        keep = []
        for z in lots:
            if date >= z['exit_date'] and z['positions'].sum() < 1e-8:
                events.append(dict(index_id=z['index_id'], model=z['model'], lead=z['lead'],
                                   event=z['event'], pnl=z['net_cashflow'], net_return=z['net_cashflow']/BUDGET,
                                   allocation=z['allocation'], entry_notional=z['entry_notional'],
                                   initial_target_notional=z['initial_target_notional'], fees=z['fees'],
                                   actual_exit=date, planned_exit=z['exit_date'],
                                   exit_delay_sessions=px.index.get_loc(date)-px.index.get_loc(z['exit_date'])))
            else:
                keep.append(z)
        lots = keep
        pos = positions()
        values = np.nan_to_num(pos*prices, nan=0.)
        held_value = values.sum(axis=1)
        equity = cash+held_value
        for k, name in enumerate(accounts):
            daily.append(dict(date=date, index_id=name, equity=equity[k], return_daily=equity[k]/prev[k]-1,
                              held_value=held_value[k], cash=cash[k], any_position=pos[k].sum() > 1e-8))
            for j in np.flatnonzero(pos[k] > 1e-10):
                holdings.append(dict(date=date, index_id=name, code=codes[j], market_value=values[k,j]))
        daily.append(dict(date=date, index_id='COMBINED', equity=equity.sum(),
                          return_daily=equity.sum()/prev.sum()-1, held_value=held_value.sum(),
                          cash=cash.sum(), any_position=pos.sum() > 1e-8))
        prev = equity.copy()
        last = np.where(np.isfinite(prices) & (prices > 0), prices, last)
    assert not lots and len(events) == len(ss), 'Do not silently discard open or unavailable events'
    d, g, l, h = pd.DataFrame(daily), pd.DataFrame(events), pd.DataFrame(ledger), pd.DataFrame(holdings)
    summaries = []
    for name in accounts+['COMBINED']:
        q = d[d.index_id.eq(name)]
        eg = g if name == 'COMBINED' else g[g.index_id.eq(name)]
        m = metrics(q, BUDGET*ns if name == 'COMBINED' else BUDGET)
        m.update(index_id=name, strategy_events=len(eg), total_fees=eg.fees.sum(),
                 initial_fill_ratio=eg.entry_notional.sum()/eg.initial_target_notional.sum(),
                 max_exit_delay_sessions=int(eg.exit_delay_sessions.max()))
        assert abs(m['profit_krw']-eg.pnl.sum()) < .02
        summaries.append(m)
    checks = dict(min_cash=min_cash, max_limit_excess=max_limit_excess, all_events_closed=True,
                  separate_cash_accounts=True, shared_security_day_cap=shared,
                  profit_matches_event_cashflows=True, n_event_lots=len(g))
    return pd.DataFrame(summaries), d, g, l, h, checks
