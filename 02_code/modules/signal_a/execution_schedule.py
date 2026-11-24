"""Current capital-aware execution: daily 10% lagged ADV, up to 10 sessions.

Replays the existing fixed-60-day candidate book. Each purchase lot is priced
and hedged from its own fill close; same-stock sales share aggregate impact.
This is a fractional adjusted-price/notional simulation, not an order router.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
from execution_policy import CURRENT_POLICY, PREVIOUS_POLICY, CAPITALS_KRW
from research_execution_pressure import (
    allpositive, turnover_mean20, volatility60, cost_model, sell_tax,
)

ROOT = Path(__file__).resolve().parents[3]
FIN = ROOT / '04_results/signal_a/final'
DEST = FIN / 'execution_10pct_10d_20261029'


def plan_purchases(prices, turnover, first_entry, target_krw, policy=CURRENT_POLICY):
    """Use the global exchange-session axis; zero-volume sessions consume time.

    Only lagged turnover determines order size. Same-day close/turnover are
    execution checks, never inputs to the ADV limit. No future price or cost
    observation can cancel a previously planned fill.
    """
    prices, turnover = np.asarray(prices, float), np.asarray(turnover, float)
    if len(prices) != len(turnover) or not 0 <= first_entry < len(prices):
        raise ValueError('Aligned panels and valid first_entry required')
    if not np.isfinite(target_krw) or target_krw < 0:
        raise ValueError('Finite nonnegative budget required')
    remaining = float(target_krw)
    tolerance = max(1e-6, target_krw * 1e-12)
    fills, attempts = [], []
    for day in range(first_entry, min(first_entry + policy.max_entry_sessions, len(prices))):
        if remaining <= tolerance:
            break
        adv = turnover_mean20(turnover[max(0, day-policy.adv_sessions):day])
        daily_limit = policy.daily_adv_fraction * adv
        row = dict(idx=day, session_number=day-first_entry+1,
                   adv20_krw=adv, adv_last_idx=day-1,
                   daily_limit_krw=daily_limit, remaining_before_krw=remaining,
                   entry_price=prices[day], buy_notional_krw=0.,
                   buy_participation=np.nan, status='filled')
        if not np.isfinite(adv):
            row['status'] = 'missing_adv20'
        elif not np.isfinite(prices[day]) or prices[day] <= 0:
            row['status'] = 'missing_entry_price'
        elif not np.isfinite(turnover[day]) or turnover[day] <= 0:
            row['status'] = 'entry_not_tradable'
        else:
            amount = min(remaining, daily_limit)
            row.update(buy_notional_krw=amount, buy_participation=amount/adv)
            remaining = max(0., remaining-amount)
            fills.append(row.copy())
        row['remaining_after_krw'] = remaining
        attempts.append(row)
    return fills, attempts


def value_purchase_lots(fills, prices, turnover, benchmark, stock_returns,
                        dates, sell, beta, expiries, stress=False):
    """Return valued lot dictionaries. Missing holdings/costs remain unvalued.

    All lots exit together: sqrt impact uses their combined sale proceeds.
    Costs and synthetic beta hedge are incurred only on the capital filled.
    """
    if not fills:
        return []
    prices, turnover = np.asarray(prices, float), np.asarray(turnover, float)
    benchmark, stock_returns = np.asarray(benchmark, float), np.asarray(stock_returns, float)
    dates = pd.DatetimeIndex(dates)
    if not 0 <= sell < len(prices) or any(f['idx'] >= sell for f in fills):
        raise ValueError('Every fill must precede the exit session')
    exit_adv = turnover_mean20(turnover[max(0, sell-20):sell])
    sigma_sell = volatility60(stock_returns[max(1, sell-60):sell])
    exit_price = prices[sell]
    exit_tradable = np.isfinite(turnover[sell]) and turnover[sell] > 0
    aggregate_sale = (sum(f['buy_notional_krw'] * exit_price/f['entry_price'] for f in fills)
                      if np.isfinite(exit_price) and exit_price > 0 else np.nan)
    sell_part = aggregate_sale / exit_adv
    output = []
    for fill in fills:
        day = fill['idx']
        row = dict(fill)
        row.update(fill_date=str(dates[day].date()), sell_date=str(dates[sell].date()),
                   exit_price=exit_price, exit_adv20_krw=exit_adv,
                   aggregate_sale_krw=aggregate_sale, sell_participation=sell_part)
        valid_path = allpositive(prices[day:sell+1])
        valid_benchmark = allpositive(1+benchmark[day+1:sell+1])
        raw_return = exit_price/fill['entry_price']-1 if valid_path else np.nan
        bench_return = np.prod(1+benchmark[day+1:sell+1])-1 if valid_benchmark else np.nan
        sigma_buy = volatility60(stock_returns[max(1, day-60):day])
        days = (dates[sell]-dates[day]).days
        rolls = sum(dates[day] < x < dates[sell] for x in expiries)
        params = [raw_return, beta, days, rolls, fill['buy_participation'],
                  sell_part, sigma_buy, sigma_sell, sell_tax(dates[sell])]
        if not valid_path:
            status = 'missing_holding_price'
        elif not valid_benchmark or not np.isfinite(beta):
            status = 'missing_hedge_inputs'
        elif not exit_tradable:
            status = 'exit_not_tradable'
        elif not np.isfinite(params).all():
            status = 'missing_cost_inputs'
        else:
            status = 'ok'
        costs = cost_model(*params, stress=stress) if status == 'ok' else {}
        total = sum(costs.values()) if costs else np.nan
        gross = raw_return-beta*bench_return
        row.update(raw_stock_return=raw_return, benchmark_return=bench_return, beta=beta,
                   gross=gross, scenario_cost=total, net=gross-total,
                   holding_days=days, rolls=rolls, sigma_buy=sigma_buy, sigma_sell=sigma_sell,
                   valuation_status=status, **costs)
        output.append(row)
    return output


def conditional_summary(frame):
    valid = frame.dropna(subset=['ret'])
    x = valid.ret
    t, p = stats.ttest_1samp(x, 0) if len(x) >= 2 else (np.nan, np.nan)
    return dict(planned=len(frame), valued=len(valid), missing=len(frame)-len(valid),
                mean_pct=x.mean()*100, median_pct=x.median()*100, t=t, p=p,
                avg_invested_pct=valid.invested_weight.mean()*100,
                avg_unfilled_pct=valid.unfilled_eligible_weight.mean()*100,
                avg_fill_days=valid.avg_fill_days.mean(),
                exit_over_limit_count=int(valid.exit_over_limit_count.sum()))


def verify_previous(events):
    old = pd.read_csv(FIN/'research_20261029/execution_events.csv')
    old = old[old.participation_capped]
    new = events[events.policy.eq(PREVIOUS_POLICY.name)]
    merged = new.merge(old, on=['capital_krw', 'stress', 'event', 'sample'],
                       suffixes=('_new', '_old'), validate='one_to_one')
    assert len(merged) == len(old) == len(new)
    for col in ('ret', 'invested_weight', 'cost_component'):
        assert np.allclose(merged[col+'_new'], merged[col+'_old'], rtol=1e-10,
                           atol=1e-12, equal_nan=True), col
    return {'status': 'passed', 'rows': len(merged),
            'columns': ['ret', 'invested_weight', 'cost_component']}


def main():
    # Read the established return/benchmark data without loading the full
    # prediction engine and its large daily sector/float/rolling panels.
    import events_data as e
    turnover_panel = e.D['tv'].reindex(e.S)
    expiries = sorted({e.second_thu(y, m) for y in range(2000, e.S[-1].year+1)
                       for m in (3, 6, 9, 12)})
    DEST.mkdir(parents=True, exist_ok=True)
    book_path = FIN/'candidates.csv'
    book = pd.read_csv(book_path)
    book = book[book.idx.eq('KOSPI200') & book['시기'].between('2002-06', '2026-06')]
    orders, lot_rows, attempt_rows = [], [], []
    policies = (PREVIOUS_POLICY, CURRENT_POLICY)
    for event, g in book.groupby('시기'):
        for r in g.itertuples():
            ib, sell = int(r.ib), int(r.sell)
            j = e.CI[r.code]
            prices, turnover = e.PXV[:, j], turnover_panel[r.code].to_numpy(float)
            for policy in policies:
                # Avoid silently allowing a strategy with overlapping entry/exit.
                assert ib+policy.max_entry_sessions-1 < sell
                for capital in CAPITALS_KRW:
                    target = capital*r.weight
                    fills, attempts = (plan_purchases(prices, turnover, ib, target, policy)
                                       if r.traded else ([], []))
                    invested = sum(f['buy_notional_krw'] for f in fills)
                    assert invested <= target + 1e-5
                    for stress in (False, True):
                        key = dict(policy=policy.name, capital_krw=capital, stress=stress,
                                   event=event, sample=r.표본, code=r.code, name=r.name)
                        valued = value_purchase_lots(fills, prices, turnover, e.BENCH['KOSPI200'],
                            e.RT[:, j], e.S, sell, r.beta, expiries, stress=stress)
                        for lot in valued:
                            lot.update(key, weight=lot['buy_notional_krw']/capital)
                            lot['net_contribution'] = lot['weight']*lot['net']
                            lot_rows.append(lot)
                        for attempt in attempts:
                            attempt_rows.append(dict(key, **attempt,
                                attempt_date=str(e.S[attempt['idx']].date()),
                                adv_last_date=str(e.S[attempt['adv_last_idx']].date())))
                        complete = all(lot['valuation_status'] == 'ok' for lot in valued)
                        status = ('excluded_cash' if not r.traded else 'unfilled_cash' if not valued
                                  else 'ok' if complete else '|'.join(sorted({
                                      lot['valuation_status'] for lot in valued if lot['valuation_status'] != 'ok'})))
                        exit_part = valued[0]['sell_participation'] if valued else 0.
                        # Adjusted-price units are synthetic, not integer exchange shares.
                        units = sum(f['buy_notional_krw']/f['entry_price'] for f in fills)
                        orders.append(dict(key, originally_traded=bool(r.traded),
                            exclusion_reason=r.exclusion_reason, original_weight=r.weight,
                            first_entry_date=str(e.S[ib].date()), signal_date=str(e.S[ib-1].date()),
                            window_end_date=str(e.S[ib+policy.max_entry_sessions-1].date()),
                            sell_date=str(e.S[sell].date()), target_krw=target,
                            invested_krw=invested, unfilled_krw=target-invested,
                            unfilled_eligible_krw=target-invested if r.traded else 0.,
                            fill_count=len(fills), first_fill_date=valued[0]['fill_date'] if valued else '',
                            last_fill_date=valued[-1]['fill_date'] if valued else '',
                            weighted_entry_price=invested/units if units else np.nan,
                            exit_participation=exit_part,
                            exit_over_limit=bool(exit_part > policy.daily_adv_fraction+1e-12),
                            valuation_status=status,
                            gross_contribution=sum(lot['weight']*lot['gross'] for lot in valued) if complete else np.nan,
                            cost_contribution=sum(lot['weight']*lot['scenario_cost'] for lot in valued) if complete else np.nan,
                            net_contribution=sum(lot['net_contribution'] for lot in valued) if complete else np.nan))
        print('Scheduled execution', event, flush=True)
    orders = pd.DataFrame(orders)
    fills, attempts = pd.DataFrame(lot_rows), pd.DataFrame(attempt_rows)
    assert (fills.buy_participation <= fills.policy.map({p.name:p.daily_adv_fraction for p in policies})+1e-12).all()
    events = []
    keys = ['policy', 'capital_krw', 'stress', 'event', 'sample']
    for key, g in orders.groupby(keys):
        complete = g.net_contribution.notna().all()
        filled = g[g.fill_count.gt(0)]
        events.append(dict(zip(keys, key),
            ret=g.net_contribution.sum() if complete else np.nan,
            cost_component=g.cost_contribution.sum() if complete else np.nan,
            gross=g.gross_contribution.sum() if complete else np.nan,
            invested_weight=g.invested_krw.sum()/key[1],
            unfilled_eligible_weight=g.unfilled_eligible_krw.sum()/key[1],
            excluded_cash_weight=g.loc[~g.originally_traded, 'original_weight'].sum(),
            avg_fill_days=filled.fill_count.mean() if len(filled) else 0.,
            exit_over_limit_count=int(g.exit_over_limit.sum()),
            valuation_status='ok' if complete else 'unvalued_holdings_or_costs'))
    events = pd.DataFrame(events)
    old_check = verify_previous(events)
    summaries, comparisons = [], []
    for key, g in events.groupby(['policy', 'capital_krw', 'stress']):
        for sample in ('ALL', 'IS', 'OOS'):
            subset = g if sample == 'ALL' else g[g['sample'].eq(sample)]
            summaries.append(dict(zip(['policy', 'capital_krw', 'stress'], key),
                                  sample=sample, **conditional_summary(subset)))
    for (capital, stress), g in events.groupby(['capital_krw', 'stress']):
        previous = g[g.policy.eq(PREVIOUS_POLICY.name)]
        current = g[g.policy.eq(CURRENT_POLICY.name)]
        pair = previous.merge(current, on=['event','sample'], suffixes=('_previous','_current'), validate='one_to_one')
        for sample in ('ALL','IS','OOS'):
            part = pair if sample == 'ALL' else pair[pair['sample'].eq(sample)]
            part = part.dropna(subset=['ret_previous','ret_current'])
            diff = part.ret_current-part.ret_previous
            t,p = stats.ttest_1samp(diff,0) if len(diff) >= 2 else (np.nan,np.nan)
            comparisons.append(dict(capital_krw=capital, stress=stress, sample=sample,
                common_events=len(part), previous_mean_pct=part.ret_previous.mean()*100,
                current_mean_pct=part.ret_current.mean()*100, difference_pct=diff.mean()*100,
                paired_t=t, paired_p=p, previous_invested_pct=part.invested_weight_previous.mean()*100,
                current_invested_pct=part.invested_weight_current.mean()*100))
    for name, frame in [('orders',orders),('fills',fills),('attempts',attempts),('events',events),
                        ('summary',pd.DataFrame(summaries)),('comparison',pd.DataFrame(comparisons))]:
        frame.to_csv(DEST/(name+'.csv'), index=False, encoding='utf-8-sig')
    source_files = [book_path, Path(__file__), Path(__file__).with_name('execution_policy.py'),
                    Path(__file__).with_name('research_execution_pressure.py')]
    spec = dict(version='execution_10pct_10d_20261029', origin='User instruction, not historical return optimization',
        current_policy=vars(CURRENT_POLICY), previous_policy=vars(PREVIOUS_POLICY),
        capitals_krw=list(CAPITALS_KRW), candidate_book='Existing fixed60 KOSPI200 candidates; not legacy hybrid',
        budget='Original stock allocation per event; no redistribution or reinvestment; excluded slots stay cash.',
        calendar='First original entry inclusive through the 10th global exchange session; nontradable days consume the window.',
        adv='Each day: previous20 exchange-session turnover, excluding current day; exactly20 finite nonnegative observations. Zeros retained.',
        execution='Buy min(remaining original target,10% lagged ADV) at adjusted close; positive same-day price and turnover required. Stop once target filled.',
        missing='No fill without valid ADV/entry data. Missing cost/held-price observations do not cancel historical fills; affected event unvalued.',
        cost='Existing detailed standard/stress cost assumptions; per-lot entry impact, holding days and rolls; pooled same-stock sale participation for exit impact.',
        hedge='Signal beta fixed. Benchmark compound return starts separately after each fill close; no hedge on cash.',
        exit='Existing exit close retained; no exit cap imposed. Aggregate sale participation breaches recorded.',
        aggregation='Returns are arithmetic event means, not CAGR. Summary conditional on valued events; comparison uses intersection of valued events.',
        invested='Cumulative filled principal / original event budget, not time-weighted exposure; avg_unfilled_pct excludes original eligibility exclusions.',
        limits=['No intraday or closing-auction depth.', 'Adjusted-price fractional units, no integer share/corporate-action order accounting.',
                'No actual futures contract, collateral or account-level cash simulation; costs charged to return.',
                'Source historical eligibility and benchmark limitations remain. No independent new OOS.'],
        previous_regression=old_check,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (DEST/'spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).query("sample == 'ALL' and stress == False").to_string(index=False),flush=True)
    print('Previous-policy exact regression:', old_check, flush=True)


if __name__ == '__main__':
    main()
