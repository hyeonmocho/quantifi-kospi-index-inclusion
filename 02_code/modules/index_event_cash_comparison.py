"""Compare OOS executions to holding each deployed won at CD91 until its release."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from index_three_portfolio_20261117 import engine_selected_checkpoint as engine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '04_results/common_index_engine_20261119/long_IS_compromise'
GROUP = ['index_id', 'event']


def detailed_orders():
    saved = pd.read_parquet(OUT / 'portfolio_orders.parquet')
    if 'units' in saved:
        return saved
    # Older ledgers lack the tranche, units and merger terms. Replay the locked
    # orders and require every archived order field to agree before using them.
    import v18_long_replay as replay
    resources = replay.load('02_code/modules/index_forecast_is_oos_20261117/evaluate.py', 'event_cash_inputs')
    resources.write = lambda *args: None
    prices, turnover, actions = replay.supplement(*resources.resources())
    forecasts = pd.read_csv(OUT.parent / 'forecasts.csv', parse_dates=['entry_date', 'exit_date'])
    targets = pd.read_csv(OUT.parent / 'targets.csv')
    counts = json.loads((OUT / 'selection.json').read_text())
    sleeves = pd.DataFrame({'index_id': list(counts), 'model': forecasts.model.unique().item()})
    engine.BUDGET = 1e10
    _, _, _, orders, _, checks = engine.simulate(
        sleeves, forecasts[forecasts.event.between('2023-01', '2026-06')], forecasts, targets,
        prices, turnover, actions, start='2023-01-01', end='2026-06-30',
        lead_map={k: replay.LEADS[:v] for k, v in counts.items()})
    assert checks['all_events_closed']
    pd.testing.assert_frame_equal(saved, orders[saved.columns], rtol=1e-12, atol=1e-5)
    orders.to_parquet(OUT / 'portfolio_orders.parquet', index=False)
    return orders


def match_fills(orders, clock):
    open_lots = defaultdict(list)
    closed = []
    for r in orders.itertuples():
        key = (r.index_id, r.event, r.lead, r.code)
        lots = open_lots[key]
        if r.side == 'buy':
            lots.append(dict(entry=r.date, original_code=r.code, units=r.units,
                             cost=r.notional + r.fee))
            continue
        balance = sum(p['units'] for p in lots)
        assert balance > 0 and 0 < r.units <= balance + max(1e-8, balance * 1e-12)
        if pd.notna(getattr(r, 'successor', None)):
            assert np.isclose(r.units, balance) and r.conversion > 0
            for p in lots:
                p['units'] *= r.conversion
            open_lots[(*key[:3], r.successor)].extend(lots)
            del open_lots[key]
            continue
        proceeds = (r.notional if r.side == 'sell' else r.cash_delivery) - r.fee
        fraction = min(1., r.units / balance)
        for p in lots:
            cost = p['cost'] * fraction
            interest = cost * (clock[r.date] - clock[p['entry']])
            closed.append(dict(index_id=r.index_id, event=r.event, lead=r.lead,
                               entry=p['entry'], exit=r.date, original_code=p['original_code'],
                               exit_code=r.code, cost=cost, pnl=proceeds*p['units']/balance-cost,
                               interest=interest))
            p['cost'] -= cost
            p['units'] *= 1 - fraction
        if fraction >= 1 - 1e-12:
            del open_lots[key]
    assert sum(p['cost'] for lots in open_lots.values() for p in lots) < .01
    return pd.DataFrame(closed)


def calculate():
    orders = detailed_orders()
    daily = pd.read_parquet(OUT / 'portfolio_daily.parquet')
    dates = pd.DatetimeIndex(daily.loc[daily.index_id.eq('COMBINED'), 'date'])
    rf = engine.cd91_rates().loc[dates[:-1]].to_numpy()
    clock = pd.Series(np.r_[0., np.cumsum(rf * (dates[1:]-dates[:-1]).days / 365)], index=dates)
    fills = match_fills(orders, clock)
    events = fills.groupby(GROUP)[['cost', 'pnl', 'interest']].sum()
    tranches = pd.read_parquet(OUT / 'portfolio_tranches.parquet')
    expected = tranches.groupby(GROUP).pnl.sum()
    events = events.reindex(expected.index, fill_value=0)
    np.testing.assert_allclose(events.pnl, expected, rtol=0, atol=.02)
    buys = orders[orders.side.eq('buy')]
    np.testing.assert_allclose(events.cost, (buys.notional+buys.fee).groupby(
        [buys.index_id, buys.event]).sum().reindex(expected.index, fill_value=0), rtol=0, atol=.02)
    total = events.groupby('event').sum().assign(index_id='COMBINED').reset_index().set_index(GROUP)
    events = pd.concat([events, total]).reset_index()
    events['excess_pnl'] = events.pnl - events.interest
    events['excess_return'] = events.excess_pnl / events.cost.replace(0, np.nan)
    account = pd.read_csv(OUT / 'portfolio.csv').set_index('index_id')
    np.testing.assert_allclose(events.groupby('index_id').pnl.sum().reindex(account.index),
                               account.profit_krw, atol=.02, rtol=0)
    summaries = []
    for idx, group in events.groupby('index_id', sort=False):
        returns = group.excess_return.dropna()
        wins, losses = returns[returns > 0], returns[returns < 0]
        summaries.append(dict(index_id=idx, events=len(returns), wins=len(wins),
                              mean=returns.mean(), median=returns.median(), win_rate=len(wins)/len(returns),
                              mean_win=wins.mean(), mean_loss=losses.mean(), payoff=wins.mean()/-losses.mean(),
                              pnl=group.pnl.sum(), interest=group.interest.sum(), excess_pnl=group.excess_pnl.sum()))
    summary = pd.DataFrame(summaries)
    fills.to_parquet(OUT / 'event_cd_matched_fills.parquet', index=False)
    events.to_csv(OUT / 'event_cd_returns.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT / 'event_cd_summary.csv', index=False, encoding='utf-8-sig')
    return events, summary


if __name__ == '__main__':
    events, summary = calculate()
    print(summary.to_string(index=False))
    print(events.pivot(index='event', columns='index_id', values='excess_return').round(5))
