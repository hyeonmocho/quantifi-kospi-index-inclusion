"""Export all end-of-day holdings from the bundled simulated ledgers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '04_results/kospi_pair_optimized_trend_20261124'


def main(verify_previous_snapshot=False):
    holdings = pd.concat([pd.read_parquet(SOURCE / f'{p}_holdings.parquet').assign(phase=p)
                          for p in ('IS', 'OOS')], ignore_index=True)
    daily = pd.concat([pd.read_parquet(SOURCE / f'{p}_daily.parquet').assign(phase=p)
                       for p in ('IS', 'OOS')], ignore_index=True)
    account = daily[daily.index_id.eq('COMBINED')].sort_values('date').copy()
    assert account.date.is_unique
    assert not holdings.duplicated(['date', 'index_id', 'code']).any()
    assert holdings.market_value.gt(0).all()
    assert set(holdings.date).issubset(set(account.date))
    totals = holdings.groupby('date').market_value.sum().reindex(account.date, fill_value=0).to_numpy()
    np.testing.assert_allclose(totals, account.held_value, rtol=1e-12, atol=1e-4)
    np.testing.assert_allclose(account.held_value + account.cash, account.equity, rtol=1e-12, atol=1e-4)

    primary = pd.read_csv(ROOT / '04_results/event_factor_prediction_20261113/pbr_name_mapping.csv').drop_duplicates('code').set_index('code')['name']
    fallback = pd.read_csv(ROOT / '01_data/processed/index_data_extension_20261117/legacy_security_names.csv').drop_duplicates('code').set_index('code').iloc[:, 0]
    fresh = pd.read_csv(ROOT / '01_data/processed/kospi_pair_fnguide_update_20261122/KOSPI200_names.csv').set_index('code')['name']
    names = primary.combine_first(fallback).combine_first(fresh)
    holdings['종목명'] = holdings.code.map(names)
    assert holdings['종목명'].notna().all()
    holdings = holdings.merge(account[['date', 'equity']], on='date', validate='many_to_one')
    holdings['전략신호'] = holdings.index_id.map({'KOSPI200': 'KOSPI200 정기 신규편입 예측', 'KOSPI100': 'KOSPI100 정기 신규편입 예측'})
    holdings['자산구분'] = '주식'
    holdings['weight'] = holdings.market_value / holdings.equity
    columns = {'code': '종목코드', 'weight': '비중', 'phase': '검증구간',
               'market_value': '종목평가액원', 'equity': '통합계좌자산원'}
    keep = ['date', '종목명', '종목코드', '비중', '전략신호', '검증구간',
            '자산구분', '종목평가액원', '통합계좌자산원']
    stock = holdings.rename(columns=columns)[keep].sort_values(['date', '전략신호', '종목코드'])
    cash = pd.DataFrame({'date': account.date, '종목명': '현금', '종목코드': 'CASH',
                         '비중': account.cash / account.equity, '전략신호': '통합계좌 현금',
                         '검증구간': account.phase, '자산구분': '현금',
                         '종목평가액원': account.cash, '통합계좌자산원': account.equity})
    all_positions = pd.concat([stock, cash], ignore_index=True).sort_values(['date', '자산구분', '전략신호', '종목코드'])
    assert not all_positions.duplicated(['date', '종목코드', '전략신호']).any()
    np.testing.assert_allclose(all_positions.groupby('date')['비중'].sum(), 1.0, atol=1e-12)
    assert set(all_positions.date) == set(account.date)
    assert all_positions['비중'].between(0, 1).all()
    if verify_previous_snapshot:
        old = pd.read_csv(ROOT / '포트폴리오_신호일_실제보유.csv', parse_dates=['date'])
        check = old[['date', '종목코드', '전략신호', '비중']].merge(stock, on=['date', '종목코드', '전략신호'], suffixes=('_old', '_daily'), validate='one_to_one')
        assert len(check) == len(old)
        np.testing.assert_allclose(check['비중_old'], check['비중_daily'], atol=5e-8, rtol=0)

    for frame, name in [(all_positions, '포트폴리오_일별_실제보유.csv'),
                         (stock, '포트폴리오_일별_주식보유.csv')]:
        saved = frame.copy()
        saved['date'] = saved.date.dt.strftime('%Y-%m-%d')
        saved.to_csv(ROOT / name, index=False, encoding='utf-8-sig', float_format='%.12g')
    stats = dict(first=str(account.date.min().date()), last=str(account.date.max().date()),
                 trading_days=len(account), stock_days=stock.date.nunique(),
                 cash_only_days=int(account.held_value.eq(0).sum()), stock_rows=len(stock),
                 all_rows=len(all_positions), last_stock_date=str(stock.date.max().date()),
                 previous_snapshot_verified=verify_previous_snapshot,
                 valuation='End-of-day simulated holdings. Weight denominator is combined account equity.',
                 split='IS and OOS are separately initialized account paths; not one uninterrupted NAV across 2023-01-01.')
    (SOURCE / 'daily_portfolio_export_audit.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
