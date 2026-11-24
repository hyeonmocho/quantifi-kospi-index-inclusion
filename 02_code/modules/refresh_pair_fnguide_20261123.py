"""Refresh the November 20 publication with code-keyed FnGuide market inputs."""
import csv
import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd

import update_pair_submission_20261122 as previous

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '04_results/kospi_pair_fnguide_20261123'
DATA = ROOT / '01_data/processed/kospi_pair_fnguide_update_20261122'
FILES = {'close': '코스피_코스닥_종가_1120', 'adjusted': '코스피코스닥_수정주가_1120',
         'turnover': '코스피_코스닥_거래대금_1120', 'shares': '코스피_코스닥_상장주식수_1120'}
ITEMS = {'close':'종가(원)', 'adjusted':'수정주가(원)', 'turnover':'거래대금(원)', 'shares':'상장주식수(주)'}


def prepare():
    OUT.mkdir(exist_ok=True)
    audit = []
    for field, stem in FILES.items():
        path = ROOT / '01_data/processed/fnguide' / (stem + '.csv')
        raw_dir = ROOT / '01_data/raw/fnguide'
        raw = raw_dir / (stem + '.xlsx')
        book = openpyxl.load_workbook(raw, read_only=True, data_only=True)
        rows = list(book.worksheets[0].iter_rows(max_row=30, values_only=True))
        book.close()
        pos = next(i for i, row in enumerate(rows) if row[0] == '코드')
        codes, names = list(rows[pos][1:]), list(rows[pos + 1][1:])
        with path.open(encoding='utf-8-sig', newline='') as fp:
            assert next(csv.reader(fp))[1:] == names
        assert len(codes) == len(set(codes))
        item = next((r[1] for r in rows if str(r[0]).strip() in ['아이템명', '계정명']), None)
        assert item == ITEMS[field], (field, item)
        values = pd.read_csv(path, header=None, skiprows=1, names=['date', *codes],
                             index_col='date', parse_dates=['date'], dtype={c: float for c in codes})
        assert values.index.is_unique and values.index.is_monotonic_increasing
        assert values.index[-1] == pd.Timestamp('2026-11-20')
        assert (values.ge(0) | values.isna()).all().all()
        tail = values.loc['2026-04-01':]
        tail.to_parquet(OUT / (field + '.parquet'))
        audit.append(dict(field=field, csv=path.name, xlsx=raw.name, item=item,
                          last=str(tail.index[-1].date()), last_valid=int(tail.iloc[-1].count()),
                          csv_sha256=previous.digest(path), xlsx_sha256=previous.digest(raw)))
        print(audit[-1], flush=True)
    (OUT / 'market_input_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


def load_panels():
    panels = {key: pd.read_parquet(OUT / (key + '.parquet')) for key in FILES}
    panels['cap'] = panels['close'].mul(panels['shares'])
    assert (panels['cap'].iloc[-1] > 0).sum() == 2637
    return panels


def run():
    previous.OUT = OUT
    previous.PRICE_PANELS = load_panels()
    previous.replay()
    previous.full_accounts()
    previous.full_accounts(cost_only=True)
    finish_replay()


def finish_replay():
    prior = ROOT/'04_results/kospi_pair_nov20_update_20261122'
    import shutil
    for name in ['input_audit.json', 'historical_input_signal_audit.csv',
                 'historical_float_selection_changes.csv', 'flow_t40_mean_path.csv']:
        shutil.copy2(prior/name, OUT/name)
    old = pd.read_csv(prior/'metrics.csv')
    new = pd.read_csv(OUT/'metrics.csv')
    comparison = old.merge(new, on=['index_id','phase'], suffixes=('_old','_new'), validate='one_to_one')
    for field in ['ending_equity_krw','cagr','sharpe','mdd']:
        comparison[field+'_difference'] = comparison[field+'_new']-comparison[field+'_old']
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(updated='2026-11-23', h2_market_inputs='FnGuide 1120 close, adjusted close, turnover and listed shares',
                    h2_cap='Unadjusted close times same-session shares', historical_prefix='Existing audited inputs through June 2026')
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    comparison.to_csv(OUT/'previous_source_comparison.csv',index=False)
    print('FnGuide replay and cost sensitivity complete', flush=True)


def factors():
    import kospi_pair_full_oos_factor_20261122 as regression
    p = load_panels()
    adj = p['adjusted']
    assert adj.index.equals(p['cap'].index) and adj.columns.equals(p['cap'].columns)
    ret = adj.where(adj.gt(0)).pct_change(fill_method=None)
    lag_cap = p['cap'].shift()
    weights = lag_cap.where(ret.notna() & lag_cap.gt(0))
    mkt = (weights*ret).sum(axis=1,min_count=100)/weights.sum(axis=1,min_count=100)
    ext = pd.DataFrame({'date':mkt.index,'MKT':mkt.to_numpy(),'market_stocks':weights.count(axis=1).to_numpy()})
    ext = ext[ext.date.gt('2026-09-07')]
    assert len(ext)==50 and ext.MKT.notna().all() and ext.date.max()==pd.Timestamp('2026-11-20')
    market = OUT/'market_factor_extension.csv'
    ext.to_csv(market,index=False)
    regression.MARKET_EXTENSION = market
    regression.FACTOR = ROOT/'01_data/processed/fnguide/팩터_1120.csv'
    regression.LEDGER = OUT
    regression.OUT = ROOT/'04_results/kospi_pair_full_oos_factor_20261123'
    regression.main()
    rows = pd.read_csv(regression.OUT/'regression_summary.csv')[
        ['strategy','period','start','end','n_days','alpha_annual','ci_low','ci_high','alpha_t','alpha_p']]
    rows['strategy'] = rows.strategy.map(regression.NAMES)
    rows['period'] = rows.period.map({'FULL':'전체(IS+OOS)','OOS':'OOS'})
    rows.columns = ['전략','기간','시작일','종료일','거래일 수','연율 알파','95% 하한','95% 상한','t값','p값']
    rows.to_csv(OUT/'four_factor_summary.csv',index=False,encoding='utf-8-sig')


def publish():
    import shutil
    import package_pair_update_20261122 as package
    package.OUT = OUT
    package.DEST = package.PKG/'04_results'/OUT.name
    package.FNGUIDE_MARKET = True
    package.export()
    data = package.PKG/'01_data/processed/kospi_pair_fnguide_update_20261122'
    shutil.copy2(ROOT/'01_data/processed/fnguide/팩터_1120.csv', data/'팩터_1120.csv')
    for name in ('close','adjusted','turnover','shares'):
        shutil.copy2(OUT/f'{name}.parquet',data/f'{name}.parquet')
    package.notebook()
    package.docs()


def presentation():
    import shutil
    import runpy
    import update_ppt_nov20_20261122 as ppt
    ppt.SOURCE = OUT
    ppt.OUT = ROOT/'04_results/ppt_fnguide_20261123'
    ppt.OUT.mkdir(exist_ok=True)
    ppt.COVER_DATE = '2026.11.23'
    for path in ppt.PATHS:
        backup = ppt.OUT/path.name
        if not backup.exists():
            shutil.copy2(path, backup)
    ppt.update()
    ppt.fix_charts()
    runpy.run_module('finalize_ppt_nov20_20261122',run_name='__main__')


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--replay' in sys.argv:
        run()
    elif '--factors' in sys.argv:
        factors()
    elif '--finish-replay' in sys.argv:
        finish_replay()
    elif '--publish' in sys.argv:
        publish()
    elif '--ppt' in sys.argv:
        presentation()
