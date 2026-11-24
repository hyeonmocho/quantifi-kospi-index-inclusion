"""Fixed four-factor attribution: standalone accounts and shared combined account.

Descriptive pooled IS/OOS returns, not a continuous-capital historical replay.
No strategy or factor-model selection is performed here.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / '04_results/kospi_pair_nov20_update_20261122'
OUT = ROOT / '04_results/kospi_pair_full_oos_factor_20261122'
FACTOR = ROOT / '01_data/processed/fnguide/팩터.csv'
MARKET_EXTENSION = None
NOTEBOOK = ROOT/'03_notebooks/KOSPI200_KOSPI100_전체기간_OOS_팩터회귀.ipynb'
MARKET = ROOT / '04_results/vendor_factor_demand_20261113/vendor_factor/daily_factors.csv'
CD = ROOT / '01_data/processed/fnguide/CD_91.csv'
OLD_CD = ROOT / '01_data/processed/fnguide/CD91_일별.csv'
MAP = {'Size & Book Value(2X3) SMB': 'SMB',
       'Size & Book Value(2X3) HML': 'HML',
       'Size & Momentum(2X3) Mom': 'MOM',
       'Size & Momentum(2X3) SMB': 'SMB_MOM'}
FACTORS = ['MKT_RF', 'SMB', 'HML', 'MOM']
NAMES = {'KOSPI200': 'KOSPI200 단독', 'KOSPI100': 'KOSPI100 단독', 'COMBINED': '두 전략 통합'}


def fit(d, lag=20, columns=FACTORS):
    x = sm.add_constant(d[columns], has_constant='add')
    y = d.excess_return
    fit = sm.OLS(y, x, missing='raise').fit(cov_type='HAC', cov_kwds={'maxlags': lag}, use_t=True)
    # Independent least-squares check of the reported point estimates.
    assert np.allclose(np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0], fit.params, atol=1e-12)
    ci = fit.conf_int().loc['const'] * 252
    result = dict(n_days=len(d), start=str(d.date.min().date()), end=str(d.date.max().date()),
                  alpha_daily=float(fit.params['const']), alpha_annual=float(fit.params['const'] * 252),
                  ci_low=float(ci.iloc[0]), ci_high=float(ci.iloc[1]),
                  alpha_t=float(fit.tvalues['const']), alpha_p=float(fit.pvalues['const']),
                  r_squared=float(fit.rsquared), hac_lag=lag,
                  position_day_fraction=float(d.any_position.mean()))
    coeff = [dict(factor=k, coefficient=float(fit.params[k]), t=float(fit.tvalues[k]),
                  p=float(fit.pvalues[k])) for k in fit.params.index]
    return result, coeff


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, sources = [], [FACTOR, MARKET, CD, OLD_CD]
    for strategy in NAMES:
        for phase in ['IS', 'OOS']:
            file = LEDGER / (f'{phase}_daily.parquet' if strategy == 'COMBINED' else f'{strategy}_{phase}_daily.parquet')
            sources.append(file)
            d = pd.read_parquet(file)
            d = d[d.index_id.eq(strategy)].sort_values('date').copy()
            assert d.date.is_unique and d[['return_daily', 'equity']].notna().all().all()
            initial = 20e9 if strategy == 'COMBINED' else 10e9
            expected = d.equity / d.equity.shift(1).fillna(initial) - 1
            assert np.allclose(expected, d.return_daily, atol=1e-12)
            assert np.allclose(d.equity, d.cash + d.held_value, atol=1, rtol=1e-12)
            d['phase'] = phase
            records.append(d)
    returns = pd.concat(records, ignore_index=True)
    assert not returns.duplicated(['index_id', 'date']).any()
    # Preserve the original market-factor definition; do not splice KOSPI-only quotes.
    market = pd.read_csv(MARKET, parse_dates=['date'])[['date', 'MKT']]
    if MARKET_EXTENSION is not None:
        extension = pd.read_csv(MARKET_EXTENSION, parse_dates=['date'])[['date', 'MKT']]
        assert market.date.max() < extension.date.min()
        market = pd.concat([market, extension], ignore_index=True)
        sources.append(MARKET_EXTENSION)
    last_market = market.dropna().date.max()
    calendar = pd.DatetimeIndex(sorted(set(market.date) | set(returns.date)), name='date')
    raw = pd.read_csv(FACTOR, encoding='utf-8-sig')
    levels = raw.rename(columns={raw.columns[0]: 'date', **MAP})
    levels['date'] = pd.to_datetime(levels.date)
    levels = levels.set_index('date')[list(k for k in MAP.values() if k in levels)].apply(pd.to_numeric, errors='raise')
    assert levels.index.is_unique and levels.index.is_monotonic_increasing
    styles = levels.reindex(calendar).pct_change(fill_method=None)
    rates = pd.read_csv(CD, encoding='utf-8-sig')
    rates.columns = ['rate_date', 'annual_percent']
    rates['rate_date'] = pd.to_datetime(rates.rate_date)
    old = pd.read_csv(OLD_CD, parse_dates=['Date']).rename(columns={'Date': 'rate_date', 'ECO': 'annual_percent'})
    overlap = old.merge(rates, on='rate_date', suffixes=('_old', '_new'))
    assert np.allclose(overlap.annual_percent_old, overlap.annual_percent_new)
    rates = pd.concat([old, rates], ignore_index=True).drop_duplicates('rate_date', keep='last').sort_values('rate_date')
    dates = pd.DataFrame({'date': calendar})
    dates['calendar_days'] = dates.date.diff().dt.days.fillna(1)
    rf = pd.merge_asof(dates, rates, left_on='date', right_on='rate_date', direction='backward', allow_exact_matches=False)
    rf['RF'] = (1 + rf.annual_percent / 100) ** (rf.calendar_days / 365) - 1
    f = rf.merge(market, on='date', how='left', validate='one_to_one').merge(styles.reset_index(), on='date', how='left', validate='one_to_one')
    f['MKT_RF'] = f.MKT - f.RF
    panel = returns.merge(f, on='date', how='left', validate='many_to_one')
    panel['excess_return'] = panel.return_daily - panel.RF
    panel['usable'] = panel[['excess_return', *FACTORS]].notna().all(axis=1)
    panel.to_parquet(OUT / 'aligned_inputs.parquet', index=False)
    f.to_csv(OUT / 'factor_inputs.csv', index=False, encoding='utf-8-sig')
    coverage, missing, results, coefficients = [], [], [], []
    for strategy in NAMES:
        for period in ['FULL', 'OOS']:
            g = panel[panel.index_id.eq(strategy)]
            if period == 'OOS':
                g = g[g.phase.eq('OOS')]
            g = g.sort_values('date')
            d = g[g.usable].copy()
            assert d.date.is_unique and (d.rate_date < d.date).all()
            assert d.date.max() == last_market
            # All exclusions must be a trailing factor-coverage gap, never silently remove internal days.
            assert g[g.date.le(d.date.max())].usable.all()
            coverage.append(dict(strategy=strategy, period=period, requested_start=str(g.date.min().date()),
                                 requested_end=str(g.date.max().date()), actual_start=str(d.date.min().date()),
                                 actual_end=str(d.date.max().date()), requested_days=len(g), used_days=len(d), excluded_days=len(g)-len(d)))
            for row in g[~g.usable].itertuples():
                missing.append(dict(strategy=strategy, period=period, date=row.date,
                                    missing_fields=';'.join(c for c in ['RF', *FACTORS] if pd.isna(getattr(row, c)))))
            for lag in [5, 20, 60]:
                result, coef = fit(d, lag)
                results.append(dict(strategy=strategy, period=period, **result))
                coefficients.extend(dict(strategy=strategy, period=period, hac_lag=lag, **r) for r in coef)
    summary = pd.DataFrame(results)
    main = summary[summary.hac_lag.eq(20)].copy()
    for name, frame in [('regression_summary', main), ('hac_sensitivity', summary),
                        ('factor_coefficients', pd.DataFrame(coefficients)), ('coverage', pd.DataFrame(coverage)),
                        ('excluded_dates', pd.DataFrame(missing))]:
        frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    # Regression is invariant to temporal order, but HAC is not; all fits above use chronological rows.
    assert len(main) == 6 and main[['alpha_annual', 'alpha_t', 'alpha_p']].notna().all().all()
    report = main.copy()
    report['strategy'] = report.strategy.map(NAMES)
    report['period'] = report.period.map({'FULL': '전체(IS+OOS)', 'OOS': 'OOS'})
    report['alpha_annual'] = report.alpha_annual.map(lambda v: f'{v:.2%}')
    report['95% CI'] = main.apply(lambda r: f'[{r.ci_low:.2%}, {r.ci_high:.2%}]', axis=1)
    table = report[['strategy', 'period', 'start', 'end', 'n_days', 'alpha_annual', 'alpha_t', 'alpha_p', '95% CI', 'r_squared']]
    table.columns = ['전략', '기간', '시작일', '종료일', '거래일 수', '연율 알파', 't값', 'p값', '95% 신뢰구간', 'R²']
    for col in ['t값', 'p값', 'R²']:
        table[col] = table[col].map(lambda v: f'{v:.4f}')
    table.to_csv(OUT / '결과요약.csv', index=False, encoding='utf-8-sig')
    limitations = [
        'IS와 OOS는 각각 초기 원금으로 시작한 기존 계좌다. 전체기간은 각 구간의 일별 수익률을 연결한 사후 귀속 분석이며 단일 연속 자본 원장의 재백테스트가 아니다. 경계일 NAV 리셋을 손익으로 계산하지 않았다.',
        '단독 전략은 각자의 최대 가용기간을 사용한다. 통합 전략은 두 전략을 함께 실행하고 거래대금 한도를 공유한 계좌이며, 두 단독 수익률의 단순 평균이 아니다.',
        'MKT는 기존 국내 주식 수정주가 캐시의 전일 시가총액 가중 시장 대용이며 2026-09-07까지다. 이후 유가증권시장 중심 원시 시세를 임의 연결하지 않았다.',
        '팩터.csv의 스타일 지수는 2026-11-12까지 유효하다. 11월13일은 결측이며, 11월20일까지의 계좌 성과를 전부 회귀한 것은 아니다.',
        'SMB/HML/MOM은 FnGuide 팩터 지수 수준의 일별 변화율 대용이다. 원시 자기금융 롱숏 팩터 수익률과의 동일성은 인증되지 않았으므로 공식 FF4 알파로 단정하지 않는다.',
        '현금 보유일을 포함한 비용 차감 계좌 수익률에서 CD91 RF를 차감한다. 원장 현금에는 별도 이자 수익을 가산하지 않았다.',
        'RF는 해당 수익일보다 앞선 CD91 금리를 사용하고 주말·휴일을 포함한 경과일/365로 환산했다. 스타일 팩터에는 RF를 중복 차감하지 않았다.',
        '연율 알파는 일별 절편×252인 산술 연율치다. CAGR, 실현 가능한 헤지 수익, 지수 편입의 인과효과와 다르다.',
        '주표는 기존과 동일한 HAC 20거래일·t분포 양측검정, 95% 신뢰구간이다. 5·60일은 사전 지정 민감도이며 유의해지는 시차를 선택하지 않았다.',
        '전체기간에는 전략 선택에 사용한 IS가 포함된다. OOS도 이전 연구에서 반복 관찰한 구간이며 새 독립 전진표본이 아니다. 일별 관측수는 독립 이벤트 수가 아니다.',
    ]
    if MARKET_EXTENSION is not None:
        limitations[2] = '기존 MKT를 9월 7일까지 유지했습니다. 9월 8일부터 FnGuide 수정주가 일간 수익률을 전일 시총으로 가중했습니다. 시총은 수정 전 종가×상장주식수로 계산했습니다.'
        limitations[3] = '새 SMB·HML·MOM과 CD91을 사용해 제외됐던 50거래일을 포함했습니다.'
    audit = dict(run_date='2026-11-23' if MARKET_EXTENSION else '2026-11-22', main_factors=FACTORS, model='Rp-RF = alpha + beta*(MKT-RF,SMB,HML,MOM) + error',
                 specification_changed=False, strategy_reoptimized=False,
                 primary_hac_lag=20, factor_coverage_end=str(last_market.date()),
                 sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(sources)},
                 checks=['Six requested regressions completed', 'No internal date gaps silently removed',
                         'Original within-phase NAV returns independently reproduced', 'OLS coefficients independently reproduced using numpy lstsq',
                         'RF dates strictly earlier than returns', 'No artificial NAV-reset return at IS/OOS boundary'],
                 limitations=limitations)
    (OUT / 'protocol.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    interpretation = ('전체기간에서는 KOSPI200·KOSPI100·통합 계좌 모두 양의 알파가 5% 기준으로 유의합니다. '
                      'OOS에서는 알파의 점추정치가 더 크지만 95% 신뢰구간이 모두 0을 포함합니다. '
                      '이는 알파가 없다는 확정이 아니라 추정 불확실성이 크다는 결과입니다. '
                      'HAC 5·20·60일에서 전체기간의 5% 판정은 유지되나, OOS KOSPI200과 통합은 60일에서만 5% 기준을 충족합니다. '
                      '주 결과는 기존 20일 기준을 유지하며 유의해지는 시차를 사후 채택하지 않습니다. '
                      '전체기간 결과에는 IS의 모형 선택 효과가 포함되어 독립적인 미래 수익 검증으로 해석할 수 없습니다.')
    text = '# KOSPI200·KOSPI100 전체기간 및 OOS 4팩터 회귀\n\n작성일: 2026-11-22\n\n' + table.to_markdown(index=False) + '\n\n## 결과 해석\n\n' + interpretation + '\n\n## 계산 기준과 범위\n\n' + '\n'.join('- ' + s for s in limitations) + '\n'
    (OUT / '분석결과.md').write_text(text, encoding='utf-8')
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [nbf.v4.new_markdown_cell('# 두 지수 편입 전략의 팩터 회귀: 전체기간과 OOS\n\n2026-11-22. 최신 계좌 원장을 사용하되 동일 정의의 팩터 공통 가용기간인 **2026-09-07까지** 평가합니다. 단독 두 전략과 실제 통합 계좌, 총 여섯 회귀를 비교합니다.'),
                nbf.v4.new_markdown_cell('## 무엇을 검증하는가\n\n$R_{p,t}-R_{f,t}=\\alpha+\\beta_M(MKT_t-R_{f,t})+\\beta_S SMB_t+\\beta_H HML_t+\\beta_U MOM_t+\\epsilon_t$\n\n알파는 지정한 팩터로 설명되지 않는 평균 초과수익입니다. 주표: HAC 20거래일, 양측검정. 연율 알파는 일별 절편×252이며 CAGR가 아닙니다.'),
                nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'04_results/kospi_pair_full_oos_factor_20261122').exists())\nOUT = ROOT/'04_results/kospi_pair_full_oos_factor_20261122'\ndisplay(pd.read_csv(OUT/'결과요약.csv'))"),
                nbf.v4.new_markdown_cell('## 결과 해석\n\n' + interpretation),
                nbf.v4.new_markdown_cell('## 사용 날짜와 제외 날짜\n\n11월20일까지 계좌 원장은 있으나 시장 팩터가 먼저 끝납니다. 팩터 결측일을 0으로 채우지 않고 제외 내역을 저장했습니다.'),
                nbf.v4.new_code_cell("display(pd.read_csv(OUT/'coverage.csv'))"),
                nbf.v4.new_markdown_cell('## 팩터 노출과 표준오차 시차 민감도'),
                nbf.v4.new_code_cell("coef = pd.read_csv(OUT/'factor_coefficients.csv')\ndisplay(coef[coef.hac_lag.eq(20)])\ndisplay(pd.read_csv(OUT/'hac_sensitivity.csv')[['strategy','period','hac_lag','alpha_annual','alpha_t','alpha_p']])"),
                nbf.v4.new_markdown_cell('## 해석 범위\n\n' + '\n\n'.join('- ' + s for s in limitations)),
                nbf.v4.new_markdown_cell('## 재현\n\n프로젝트 루트에서 `python 02_code/modules/kospi_pair_full_oos_factor_20261122.py`를 실행하면 원장과 원자료로 다시 추정합니다. 입력 파일 SHA256과 검증 내역은 결과 폴더의 protocol.json에 기록했습니다.')]
    notebook = NOTEBOOK
    if MARKET_EXTENSION is not None:
        def result_sentence(period):
            q = main[main.period.eq(period)]
            return ', '.join(f'{NAMES[r.strategy]} {r.alpha_annual:+.2%} (p={r.alpha_p:.4f})' for r in q.itertuples())
        interpretation = '전체기간 연율 알파는 ' + result_sentence('FULL') + '입니다. OOS는 ' + result_sentence('OOS') + '입니다.'
        text = '# KOSPI200·KOSPI100 전체기간 및 OOS 4팩터 회귀\n\n' + table.to_markdown(index=False) + '\n\n' + interpretation + '\n\n' + '\n'.join('- '+s for s in limitations)
        (OUT/'분석결과.md').write_text(text, encoding='utf-8')
        nb.cells[0].source = '# 두 지수 편입 전략의 팩터 회귀\n\n2026년 11월 20일까지 단독 전략 두 개와 통합계좌를 분석했습니다.'
        for c in nb.cells:
            c.source = c.source.replace('kospi_pair_full_oos_factor_20261122', OUT.name)
            if c.source.startswith('## 결과 해석'):
                c.source = '## 결과 해석\n\n'+interpretation
            if c.source.startswith('## 사용 날짜'):
                c.source = '## 사용 날짜\n\n시장 및 스타일 팩터 공통 가용기간은 2026년 11월 20일까지입니다.'
            if c.source.startswith('## 재현'):
                c.source = '## 재현\n\n`python 02_code/modules/refresh_pair_fnguide_20261123.py --factors`'
    nbf.write(nb, notebook)
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbf.write(nb, notebook)
    print(table.to_string(index=False))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == '__main__':
    main()
