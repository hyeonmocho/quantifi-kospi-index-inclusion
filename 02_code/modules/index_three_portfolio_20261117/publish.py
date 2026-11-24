from pathlib import Path
import json
import pandas as pd
import numpy as np
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.io import capture_output

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'04_results/index_three_portfolio_20261117'


def validate_saved():
    daily=pd.read_parquet(OUT/'daily.parquet');ledger=pd.read_parquet(OUT/'ledger.parquet')
    holdings=pd.read_parquet(OUT/'holdings.parquet');monthly=pd.read_csv(OUT/'monthly.csv')
    results=[]
    for sc,g in daily.groupby('scenario'):
        combined=g[g.index_id.eq('COMBINED')].set_index('date').sort_index()
        sleeves=g[~g.index_id.eq('COMBINED')]
        summed=sleeves.groupby('date')[['equity','cash','held_value']].sum()
        assert np.allclose(summed,combined[summed.columns],atol=.02,rtol=0)
        assert (sleeves.cash>=-.001).all()
        assert np.allclose(combined.equity,combined.cash+combined.held_value,atol=.02,rtol=0)
        previous=combined.equity.shift(1).fillna(9e9)
        assert np.allclose(combined.return_daily,combined.equity/previous-1,atol=1e-12)
        hh=holdings[holdings.scenario.eq(sc)].groupby(['date','index_id']).market_value.sum()
        sv=sleeves.set_index(['date','index_id']).held_value
        assert np.allclose(hh.reindex(sv.index).fillna(0),sv,atol=.02,rtol=0)
        trades=ledger[ledger.scenario.eq(sc)&ledger.side.isin(['buy','sell'])]
        amounts=trades.groupby(['date','code']).agg(amount=('notional','sum'),limit=('limit','min'))
        assert amounts.amount.le(amounts.limit+.01).all()
        assert np.allclose(trades.fee,trades.notional*.0025,atol=.01,rtol=0)
        m=monthly[(monthly.scenario==sc)&(monthly.index_id=='COMBINED')]
        assert np.isclose((1+m.return_on_start_equity).prod(),combined.equity.iloc[-1]/9e9)
        assert abs(m.profit_krw.sum()-(combined.equity.iloc[-1]-9e9))<.02
        nav=combined.equity/9e9
        running=np.maximum.accumulate(np.r_[1.,nav])[1:]
        dd=nav/running-1;trough=dd.idxmin();peak=nav.loc[:trough].idxmax()
        results.append(dict(scenario=sc,sessions=len(combined),shared_cap_breaches=0,
                            accounting_passed=True,monthly_reconciliation=True,
                            MDD_peak=str(peak.date()),MDD_trough=str(trough.date())))
    pd.DataFrame(results).to_csv(OUT/'saved_ledger_validation.csv',index=False)


def main():
    validate_saved()
    nb=nbformat.v4.new_notebook();cells=[]
    md=lambda s:cells.append(nbformat.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbformat.v4.new_code_cell(s))
    md('''# KOSPI200·KOSPI100·KRX300 예측 편입주 통합 성과

**각 전략 30억원, 총 90억원. OOS 2023-01-02~2026-06-30.**

같은 규정 엔진의 성과가 갑자기 급감한 것처럼 보인 원인은 **IS와 OOS, F1 선택과 샤프 선택을 혼용한 설명**이었습니다. 이 노트는 수치의 출처를 먼저 구분하고, 기존 IS 선택을 바꾸지 않은 세 전략의 원화 손익을 합칩니다.

KRX300의 자본잠식 제외·과거 유니버스·최신 전체 규정 인증은 미완료입니다. 이번 결과는 **재무 데이터 보완 전 기존 연구 엔진의 통합 성과**이며, 실전 주문이 아닙니다. OOS는 기존 연구에서 이미 관찰한 기간입니다.''')
    code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd()
while not (ROOT/'04_results').exists():
    if ROOT==ROOT.parent: raise RuntimeError('프로젝트 루트를 찾지 못했습니다.')
    ROOT=ROOT.parent
OUT=ROOT/'04_results/index_three_portfolio_20261117'
labels={'sharpe_selected':'IS 샤프 선택','f1_selected':'IS F1 선택','f1_cap_third':'IS F1 선택 + 종목당 1/3 상한'}
pd.set_option('display.max_columns',20)
def table(df, pct=(), decimals=()):
    x=df.copy()
    for c in pct:x[c]=x[c].map(lambda v:f'{v:.2%}' if pd.notna(v) else '—')
    for c in decimals:x[c]=x[c].map(lambda v:f'{v:.3f}' if pd.notna(v) else '—')
    display(x)
perf=pd.read_csv(OUT/'performance.csv')
''')
    md('''## 1. KRX300 샤프가 서로 다른 이유

- **0.863 → 0.880:** IS에서 샤프 최대 모델을 선택한 뒤 OOS에서 평가. 투자 매력도 모형·심사 80거래일 전 진입은 같고, 전략 원금을 3억원에서 30억원으로 변경한 비교입니다.
- **0.135와 0.029:** 각각 F0.5·F1 최대 모델의 **IS** 성과입니다. 다른 예측 모형·40거래일 진입이며 원금도 3억원입니다.
- **F1 모델을 OOS·30억원으로 맞춘 값은 0.341**입니다. 따라서 기간 혼용뿐 아니라, 예측 정확도로 선택한 모델과 투자성과로 선택한 모델의 경제적 차이도 존재합니다.

3억→30억원에서 일부 청산이 늦어져 관측 수익이 개선됐지만, 규모가 커지면 일반적으로 알파가 좋아진다는 뜻은 아닙니다. 비선형 시장충격은 새로 추정하지 않았습니다.''')
    code('''audit=pd.read_csv(OUT/'KRX300_sharpe_source_audit.csv')
audit['원금(억원)']=audit.budget_krw/1e8
cols=['phase','model','lead','period_start','period_end','원금(억원)','cagr','sharpe','mdd']
table(audit[cols],['cagr','mdd'],['sharpe'])
''')
    md('''## 2. 무엇을 합쳤나

| 방식 | KOSPI200 | KOSPI100 | KRX300 |
|---|---|---|---|
| IS 샤프 선택 | 최근 20일 평균 시총 모형·80일 | 기존 규칙·60일 | 과거 조건부 투자 매력도·80일 |
| IS F1 선택 | 축소 추세 시총 모형·40일 | 기존 규칙·60일 | 시나리오 확률 보정·동일 종목 수·40일 |
| F1 + 종목당 상한 | 위 F1 모델에 전략 내 종목당 원금 1/3 상한 | 동일 | 동일 |

일수는 **공표일이 아니라 심사기준일 이전 예측 시점**입니다. 매매는 저장된 신호·진입·청산 일정을 사용합니다. 세 번째 안은 등록된 전진집행의 종목 상한을 기존 역사적 신호에 적용한 별도 진단입니다. 규정 인증을 마친 최신 실전 엔진의 성과로 해석하지 않습니다.

- 전략당 30억원의 투자 원금 상한을 유지하고 이익은 현금으로 남깁니다. 손실로 현금이 줄면 외부 자금으로 보충하지 않습니다.
- 세 전략 사이의 예산 이동·미사용 현금 대여는 하지 않습니다. 중복 종목은 허용합니다.
- 동일 종목·동일 거래일의 **매수와 매도 금액 합계**가 `min(직전 20일 평균 거래대금 10%, 당일 거래대금 10%)`를 넘지 않게 합니다.
- 매도 먼저, 같은 방향의 주문은 요청금액에 비례해 배분합니다. 매수는 최대 10거래일, 청산은 전량 체결될 때까지 이어갑니다.
- 비용은 매수·매도 각각 25bp. 대기 현금 이자는 0, 시장 베타 헤지는 없습니다.
- 샤프는 전체 거래일의 계좌 일별 수익률로 계산하며 비교 일관성을 위해 무위험수익률은 0입니다.

**통합 수익률은 일별 원화 손익을 합쳐 통합 계좌에서 다시 계산합니다. 개별 샤프·MDD·CAGR를 평균내지 않습니다.** CAGR는 고정 투자원금 운용 계좌의 시작·종료 자산을 연율화한 요약치로, 수익을 매회 재투자했다는 뜻이 아닙니다.''')
    code('''p=perf.copy();p['방식']=p.scenario.map(labels)
p['순이익(억원)']=p.profit_krw/1e8;p['종료자산(억원)']=p.ending_equity_krw/1e8
cols=['방식','index_id','순이익(억원)','종료자산(억원)','cagr','volatility','sharpe','mdd','position_days']
table(p[cols],['cagr','volatility','mdd'],['sharpe','순이익(억원)','종료자산(억원)'])
''')
    md('''## 3. NAV와 손실 경로

각 전략은 30억원, 통합은 90억원을 1로 정규화했습니다. **매매에 투입하는 원금은 고정**이며 누적 손익으로 계좌 NAV가 움직입니다. 현금으로 대기한 날짜도 포함합니다.

통합으로 위험이 분산되더라도 세 전략은 한국 주식 롱온리이고 정기변경 일정도 겹칩니다. 지수가 다르다는 이유만으로 독립적인 세 전략이나 무손실 매매로 취급할 수 없습니다.''')
    code("display(Image(filename=str(OUT/'NAV_drawdown.png')))")
    code('''corr=[]
for sc in ['sharpe_selected','f1_selected']:
    print(labels[sc], '일별 계좌수익률 상관계수')
    display(pd.read_csv(OUT/f'{sc}_return_correlation.csv',index_col=0).round(3))
display(pd.read_csv(OUT/'saved_ledger_validation.csv'))
''')
    md('''## 4. 언제 포지션이 있었나

아래 비중은 **최초 통합 원금 90억원 대비 주식 평가금액**입니다. 보유 중 주가 상승으로 100%를 넘을 수 있으므로 현재 NAV 대비 레버리지 비율과 혼동하지 않습니다. 전략당 신규 투자 예산은 30억원을 초과하지 않습니다.''')
    code("display(Image(filename=str(OUT/'exposure.png')))")
    code('''daily=pd.read_parquet(OUT/'daily.parquet')
q=daily[daily.index_id.eq('COMBINED')].copy();q['연도']=q.date.dt.year
year=q.groupby(['scenario','연도']).agg(보유거래일=('any_position','sum'),전체거래일=('date','size')).reset_index()
year['보유일비율']=year.보유거래일/year.전체거래일;year['방식']=year.scenario.map(labels)
table(year[['방식','연도','보유거래일','전체거래일','보유일비율']],['보유일비율'])
annual=pd.read_csv(OUT/'annual.csv');annual=annual[annual.index_id.eq('COMBINED')].copy()
annual['방식']=annual.scenario.map(labels);annual['손익(억원)']=annual.profit_krw/1e8
table(annual[['방식','date','손익(억원)','return_on_start_equity','return_on_fixed_principal','partial_2026']],['return_on_start_equity','return_on_fixed_principal'],['손익(억원)'])
''')
    md('''2026년은 6월 30일까지의 부분기간입니다. `return_on_start_equity`는 연초 계좌 자산 대비 수익률, `return_on_fixed_principal`은 최초 90억원 대비 해당 연도 손익입니다. 서로 다른 분모이므로 구분합니다.

## 5. 공유 유동성 제약과 검산

단독 계좌들을 단순히 더한 경우 같은 종목의 거래대금 한도가 중복 사용될 수 있습니다. 아래에서는 그 중복을 찾고, 공동 한도를 적용한 통합 성과와 비교합니다. 각 단독 계좌의 기존 일별 평가금액·성과 지표 재현, 공동 주문 한도, 현금 비음수, 주식평가+현금=총자산, 월별→전체 손익 집계를 확인했습니다.''')
    code('''ind=pd.read_csv(OUT/'independent_reference.csv');ind=ind[ind.index_id.eq('COMBINED')]
shared=perf[perf.index_id.eq('COMBINED')]
compare=ind.merge(shared,on=['scenario','index_id'],suffixes=('_단순합산','_공유한도'))
compare['공유한도_손익변화(억원)']=(compare.profit_krw_공유한도-compare.profit_krw_단순합산)/1e8
table(compare[['scenario','sharpe_단순합산','sharpe_공유한도','공유한도_손익변화(억원)']],decimals=['sharpe_단순합산','sharpe_공유한도','공유한도_손익변화(억원)'])
audit=pd.read_csv(OUT/'independent_liquidity_audit.csv')
display(audit.assign(한도초과=lambda x:x.excess.gt(.01)).groupby('scenario').agg(독립계좌합산_초과종목일=('한도초과','sum'),중복한도금액합=('excess','sum')))
display(pd.read_csv(OUT/'concentration.csv'))
display(json.loads((OUT/'validation.json').read_text(encoding='utf8')))
''')
    md('''## 6. 해석의 범위와 재현

F1 선택은 편입 예측의 적중·누락을 기준으로 고른 모델이고, 샤프 선택은 비용 반영 운용성과를 기준으로 고른 모델입니다. 한쪽의 IS 수치와 다른 쪽의 OOS 수치를 모델 업데이트 전후처럼 비교하지 않습니다. 이번 통합안의 더 좋은 OOS 결과를 보고 새 모델·비중을 선택한 것은 아닙니다.

KRX300 재무 데이터가 도착하면 공개 적격성 규칙부터 반영하고 **동일한 예산·비용·분할 기간으로 전후 비교**해야 합니다. 현재 결과를 신규 규정 완성 후 성과로 인용하지 않습니다. 시장충격·호가 대기열·실제 주문 체결·배당 및 기업행위 원장의 추가 검증은 별도입니다.

결과: `04_results/index_three_portfolio_20261117/`  
재계산: `python 02_code/modules/index_three_portfolio_20261117/run.py`  
노트북 재생성: `python 02_code/modules/index_three_portfolio_20261117/publish.py`''')
    nb.cells=cells;nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    path=ROOT/'03_notebooks/지수편입예측_3전략통합_90억원_OOS_20261117.ipynb'
    nbformat.write(nb,path)
    # Execute every cell in the current Python process. This does not require
    # reading the inaccessible per-user Jupyter kernel registry or launching a UI.
    shell=InteractiveShell.instance()
    execution_count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        execution_count+=1
        with capture_output() as captured:
            exec(compile(cell.source,f'<portfolio-notebook-cell-{execution_count}>','exec'),shell.user_ns)
        cell.execution_count=execution_count;cell.outputs=[]
        for name,text in [('stdout',captured.stdout),('stderr',captured.stderr)]:
            if text:cell.outputs.append(nbformat.v4.new_output('stream',name=name,text=text))
        for out in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
    nbformat.write(nb,path)
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    (OUT/'notebook_validation.json').write_text(json.dumps(dict(executed=True,execution_method='in-process IPython; every code cell executed',cells=len(nb.cells),code_cells=execution_count,errors=len(errors),notebook=str(path)),ensure_ascii=False,indent=2),encoding='utf8')
    print(path,flush=True)


if __name__=='__main__':main()
