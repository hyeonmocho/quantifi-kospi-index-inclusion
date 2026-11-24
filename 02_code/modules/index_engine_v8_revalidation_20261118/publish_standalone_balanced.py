"""Publish the standalone-IS balanced selection without selecting on OOS."""
from research import ROOT, OUT
import os, json
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md('''# 세 지수 편입 예측: 단독 IS에서 예측력과 성과를 함께 고려한 선정
2026-11-18 · KOSPI200 / KOSPI100 / KRX300 · 각 30억원 · 롱온리

## 1. 이번에 정한 기준
**예측력을 최대로 높이는 모형과 수익을 최대로 높이는 모형이 같을 필요는 없다. 각 지수에서 둘 다 괜찮은 한 가지 조합을 고른다.**

이를 재현 가능한 규칙으로 다음처럼 표현했다.
1. 각 지수의 **원래 단독 IS 종료연도**를 유지한다: KOSPI200·100은 2016년, KRX300은 2022년.
2. 같은 진입시점의 기본 규칙 엔진보다 IS F1이 떨어지지 않고, 비용 차감 후 회차 평균 순수익이 양수인 조합만 남긴다.
3. 이 중 **IS 일별 계좌 샤프**가 가장 높은 조합을 선택한다. 동률이면 F1, 적은 설명변수, 짧은 진입 리드 순으로 결정한다.
4. 모형·진입시점·집행 방식을 저장한 뒤 해당 전략의 OOS에 그대로 적용한다. OOS 성과가 더 좋은 다른 후보로 교체하지 않는다.

이 기준은 프로젝트의 절충 규칙이며, 학술적으로 유일한 최적 기준이 아니다. 기본 엔진도 후보에 포함한다. 복잡한 모형의 추가 효과가 없다면 단순한 엔진을 선택한다.
**이번 후보군 안의 선정 결과**다. 과거 모든 실험에서 사용한 후보 확장·분할추적 등의 모든 조합을 재탐색한 전역 최적값은 아니다.
''')
code('''from pathlib import Path
import json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'04_results').exists())
R=ROOT/'04_results/standalone_balanced_selection_20261118'
REPORT=ROOT/'05_reports/standalone_balanced_selection_20261118';REPORT.mkdir(exist_ok=True)
read=lambda n:pd.read_csv(R/(n+'.csv'))
perf=read('performance');selected=read('selected');summary=read('selected_performance')
forecasts=read('forecasts');events=pd.read_parquet(R/'events.parquet');daily=pd.read_parquet(R/'daily.parquet')
protocol=json.loads((R/'protocol.json').read_text(encoding='utf8'))
FAMILY={'baseline':'기본 규칙','ma20':'20일 이동평균 시총','shrunk_trend':'완만한 추세 보정','calibration':'빈도 확률보정','legacy31':'다변수 로지스틱 31변수','compact7':'소수 규정변수 로지스틱 7변수','augmented35':'기존+규정 로지스틱 35변수'}
POLICY={'count':'기본 엔진과 같은 수 선정·동일비중','attractiveness':'확률·기대수익 필터 후 동일비중'}
def label(m):
 f,p=m.split('__');return FAMILY[f]+' / '+POLICY[p]
def styled(x):
 pct=[c for c in ['precision','recall','f1','trade_precision','trade_recall','trade_f1','baseline_F1','mean_event_net','median_event_net','win_rate','cagr','volatility','mdd','total_return'] if c in x]
 return x.style.format({c:'{:.2%}' for c in pct}|{c:'{:.3f}' for c in ['sharpe','year_cluster_t','year_cluster_p','payoff'] if c in x},na_rep='—').hide(axis='index')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':130})
display(selected.assign(엔진=selected.family.map(FAMILY),집행=selected.policy.map(POLICY))[['index_id','엔진','집행','lead','IS_n']])
''')
md('''## 2. 기간과 표본: 달력 기간과 실제 평가 회차를 구분한다
IS를 최근 공통 기간으로 잘라서 바꾸지 않았다. 다만 초기 자료가 없는 회차까지 검증했다고 해석하면 안 된다.

- KOSPI200: 계획 기간 2002~2016년, 이번에 읽은 기존 공통 코호트 파일의 유효 회차는 **11회**다. 2002·2003·2005·2010년은 이 파일에서 제외되어 있다. 과거 다른 표의 12회와 같다고 인용하지 않는다.
- KOSPI100: 2006~2016년 중 기존 공통 **10회**. 2015년은 기존 코호트에서 제외된다.
- KRX300: 2018년 하반기~2022년 **9회**.

학습형 모델은 그 예측일까지 결과가 확정된 과거 최소 4회가 필요하다. 이전에는 **기본 규칙으로 운용하는 준비 구간**을 포함했다. 준비 구간 수익을 로지스틱의 개선 효과라고 주장하지 않는다. IS 내부에서도 해당 회차의 정답을 미리 학습하지 않는다. OOS에서는 각 지수의 IS 종료연도 이후 정답으로 재학습하지 않는다.

OOS는 KOSPI200·100의 경우 2017년부터, KRX300은 2023년부터이며 종료일은 2026-06-30이다. 선택된 진입시점의 예측 자료가 빠지면 그 회차를 현금 수익 0으로 대신하거나 몰래 삭제하지 않고 미완료로 표시한다.
''')
code('''display(pd.DataFrame([{'지수':k,'IS 시작':protocol['IS_start'][k],'IS 종료연도':protocol['standalone_IS_end'][k],'IS 회차':len(v),'실제 회차':', '.join(v)} for k,v in protocol['IS_events'].items()]))
display(summary[[c for c in ['index_id','phase','status','period_start','period_end','n_events','expected_events','warmup_events','return_fallback_events'] if c in summary]])
''')
md('''## 3. 어떤 모형과 매매 방식을 비교했나?
| 모형 | 직관 |
|---|---|
| 기본 규칙 | 관측 시총과 남은 기간 추정 시총으로 심사 평균을 만들고 지수 선정 규칙을 적용한다. |
| 20일 이동평균 시총 | 하루 시총 대신 최근 20일 수준으로 남은 기간을 추정해 일시적 등락을 완화한다. |
| 완만한 추세 보정 | 최근 추세를 그대로 끝까지 연장하지 않고 약하게 반영한다. |
| 빈도 확률보정 | 시나리오에서 편입된 빈도를 과거 적중 결과와 연결하는 1변수 로지스틱이다. |
| 다변수 로지스틱 31변수 | 시나리오 빈도·순위·시총·수급 등 기존 입력들을 함께 사용한다. |
| 소수 규정변수 로지스틱 7변수 | 빈도의 로그오즈·기본 선정 여부·남은 기간과 규정 경계까지의 거리 4개를 사용한다. |
| 기존+규정 로지스틱 35변수 | 기존 31개에 규정 경계 4개를 추가한다. |

진입 시점은 **심사마감일 전 20·40·60·80거래일**이다. 보유기간 자체가 20·40·60·80일이라는 뜻은 아니다. 예측 정보일 이후 정해진 진입일과 기존 청산일을 사용한다.

확률 모형은 두 가지 매매 방식을 비교했다. 첫째는 기본 엔진이 고른 수만큼 상위 후보를 매수한다. 둘째는 **확률 60% 이상·시나리오 빈도 양수·20일 상대 모멘텀 30% 이하·비용 차감 예상 업종 초과수익 양수**를 만족할 때 매수한다. 예상 수익은 완료된 과거 회차의 적중·실패 평균을 사용하고 표본이 적으면 0 쪽으로 줄인다. 이번에 이 필터 문턱들을 새로 최적화하지 않았다.

지수별 44개 모형·집행·진입 조합을 비교했다. 부스팅은 제외했다. 과거 후보 확장 모델 등의 수익률을 이번 결과와 동일 표본·동일 입력의 값처럼 직접 섞지 않는다.
''')
code('''for idx in ['KOSPI200','KOSPI100','KRX300']:
 q=read(idx+'_IS_candidates').sort_values(['eligible_balanced','sharpe','f1'],ascending=[False,False,False]).copy()
 q['방식']=q.model.map(label)
 print(idx+' — IS 상위 후보와 균형 기준 통과 여부')
 display(styled(q[['방식','lead','eligible_balanced','baseline_F1','precision','recall','f1','mean_event_net','sharpe','mdd']].head(10)))
''')
md('''## 4. 선정한 엔진의 예측력
정밀도는 **예측한 신규 편입 중 실제 신규 편입 비율**, 재현율은 **실제 신규 편입 중 포착한 비율**, F1은 둘의 조화평균이다. 기존 구성종목을 맞힌 건수를 적중에 넣지 않는다.

여기서 예측 명단과 매매 명단은 구분된다. 수익성 필터를 쓰면 실제 거래 종목의 적중률은 좋아도 놓친 편입이 늘 수 있다. 따라서 전체 엔진의 정밀도·재현율과 거래 명단의 정밀도·재현율을 함께 공개한다. F1을 넘었다는 것만으로 증권사보다 우수하다고 주장할 수는 없다.
''')
code('''valid=summary[summary.status.eq('valid')].copy();valid['방식']=valid.model.map(label)
display(styled(valid[['index_id','phase','방식','lead','n_events','hits','predicted','actual','precision','recall','f1','trade_targets','trade_precision','trade_recall','trade_f1']]))
sf=forecasts.merge(selected[['index_id','model','lead']],on=['index_id','model','lead'])
rows=[]
for (idx,phase,warm),g in sf.groupby(['index_id','phase','warmup']):
 h,p,a=g[['hits','predicted','actual']].sum()
 rows.append(dict(index_id=idx,phase=phase,区分='기본 규칙 준비 구간' if warm else '선정 모형 구간',n_events=len(g),precision=h/p if p else np.nan,recall=h/a if a else np.nan,f1=2*h/(p+a) if p+a else np.nan))
display(styled(pd.DataFrame(rows).rename(columns={'区分':'구분'})))
''')
md('''## 5. 각 30억원 단독 계좌의 성과
각 전략의 초기·회차 주문 예산은 **30억원으로 고정**한다. 수익은 현금으로 남기고 매매 예산을 수익만큼 늘리지 않는다. 손실이 나면 가용 현금이 줄어 주문액이 줄 수 있다.

- 매수·매도 각각 25bp 비용. 거래대금 한도는 직전 20일 평균 거래대금의 10%와 당일 거래대금 10% 중 작은 값이다.
- 매수는 최대 10거래일 동안 한도 안에서 진행한다. 무조건 10등분하는 것이 아니다. 매도는 완료될 때까지 한도를 적용한다.
- 평균 순수익은 **회차 손익/고정 30억원**의 평균이다. 연평균 수익률과 다르다. 거래하지 않은 유효 회차의 수익 0도 포함한다.
- 샤프는 대기 현금 기간을 포함한 일별 총계좌 수익률, 무위험수익률 0 기준이다. MDD는 전체 계좌 최대낙폭이다.
- CAGR은 시작·끝 계좌 자산의 연율 환산치다. 주문 예산을 복리 재투자했다는 뜻은 아니다.
''')
code('''display(styled(valid[['index_id','phase','n_events','mean_event_net','total_return','cagr','volatility','sharpe','mdd','win_rate','payoff','year_cluster_t','year_cluster_p']]))
sd=daily.merge(selected[['index_id','model','lead']],on=['index_id','model','lead'])
se=events.merge(selected[['index_id','model','lead']],on=['index_id','model','lead'])
fig,axes=plt.subplots(3,2,figsize=(13,10))
for i,idx in enumerate(['KOSPI200','KOSPI100','KRX300']):
 for j,phase in enumerate(['IS','OOS']):
  ax=axes[i,j];q=sd[sd.index_id.eq(idx)&sd.phase.eq(phase)].sort_values('date')
  if len(q):ax.plot(q.date,q.equity/3e9,color=['#176b87','#b86619'][j],lw=1.6)
  else:ax.text(.5,.5,'완전한 예측·집행 경로 미확보',ha='center',transform=ax.transAxes)
  ax.set_title(idx+' / '+phase);ax.axhline(1,c='gray',lw=.6);ax.grid(alpha=.2);ax.set_ylabel('총자산 / 초기 30억원')
fig.suptitle('선정 모형의 단독 계좌 — 회차 주문 예산 고정, 수익 재투자 없음',fontsize=14)
fig.tight_layout();fig.savefig(REPORT/'selected_NAV.png');plt.show()
display(styled(se[['index_id','phase','event','net_return','entry_notional','actual_exit']].rename(columns={'net_return':'mean_event_net'})))
''')
md('''## 6. 복잡한 모형이 기본 엔진을 얼마나 개선했나?
아래는 **선정 모형과 같은 진입일**의 기본 엔진을 IS에서 비교한 차이다. 모형을 복잡하게 만들었더라도 차이가 0이면 이번 매매 명단·성과에서는 추가 효과가 없다는 뜻이다. 전체 기간이 같은 경우만 비교하며 서로 다른 지수의 샤프 차이를 엔진 우열로 해석하지 않는다.
''')
code('''rows=[]
for s in selected.itertuples():
 a=perf[perf.index_id.eq(s.index_id)&perf.phase.eq('IS')&perf.lead.eq(s.lead)]
 b=a[a.model.eq('baseline__count')].iloc[0];w=a[a.model.eq(s.model)].iloc[0]
 rows.append({'지수':s.index_id,'선정':label(s.model),'진입 리드':s.lead,'기본 F1':b.f1,'선정 F1':w.f1,'F1 차이(%p)':100*(w.f1-b.f1),'기본 샤프':b.sharpe,'선정 샤프':w.sharpe,'샤프 차이':w.sharpe-b.sharpe,'회차 평균 순수익 차이(%p)':100*(w.mean_event_net-b.mean_event_net)})
comparison=pd.DataFrame(rows);display(comparison.round(4));comparison.to_csv(R/'selected_vs_baseline_IS.csv',index=False,encoding='utf-8-sig')
''')
md('''## 7. 해석 범위와 발표에서의 표현
**발표용 연구 모형 선정**과 **실전 인증 완료**는 다르다. 세 가지 한계는 유지한다.

1. **반복 확인한 OOS**: 이번 선택 코드는 IS만 사용하지만 같은 과거 OOS를 이전 연구에서 여러 번 보았다. 결과는 역사적 검증이며 새 전진표본으로 독립 확인한 성과가 아니다. IS에서 44개를 비교한 선택 편향도 있다. 표시한 t·p는 회차 수익에 연도 군집 표준오차를 적용한 진단값이며 후보 탐색 전체를 보정한 확증 검정이 아니다.
2. **자료·규정 공백**: 초기 KOSPI100 규정, KRX300의 과거 BMI·최신 TMI 전체 적격 명단, 일부 당시 공식 적격성 인증에는 미완료가 남아 있다. 현재 자료에서의 연구 성능을 공식 규정 완전 재현으로 발표하지 않는다.
3. **실제 운용 확인**: 일별 거래대금 기반 체결 모형이 실제 주문·대차·호가 원장은 아니다. 이번 세 전략은 롱온리이며 실전 주문 설정은 변경하지 않았다. 세 계좌를 합치면 같은 종목의 거래대금 한도를 공유해야 하므로 단독 수익의 단순 평균으로 통합 성과를 만들지 않는다.

증권사 대비 비교는 동일 시점·동일 회차의 별도 비교 결과가 필요하다. 이 선정 실험만으로 증권사보다 우수하다는 결론을 내리지 않는다.

## 8. 재현과 검증
연구 계산: `02_code/modules/index_engine_v8_revalidation_20261118/select_standalone_balanced.py`

노트북 작성: 같은 폴더의 `publish_standalone_balanced.py`

선정 저장: `04_results/standalone_balanced_selection_20261118/IS_selection_freeze.json`

예측 명단·확률·종목 주문·회차 손익·일별 계좌를 결과 폴더에 함께 저장했다. 아래 감사 항목은 계산 일관성 확인이며 실제 시장에서의 수익 보장이 아니다.
''')
code('''checks=json.loads((R/'checks.json').read_text(encoding='utf8'));display(pd.Series(checks,name='통과').to_frame());assert all(checks.values())
try:
 missing=read('missing')
 display(missing if len(missing) else pd.DataFrame({'결과':['예측 누락 없음']}))
except pd.errors.EmptyDataError:print('예측 누락 없음')
''')

nb=nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
runtime=OUT/'runtime';kernel=runtime/'kernels/python3'
os.environ['JUPYTER_DATA_DIR']=str(runtime);os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(runtime/'ipython')
km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
NotebookClient(nb,timeout=240,km=km,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.validate(nb)
path=ROOT/'03_notebooks/세지수_단독IS_예측력과성과_균형선정_20261118.ipynb';nbf.write(nb,path)
export=HTMLExporter();export.exclude_input=True;html,_=export.from_notebook_node(nb)
report=ROOT/'05_reports/standalone_balanced_selection_20261118';report.mkdir(exist_ok=True)
(report/'균형선정_결과.html').write_text(html,encoding='utf8')
print(path)
