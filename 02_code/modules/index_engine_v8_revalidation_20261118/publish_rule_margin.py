"""Executed, self-contained notebook for the rule-distance ablation."""
from context import ROOT,OUT
import os,json
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# 지수 편입 예측: 소수 규정 변수의 추가 효과 검증
2026-11-18 · KOSPI200 / KOSPI100 / KRX300 · 롱온리 · 기존 V8와 동일 입력

## 1. 무엇을 확인하는 실험인가?
복잡한 모형을 추가하지 않고, **규정의 경계와 얼마나 떨어져 있는지** 네 가지 변수로 표현한다.
기존 로지스틱에 추가한 효과와, 소수 변수만 사용하는 로지스틱의 성과를 분리한다. 결과가 악화된 경우도 모두 공개한다.

원래 규칙 엔진, 후보 유니버스, 기업행위 처리, 선정 수, 진입·청산일은 바꾸지 않는다. 이번에 새로 관측한 미래 OOS가 아니라 반복 확인해 온 역사적 OOS의 추가 연구다. 기존 모델 선택 파일이나 실전 주문 설정은 변경하지 않는다.

## 2. 네 가지 규정 변수
|변수|계산|주의점|
|---|---|---|
|업종 시총 경계 여유|해당 시기 누적 시총 기준 − 본 종목보다 앞선 종목들의 누적 시총 비중|한국200의 과거 70/80/85% 프로파일 유지. 유동성 차순위 대체·제조업 예외는 이 변수 하나에 담기지 않는다.|
|유동성 순위 여유|해당 기준 비율 − 심사기간 거래대금 순위/업종 종목 수|주문 체결의 ADV20 제약과는 별도다.|
|진입 경계 시총 차이|log(본 종목 예상 평균 시총 / 경계 종목 예상 평균 시총)|200은 업종 기존 수×90% 순위,100은 예상200 중80위,300은 업종 종목 수×30% 순위 기준. 정수 경계는 내림.|
|전체 규모 경계 차이|log(본 종목 예상 평균 시총 / 규모 기준 종목 시총)|200·300은 적격 유니버스200·300위,100은 예상200 중100위. 최종 보충 명단을 완벽히 재현한 문턱이 아니라 설명변수다.|

예: 경계 종목 시총이 1조원이고 후보가 1.1조원이면 로그 시총 차이는 log(1.1)=0.0953이다. 같은 한 계단 차이도 시총 금액 차이가 크면 확률이 다를 수 있다.
KOSPI100의80/120 완충 기준은 기존 연구 대체 규칙이며, KRX300의BMI와 최신TMI 연결 인증 공백도 유지된다. 변수 추가로 이 자료 공백이 해결됐다고 주장하지 않는다. 결측은 학습 IS 중앙값과 결측 표시자로 처리하며 OOS 자료로 대체하지 않는다.

## 3. 비교 설계
진입일은 기존 IS 선택값인 KOSPI20040일·KOSPI10040일·KRX30080일을 고정한다. 각 회차의 기본 엔진 선정 수와 같은 수를 각 모형이 선택한다. 모든 포지션은 동일 비중이며 30억원의 고정 예산, 편도25bp(왕복50bp), 최대10거래일 매수, 직전ADV20·당일 거래대금 각10% 중 작은 한도와 청산 완료까지 동일 한도를 적용한다. 수익 재투자는 없다. 종목이 바뀌면 체결률이 달라질 수 있으며 그 차이도 성과에 포함된다.

모든 로지스틱은 기존과 같은 C=0.1과 표준화·중앙값 결측대체를 쓴다. 같은 회차의 여러 종목·예측시점은 학습 회차 가중치 합이 같도록 한다. 최소4개 완료 회차 이후 순차 검증하며, OOS 학습 정답은2022년까지로 고정한다. 같은 회차의 미래 정답은 학습하지 않는다.
IS F1→정밀도→적은 변수 순으로 사전 정의한 후보 중 선택하며, OOS 수익으로 선택을 바꾸지 않는다. 이번 실험은 진입일·C·확률 문턱의 추가 탐색이 아니다.
''')
code('''from pathlib import Path
import json,pandas as pd,numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'04_results').exists())
R=ROOT/'04_results/index_engine_v8_revalidation_20261118/rule_margin_audit'
REPORT=ROOT/'05_reports/index_engine_v8_revalidation_20261118/rule_margin';REPORT.mkdir(parents=True,exist_ok=True)
read=lambda n:pd.read_csv(R/(n+'.csv'))
perf=read('performance');acc=read('accuracy');pairs=read('paired_improvement')
NAMES={'baseline':'기본 규칙','legacy':'기존 로지스틱31변수','legacy_no_rank':'기존−순위보정 표시자','core':'핵심3변수','compact_rules':'핵심3+규정4','compact_rank':'핵심3+규정4+순위보정','legacy_rules':'기존31+규정4'}
for x in ['coverage_margin','liquidity_margin','entry_cap_gap','size_cap_gap']:NAMES['compact_without_'+x]='소형−'+x
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':130})
summary=perf.merge(acc,on=['index_id','model','phase'],suffixes=('','_accuracy'))
summary['방식']=summary.model.map(NAMES)
protocol=json.loads((R/'protocol.json').read_text(encoding='utf8'))
display(acc[acc.model.eq('baseline')][['index_id','phase','n_events','first_event','last_event']])
display(pd.DataFrame([{'방식':NAMES[m],'원변수 수':len(c),'입력 변수':', '.join(c)} for m,c in protocol['features'].items()]))
''')
md('''## 4. IS에서 무엇을 선택했나?
기본 엔진과 변수 제거 실험은 기준·진단용이다. 선택 후보는 기존31변수, 핵심3변수, 핵심3+규정4, 여기에 순위보정 표시자 추가, 기존31+규정4의 다섯 가지다. 아래 선택은 OOS 예측을 계산하기 전에 저장했다. IS 예측 지표의 선택과 IS 샤프 최대 선택은 서로 다르며 이번에는 예측 지표를 사용했다.
''')
code("freeze=json.loads((R/'IS_selection_freeze.json').read_text(encoding='utf8'));display(pd.DataFrame(freeze['choices']))")
md('''## 5. 예측력과 비용 차감 성과
정밀도=예측한 신규 편입 중 적중 비율, 재현율=실제 신규 편입 중 포착 비율. 현재 구성종목을 맞혔다고 적중에 넣지 않는다. 비교 모형의 선정 수·정답 수가 같으므로 같은 지수에서 적중 수가 늘면 정밀도·재현율은 함께 좋아진다.
회차 순수익은 손익/고정30억원이다. 샤프는 대기 현금을 포함한 전체 기간 일별 계좌 수익률, RF=0, MDD는 회차 평균이 아닌 계좌의 고점 대비 최대 하락률이다. CAGR은 고정 예산 손익을 쌓은 계좌 자산의 시작·끝 값을 연율화한 것으로, 주문원금을 복리로 늘렸다는 뜻이 아니다.
''')
code('''main=['baseline','legacy','core','compact_rules','compact_rank','legacy_rules']
for phase in ['IS_forward','OOS']:
 print(phase)
 for idx in ['KOSPI200','KOSPI100','KRX300']:
  print(idx)
  q=summary[summary.phase.eq(phase)&summary.index_id.eq(idx)&summary.model.isin(main)].copy()
  q['model']=pd.Categorical(q.model,main,ordered=True);q=q.sort_values('model')
  display(q[['방식','n_events','hits','predicted','actual','precision','recall','f1','mean_event_net','cagr','volatility','sharpe','mdd','win_rate','payoff']].style.format({c:'{:.2%}' for c in ['precision','recall','f1','mean_event_net','cagr','volatility','mdd','win_rate']}|{'sharpe':'{:.3f}','payoff':'{:.2f}'}).hide(axis='index'))
''')
md('''## 6. 추가된 네 변수가 정말 개선했나?
주 비교는 기존31+규정4 대 기존31, 핵심3+규정4 대 핵심3이다. 규정변수만 하나씩 제거한 실험도 공개한다.
동일 회차 손익 차이를 연도별로 묶은 표준오차와 연도 단위 부호반전 검정을 사용한다. OOS7회·4년으로 검정력이 낮고,4개 연도 부호반전 양측 검정의 최소 p값은0.125다. 따라서 이 검정만으로5% 유의성을 달성할 수 없다. 여섯 주 비교의 연도군집 p에는 Holm 보정을 함께 표시한다. 나머지는 탐색적 비교다.
수익이 가장 많이 개선된 한 회차를 제외한 평균 차이도 본다. 이를 사후 매매 규칙으로 사용하지 않는다.
''')
code('''primary=pairs[pairs.phase.eq('OOS') & (((pairs.model=='legacy_rules')&(pairs.reference=='legacy'))|((pairs.model=='compact_rules')&(pairs.reference=='core')))].copy()
from statsmodels.stats.multitest import multipletests
primary['holm_p']=multipletests(primary.cluster_p.fillna(1),method='holm')[1]
display(primary[['index_id','model','reference','mean_difference','better_events','worse_events','cluster_t','cluster_p','holm_p','year_signflip_p','leave_best_event_out_mean']].style.format({'mean_difference':'{:+.2%}','leave_best_event_out_mean':'{:+.2%}','cluster_t':'{:.3f}','cluster_p':'{:.3f}','holm_p':'{:.3f}','year_signflip_p':'{:.3f}'}).hide(axis='index'))
display(pairs[pairs.phase.eq('OOS')])
primary.to_csv(R/'primary_holm.csv',index=False,encoding='utf-8-sig')
''')
md('''## 7. OOS 계좌 경로와 회차별 개선분
곡선은 시작 자산1에서 고정 예산 손익을 누적한 계좌 경로다. 미투입 금액·실현 이익은 현금이다. 단일 지수의 모든 비교 모형에 동일 기간을 적용한다.
''')
code('''daily=pd.read_parquet(R/'daily.parquet');events=pd.read_parquet(R/'events.parquet')
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
for ax,idx in zip(axes,['KOSPI200','KOSPI100','KRX300']):
 for m in ['baseline','legacy','compact_rules','legacy_rules']:
  q=daily[daily.index_id.eq(idx)&daily.phase.eq('OOS')&daily.model.eq(m)].sort_values('date')
  ax.plot(q.date,q.equity/3e9,label=NAMES[m],lw=1.5)
 ax.set_title(idx);ax.set_ylabel('계좌 자산 / 최초30억원');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(REPORT/'OOS_NAV.png');plt.show()
for idx,q in events[events.phase.eq('OOS')].groupby('index_id'):
 w=q.pivot(index='event',columns='model',values='net_return')
 z=pd.DataFrame({'기존+규정 − 기존':w.legacy_rules-w.legacy,'핵심+규정 − 핵심':w.compact_rules-w.core,'소형규정 − 기존':w.compact_rules-w.legacy})
 print(idx);display(z.style.format('{:+.2%}'))
''')
md('''## 8. 확률 품질·변수 제거·계수
Brier와Log loss는 낮을수록 좋다. 후보가 많은 회차 하나가 압도하지 않게 회차별 평균 후 다시 평균했다. 후보 대다수가 비편입이므로 낮은 Brier만으로 매수 후보 선정 성능이 좋다고 볼 수 없다.
`legacy_no_rank`는 rank_selected 표시자 하나를 제거한 결과다. 후보 유니버스 전체에서 순위 보정의 영향을 제거한 실험은 아니다. 표준화된 회귀계수는 인과효과나 주가 상승률을 의미하지 않는다.
''')
code("quality=read('probability_quality');coef=read('coefficients');display(quality[quality.phase.eq('OOS')]);display(summary[summary.phase.eq('OOS')][['index_id','방식','precision','recall','mean_event_net','sharpe','mdd']]);display(coef[coef.model.eq('compact_rules')])")
md('''## 이번 결과의 판단
**네 변수를 기존 로지스틱에 추가**하면 KOSPI100·200은 OOS 선정과 손익에 추가 개선이 없다. KRX300은 적중134→138건, 정밀도65.37→67.32%, 재현율67.34→69.35%, 평균 회차 순수익13.28→14.83%, 샤프0.996→1.084로 높아졌다. 반면 MDD는−20.06→−22.26%로 악화됐다.

KRX300 추가 개선은7회 중2회에서 양수이고4회에서 음수,1회 동일이다. 2025년12월의 개선분이 특히 크다. 이 최고 개선 회차를 제외하면 나머지 평균 개선은−0.33%p이며, 전체 개선분의 연도군집 t=0.862,p=0.452다. 안정적 우월성 증명이나 실전 기본 모형 교체 근거로 단정하지 않는다.

**변수를 줄인7변수 로지스틱**은 KOSPI100에서 기존31변수와 동일한 OOS 성과를 냈다. KOSPI200은 평균 회차 순수익11.06→11.85%, 샤프0.693→0.726이지만 차이는2025년6월 한 회차다. KRX300은13.28→13.86%, 샤프0.996→1.022이나 MDD는−20.06→−23.80%로 악화됐다. 단순화 가능성은 있으나 모든 지표가 개선된 것은 아니다.

IS 예측력 기준으로 미리 선택된 모형은 KOSPI100·200 모두3변수, KRX300은8변수다. **사후 OOS에서 가장 좋아 보이는35변수 KRX300을 IS 선택 모형으로 바꾸어 표기하지 않는다.** IS 성과와 예측력의 기준을 분리해 연구 후보로 보존하고 새 전진 회차에서 확인해야 한다.
''')
md('''## 9. KRX300 IS 샤프0.263과 과거0.8 부근의 대조
이 표의 최신35변수 로지스틱은 **기본 엔진과 같은 종목 수를 사는 분류모형 실험**이다. 과거 수익 중심의 `attractiveness` 전략과 동일하지 않다. 과거0.863(3억원)·0.880(30억원)은2023~2026년OOS였고, 과거 매력도 전략의2018~2022년9회IS는0.748이었다.

과거 매력도 전략은 보정 확률60% 이상, 업종 대비20일 상승률30% 이하, 과거 적중/실패 조건부 업종초과수익을 이용한 비용 차감 기대수익 양수 조건으로 종목을 줄였다. 현재 분류모형은 그 조건 대신 점수 상위 종목을 기본 선정 수만큼 매수한다.

동일IS5회(2020년12월~2022년12월)·30억원으로 과거 매력도 명단을 재집행하면 샤프1.295, 회차 평균13.84%다. 다만 과거69건 중6건은 현재 자본잠식 기준으로 제외되므로 그 명단을 그대로 실전 복원할 수 없다. 해당6건을 제외하고 기존 배정액을 현금으로 두면 샤프1.150·회차 평균11.42%, 남은 종목에 동일 비중으로 재배분하면1.157·12.48%다. 최신35변수의0.263·1.89%와 구분한다.

이는 **기존 전략 자체가0.8에서0.263으로 붕괴했다는 뜻이 아니다.** 선정·매매 정책과 자료 버전이 바뀐 비교였다. 반대로 과거 매력도 전략도 이5회만으로 새로 선택하거나 전체기간IS성과를1.295로 대체 표기하면 안 된다. 초기4회 제외와 현재 적격성 필터 적용은 원인 대조를 위한 사후 민감도 분석이다. 이전 확률·명단을 유지했으므로 최신 자료로 전부 재학습한 결과도 아니다.
''')
code('''reconciliation=R/'is_sharpe_reconciliation'
if reconciliation.exists():
 display(pd.read_csv(reconciliation/'bridge.csv')[['scenario','n_events','budget_krw','mean_event_net','sharpe','mdd']])
 e=pd.read_csv(reconciliation/'events.csv')
 display(e[e.scenario.isin(['old_5events_30eok','current_35features_5events_30eok'])].pivot(index='event',columns='scenario',values='net_return').style.format('{:+.2%}'))
''')
md('''## 10. 감사와 해석 범위
기존 기본·31변수 로지스틱의12개 IS/OOS 결과를 그대로 재현했는지, 주문수수료·유동성·현금·손익 원장 일치, 모든 학습 정답의 시점과2022년 상한을 확인한다. 입력 해시와 회차별 원장·예측 확률·변수 정의·IS 선택 기록을 결과 폴더에 보존한다.

새 후보의 공식 적격성 인증, 증권사와 동일 시점 비교, 새로운 전진 OOS는 이번 실험 범위가 아니다. 업종/섹터 비교군도 바꾸지 않았다. 결과가 좋더라도 이 실험으로 실전 인증이 완료되는 것은 아니다.

## 11. 참고 및 재현
- Franz(2020), Forecasting index changes in the German DAX family: https://doi.org/10.1057/s41260-020-00153-6 . 공식 순위 기준 보정과 우리의 자체 순위 근사는 구분한다.
- Fernandes·Mergulhão(2016), Anticipatory effects in the FTSE100 index revisions: https://doi.org/10.1016/j.jempfin.2016.02.009 . 선정 경계와 확률 추정의 발상 참고. 한국 규정 변수·로지스틱은 독립 응용이며 원문 전체 재현이 아니다.
- 실행: `python 02_code/modules/index_engine_v8_revalidation_20261118/rule_margin_audit.py` 이후 `publish_rule_margin.py`.
''')
code("checks=json.loads((R/'checks.json').read_text(encoding='utf8'));display(pd.Series(checks).to_frame('통과'));assert all(checks.values())")
nb=nbf.v4.new_notebook(cells=cells);nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
runtime=OUT/'runtime';kernel=runtime/'kernels/python3'
os.environ['JUPYTER_DATA_DIR']=str(runtime);os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(runtime/'ipython')
km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
NotebookClient(nb,timeout=240,km=km,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.validate(nb)
path=ROOT/'03_notebooks/지수편입예측_규정변수_로지스틱_추가효과검증_20261118.ipynb';nbf.write(nb,path)
ex=HTMLExporter();ex.exclude_input=True;html,_=ex.from_notebook_node(nb)
report=ROOT/'05_reports/index_engine_v8_revalidation_20261118/rule_margin';report.mkdir(parents=True,exist_ok=True)
(report/'규정변수_추가효과검증.html').write_text(html,encoding='utf8')
print(path)
