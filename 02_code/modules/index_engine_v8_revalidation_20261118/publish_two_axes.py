"""Update the user's presentation notebook, preserving its previous version."""
from research import ROOT,OUT
import json,os,shutil
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

R=ROOT/'04_results/two_axis_selection_20261118'
P=ROOT/'03_notebooks/지수편입예측_3전략_발표용통합노트_20261118.ipynb'
backup=R/'presentation_before_two_axes.ipynb'
if not backup.exists():shutil.copy2(P,backup)
old=nbf.read(backup,4);cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# 지수 편입을 미리 예측하는 세 가지 롱온리 전략
**조현모 · 퀀티파이 · 2026-11-18 · 두 축 독립 선정 / KOSPI100 부분 명단 복원**

이번 발표는 두 질문을 분리한다.

| 축 | 질문 | 단독 IS에서 선택할 기준 | OOS에서 확인할 것 |
|---|---|---|---|
| **1. 예측 엔진** | 실제 신규 편입을 얼마나 잘 찾는가? | F1 최대, 동률이면 정밀도·단순성 | 정밀도·재현율, 같은 시점 증권사 리포트와 비교 |
| **2. 운용 전략** | 비용과 체결 제약 후 위험 대비 얼마나 잘 버는가? | 모든 유효 후보 중 **IS 일별 계좌 샤프 최대** | 고정한 모형의 OOS 샤프·수익률·MDD·승률·손익비·유의성 |

**예측력과 수익성이 하나의 모형으로 수렴할 필요는 없다.** 성과축에는 F1이나 평균 수익률 문턱을 걸지 않고 샤프 자체를 최대화한다. 평균 수익률을 최대화하는 방식이 아니다. OOS 샤프가 좋다는 이유로 모형을 다시 고르지는 않는다.
두 축 모두 KOSPI200·100은 원래 단독 IS 종료연도 2016년, KRX300은 2022년을 유지한다. OOS는 이전에도 본 역사적 구간이며 평가 종료일은 2026-06-30이다.

이 버전은 이전의 한 모형 절충 선정표를 대체한다. 이전 노트북은 결과 폴더에 보관했다. 아래 수치는 이번 두 축의 선택값과 연결되며 과거 다른 통합계좌 성과와 혼합하지 않는다.
''')
code('''from pathlib import Path
import json,pandas as pd,numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'04_results').exists())
R=ROOT/'04_results/two_axis_selection_20261118';REPORT=ROOT/'05_reports/two_axis_selection_20261118';REPORT.mkdir(exist_ok=True)
read=lambda n:pd.read_csv(R/(n+'.csv'))
choices=read('choices');perf=read('performance');fr=read('forecasts');broker=read('broker_summary')
daily=pd.read_parquet(R/'daily.parquet');events=pd.read_parquet(R/'events.parquet')
family={'baseline':'기본 규칙','ma20':'20일 이동평균 시총','shrunk_trend':'완만한 추세 보정','calibration':'시나리오 빈도 확률보정','legacy31':'31변수 로지스틱','compact7':'7변수 규정 로지스틱','augmented35':'35변수 로지스틱'}
axis={'prediction':'① 예측력','performance':'② 운용 성과'}
policy={'count':'선정 후보 동일비중','attractiveness':'확률·기대수익 필터 후 동일비중'}
def pretty(x):
 x=x.copy()
 if 'axis' in x:x['axis']=x.axis.map(axis)
 if 'model' in x:x['model']=x.model.map(lambda s:family.get(s.split('__')[0],s))
 if 'who' in x:x['who']=x.who.map({'engine':'선정 엔진','broker':'증권사'})
 pct=set(['precision','recall','f1','mean_event_net','mean_net','net_return','win_rate','cagr','volatility','mdd','total_return','trade_precision','trade_recall'])&set(x.columns)
 return x.style.format({c:'{:.2%}' for c in pct}|{c:'{:.3f}' for c in ['sharpe','year_cluster_t','year_cluster_p','payoff'] if c in x},na_rep='—').hide(axis='index')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':130})
''')
for start in ['## 1. 전략을 만들게 된 배경','## 2. 지수 편입 전략 소개','## 3. 엔진 관련 논문과 설계']:
 found=next((c for c in old.cells if c.cell_type=='markdown' and c.source.startswith(start)),None)
 if found:cells.append(nbf.v4.new_markdown_cell(found.source))
md('''### 3-3. 두 축의 실제 선택값과 기간
기본 규칙, 이동평균, 완만한 추세, 빈도 확률보정, 7·31·35변수 로지스틱을 비교했다. 확률 모형은 동일 종목수 선정과 기대수익 필터 집행을 비교했다. 진입 시점은 심사마감일 전 20·40·60·80거래일이다. **보유기간이 그 일수라는 뜻은 아니다.**

예측축에서는 거래 필터와 관계없는 원래 예측 명단의 F1을 비교한다. 운용축에서는 필터 사용 여부까지 포함해 지수별 44개 기존 IS 조합 중 최고 샤프를 선택한다. 과거 모든 후보 확장·추적매매 조합을 재탐색한 전역 최적값은 아니다.

| 지수 | IS 달력 기간 | 이번 IS 유효 회차 | OOS 기간 |
|---|---|---:|---|
| KOSPI200 | 2002~2016 | 11회: 2004, 2006~2009, 2011~2016 | 2017~2026년 상반기 |
| KOSPI100 | 2006~2016 | 10회: 2015년 제외 | 2017~2026년 상반기 |
| KRX300 | 2018년 하반기~2022 | 9회 | 2023~2026년 상반기 |

학습형 모델의 초기 4회는 기본 규칙을 사용한 준비 구간이다. IS 중간에도 완료된 과거 회차만 학습하며 OOS 정답으로 재학습하지 않는다. 준비 구간의 성과를 로지스틱의 추가 효과로 설명하지 않는다.
''')
code("display(choices.assign(축=choices.axis.map(axis),엔진=choices.family.map(family),집행=choices.policy.map(policy))[['index_id','축','엔진','집행','lead','IS_n']])")
md('''### 3-4. KOSPI100 2024년 6월 오류는 어떻게 처리했나?
2024-04-01 예측에서 기본 명단은 100개가 만들어졌지만, 순위 보정 및 일부 시나리오에서는 **예상 전체 구성종목 99개**만 남았다. 이전에는 오류로 중단했으나, 이번 연구 버전은 사용자의 지시에 따라 **99개를 그대로 유지**한다.

- 100번째 종목을 임의로 추가하지 않는다. 부족한 시나리오를 버리지 않고 256개 전체를 편입빈도 분모로 사용한다.
- 99개 전체를 사는 전략이 아니다. 기존 구성종목을 제외한 **신규 편입 후보**를 분석·매매한다.
- 전체 회차를 현금으로 대체하지 않았다. 복원된 예측 입력으로 선택된 엔진을 다시 계산했다. 실제 주문이 없는 경우에만 해당 예산이 현금으로 남는다.
- 부분 명단 허용은 명시적으로 켠 연구 분기에만 적용한다. 공식 구성종목 100개를 완전히 재현했다는 뜻은 아니며 기본 엄격 검증 동작은 유지했다.
''')
code('''audit=json.loads((R/'repair_audit.json').read_text(encoding='utf8'))
display(pd.DataFrame(audit['underfill_observations']))
display(fr[fr.index_id.eq('KOSPI100')&fr.event.eq('2024-06')][['axis','model','lead','predicted','hits','actual','trade_count','trade_hits']])
display(events[events.index_id.eq('KOSPI100')&events.event.eq('2024-06')][['axis','event','net_return','entry_notional','actual_exit']])
''')
md('''## 4. 예측축: 편입 예측력과 증권사 비교
정밀도는 예측한 신규 편입 중 적중 비율, 재현율은 실제 신규 편입 중 포착 비율이다. F1은 둘의 조화평균이다. 기존 구성종목을 맞혔다고 적중 건수를 늘리지 않는다.

아래 첫 표는 고정된 심사마감 전 진입시점에서의 예측력이다. 증권사 표는 공정한 비교를 위해 **IS에서 선택한 같은 엔진 종류를 리포트 날짜에 다시 계산**한다. 두 표는 예측 날짜와 표본이 다르므로 동일 성적표처럼 직접 연결하지 않는다.
''')
code('''display(pretty(perf[perf.axis.eq('prediction')][['index_id','phase','model','lead','n_events','hits','predicted','actual','precision','recall','f1']]))
display(pretty(broker))
b=read('broker_pairs');display(b[['index_id','report_id','event','who','information_date','hits','predicted','actual','status']])
''')
md('''### 4-1. 세 지수의 증권사 비교 범위
비교 대상에는 **KOSPI200·KOSPI100·KRX300 모두**를 포함한다. 다만 평가 가능한 범위는 다르다. KOSPI200은 선택된 엔진의 동일 시점 OOS 비교, KRX300은 초기 전체 명단과 최근 부분 명단을 구분한다. KOSPI100은 공표 전 예측 명단을 아직 확보하지 못했다. 실제 변경 결과나 다른 지수의 전망을 예측 리포트로 대체하지 않는다.

KOSPI100은 KRX100, 코스피 대형주 지수와 다른 지수다. 추가 검색에서 나온 규모 이동 예상표, 삼성전자 CAP 조정 분석, 공표 이후 구성명단은 KOSPI100 신규 편입 예측의 정밀도·재현율을 평가할 근거가 아니다.
''')
code("display(read('broker_three_index_coverage'))")
md('''### 4-2. KRX300 전체 명단: 2018~2019년 3개 리포트·2회차
하나금융투자의 2018-10-22, 2019-03-18, 2019-04-22 보고서를 동일 날짜의 엔진과 연결했다. **당시 완료된 학습 회차가 4개보다 적으므로 실제 운용 규칙대로 기본 규칙 준비 모형을 사용한다.** 2022년까지 학습한 로지스틱을 2018년으로 소급 적용하지 않는다. 아래는 선택된 7변수 로지스틱의 독립 OOS 우위를 입증하는 표가 아니다.
''')
code('''broad=read('broker_broad_summary');bp=read('broker_broad_pairs')
display(pretty(broad[broad.scope.eq('full_roster_pretraining')]))
display(pretty(bp[bp.scope.eq('full_roster_pretraining')][['report_id','event','who','engine','information_date','train_events','predicted','hits','actual','precision','recall','f1','status','net_return']]))
''')
md('''### 4-3. KRX300 부분 명단: 2024년 1월 삼성증권
이 리포트는 LG에너지솔루션·에코프로머티 두 종목을 언급하지만 전체 신규 편입 명단은 아니다. 엔진도 같은 날 선택 모형의 확률 상위 **2개**를 뽑아 **동일 개수의 적중률과 매매 결과**를 보조 비교한다. 이는 고정 진입일 전체 포트폴리오와 다른 실험이다. 부분 명단에는 재현율과 F1을 표시하지 않는다.

엔진은 2022년까지 완료된 과거 자료로만 학습한다. 1회차이므로 승률 0%나 100%를 안정적인 특성으로 해석하지 않으며 손실·수익 표본이 한쪽에만 있으면 손익비는 계산하지 않는다.
''')
code('''display(pretty(broad[broad.scope.eq('partial_same_count_OOS')]))
display(pretty(bp[bp.scope.eq('partial_same_count_OOS')][['report_id','who','engine','information_date','train_events','predicted','hits','precision','status','net_return','names']]))
display(bp[['report_id','source_url']].drop_duplicates())
display(pd.Series(json.loads((R/'broker_broad_checks.json').read_text(encoding='utf8')),name='검증 통과').to_frame())
''')
md('''### 4-4. 비교 결과를 읽는 기준

같은 회차에 여러 리포트가 있어 회차별 평균을 먼저 계산했다. 승률·손익비 역시 같은 날짜, 같은 30억원, 같은 비용·거래대금 제약으로 매매한 뒤 회차별 평균 수익으로 계산한다. 리포트 인쇄일을 이용 가능한 날짜의 대용으로 쓰므로 실제 최초 배포 시각은 완전히 인증되지 않았다.

**발표 대본:** “첫 축은 명단을 맞히는 능력입니다. 증권사와 동일 시점으로 맞춰 비교하고, 초기 준비 모형과 최근 학습 모형을 구분했습니다. KOSPI100은 대응 리포트가 없어 우열을 말하지 않습니다. 두 번째 축은 비용과 유동성 제약을 반영한 IS 샤프를 최대화해 별도로 선택합니다.”

## 성과표를 읽는 공통 기준
각 지수의 고정 주문 예산은 30억원이다. 이익은 현금으로 남기고 다음 주문 예산을 늘리지 않는다. 손실이 나면 가용 현금에 따라 실제 주문은 줄어든다.

편도 25bp, 왕복 약 50bp 비용과 직전 ADV20 10%·당일 거래대금 10% 중 작은 한도를 적용한다. 매수는 최대 10거래일, 매도는 청산 완료까지 한도를 적용한다. 회차 평균 순수익은 회차 손익/고정 30억원이다. CAGR은 계좌의 시작·끝 자산을 연율화한 값이며 복리 주문예산을 뜻하지 않는다. 샤프는 현금 대기일 포함 일별 계좌 수익률(RF=0), MDD는 전체 계좌 최대낙폭이다.
''')
for no,idx,desc in [(5,'KOSPI200','업종별 평균 시총과 유동성 조건을 이용해 정기 신규 편입 후보를 예측한다.'),(6,'KOSPI100','KOSPI200 내부에서 상위 대형주로 올라갈 후보를 예측한다. 소수 종목의 성패가 계좌에 크게 작용한다.'),(7,'KRX300','KOSPI·KOSDAQ을 합친 산업군·규모·유동성·재무 적격성을 반영한다. 과거 부모 유니버스 인증 공백은 남아 있다.')]:
 md(f'''## {no}-1. {idx} 단독 전략
{desc}

예측축은 명단을 맞히는 기준으로, 운용축은 수익·위험 기준으로 선택했다. 두 축의 결과를 같은 기간·같은 예산으로 함께 보여준다. 기대수익 필터를 사용하는 모형은 확률 60% 이상, 시나리오 빈도 양수, 20일 상대 모멘텀 30% 이하, 비용 차감 예상 업종 초과수익 양수를 요구한다. 예상 수익은 완료된 과거 IS 회차에서만 추정한다.

### {no}-2. 성과 및 예측력
''')
 code(f'''q=perf[perf.index_id.eq('{idx}')]
display(pretty(q[['axis','phase','model','lead','n_events','precision','recall','f1','mean_event_net','cagr','sharpe','mdd','win_rate','payoff','year_cluster_t','year_cluster_p']]))
fig,axes=plt.subplots(1,2,figsize=(12,3.8))
for ax,phase in zip(axes,['IS','OOS']):
 for key,color in [('prediction','#176b87'),('performance','#b86619')]:
  d=daily[daily.index_id.eq('{idx}')&daily.phase.eq(phase)&daily.axis.eq(key)].sort_values('date')
  ax.plot(d.date,d.equity/3e9,label=axis[key],color=color,lw=1.5)
 ax.set_title('{idx} '+phase);ax.set_ylabel('총자산 / 초기 30억원');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(REPORT/'{idx}_two_axes.png');plt.show()
''')
md('''## 8. 통합 전략에 적용할 선택값
통합 운용에는 **운용 성과축의 세 모형**을 가져간다. 예측축 모형을 통합계좌에 추가해 이중으로 투자하지 않는다. 이번 작업은 두 축의 단독 검증과 2024년 누락 복원까지다.

이 선택값으로 공유 거래대금 한도를 적용한 통합 원장은 **아직 재계산하지 않았다**. 이전 모형의 90억원 통합 성과를 이번 선택값의 성과로 재사용하지 않는다. 통합 시 각 30억원 현금 계정을 구분하고 같은 종목·같은 날 주문의 거래대금 한도를 공유해야 한다.

## 검증 한계
표시된 t·p는 회차 순수익에 연도 군집 표준오차를 적용한 진단값이다. 다수 후보 탐색 전체를 보정한 확증 검정은 아니다. 반복해서 본 OOS에서 좋게 나온다는 사실만으로 과적합 문제가 사라지지는 않는다.

초기 KOSPI100 규정과 KRX300 BMI/TMI 연결, 일부 당시 적격성의 공식 근거 공백은 유지된다. 99개 부분 명단 허용은 계산 가능성을 회복한 연구상의 처리이며 그 규정 공백을 인증한 것은 아니다. 단독 IS 성과에는 학습 준비 구간의 기본 규칙 운용이 포함되며, 모든 기간을 학습 모델의 성과로 해석하지 않는다.
''')
ref=next((c for c in old.cells if c.cell_type=='markdown' and c.source.startswith('## 9. Reference')),None)
if ref:md(ref.source.split('**재현')[0])
md('''### 이번 결과의 재현 경로
`02_code/modules/index_engine_v8_revalidation_20261118/repair_partial_k100.py` → `select_two_axes.py` → `extend_broad_broker.py` → `publish_two_axes.py`

결과: `04_results/two_axis_selection_20261118/`. IS 성적표는 앞선 동일 입력 실험을 재사용하고, OOS에 2024년 6월 입력만 복원했다. 원래 IS 입력이 바뀌지 않았는지 확인했다. 선택값은 OOS 재평가 전에 저장했다.
''')
code("checks=json.loads((R/'checks.json').read_text(encoding='utf8'));display(pd.Series(checks,name='통과').to_frame());assert all(checks.values())")
md('''## 10. Q&A
**왜 예측력 최고와 성과 최고가 다르나요?** F1은 편입 여부를 맞힌 건수를 평가한다. 운용 성과는 성공·실패 종목의 손익 크기, 이미 오른 정도, 진입 시점, 체결률까지 반영하기 때문이다.

**OOS에서 잘 나온 것을 최종 선택하면 안 되나요?** 그러면 OOS가 다시 선택용 표본이 된다. 여기서는 IS 선택을 고정하고 OOS의 좋은 결과와 나쁜 결과를 모두 공개한다.

**99개만 나왔는데 어떻게 매매하나요?** 99개는 전체 구성종목 예상치다. 그중 신규 편입 후보에만 신호를 만들고, 가짜 100번째 종목은 추가하지 않는다. 누락된 회차를 성과표에서 삭제하지도 않는다.

**증권사보다 좋은 엔진이 완성됐나요?** 현재 비교 가능한 동일 시점 표본의 결과로만 답해야 한다. KOSPI100은 자료 미확보, KRX300은 초기 준비 모형과 부분 명단의 제한적 비교이며, KOSPI200도 회차 수와 배포시점 인증의 한계가 있다.
''')
nb=nbf.v4.new_notebook(cells=cells);nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
runtime=OUT/'runtime';kernel=runtime/'kernels/python3'
os.environ['JUPYTER_DATA_DIR']=str(runtime);os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(runtime/'ipython')
km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
NotebookClient(nb,timeout=240,km=km,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.validate(nb);nbf.write(nb,P)
report=ROOT/'05_reports/two_axis_selection_20261118';export=HTMLExporter();export.exclude_input=True;html,_=export.from_notebook_node(nb)
(report/'두축_발표용통합노트.html').write_text(html,encoding='utf8')
print(P)
