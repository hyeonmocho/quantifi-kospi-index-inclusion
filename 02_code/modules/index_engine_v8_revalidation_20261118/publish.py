"""Executed presentation notebook: one V8 version for prediction and performance."""
from context import ROOT,OUT,dump
import json,sys,os,hashlib
import pandas as pd,numpy as np
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

def prepare_tables():
 f=pd.read_csv(OUT/'forecasts.csv');choices=pd.read_csv(OUT/'performance_choices.csv')
 acc=pd.read_csv(OUT/'accuracy_summary.csv');perf=pd.read_csv(OUT/'portfolio_roles.csv');ranges=[]
 for r in choices.itertuples():
  for phase in ['IS_forward','OOS']:
   g=f[f.index_id.eq(r.index_id)&f.model.eq(r.model)&f.lead.eq(r.lead)&f.phase.eq(phase)]
   if phase=='IS_forward':g=g[g.event.isin(json.loads((OUT/'common_IS_events.json').read_text())[r.index_id])]
   ranges.append(dict(index_id=r.index_id,phase=phase,first_event=g.event.min(),last_event=g.event.max(),n_events=len(g),information_start=g.information_date.min(),information_end=g.information_date.max(),entry_start=g.entry_date.min(),planned_exit_end=g.exit_date.max()))
 pd.DataFrame(ranges).to_csv(OUT/'selected_periods.csv',index=False,encoding='utf-8-sig')
 before=pd.read_csv(ROOT/'04_results/index_engine_v7_20261118/portfolio_roles.csv');before=before[before.role.eq('IS_performance_selected')&before.phase.eq('OOS')].copy();before['version']='V7'
 after=perf[perf.role.eq('IS_performance_selected')&perf.phase.eq('OOS')].copy();after['version']='V8'
 pd.concat([before,after]).to_csv(OUT/'V7_V8_performance_bridge.csv',index=False,encoding='utf-8-sig')
 # Forecast availability is measured separately from valid-only precision/recall.
 source=ROOT/'04_results/index_engine_v8_20261118'
 dump('release_status',dict(data_frozen=True,IS_selection_rerun=True,historical_OOS_replayed=True,independent_forward_observations=0,rule_certification_complete=False,broker_outperformance_proven=False,unavailable_source_files=str(source/'task_completion_status.json')))

def publish():
 old=nbf.read(OUT/'presentation_before_v8.ipynb',as_version=4);cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):cells.append(nbf.v4.new_code_cell(s))
 md('''# 지수 편입을 미리 예측하는 세 가지 롱온리 전략
### KOSPI200 · KOSPI100 · KRX300
**조현모 · 퀀티파이 · 2026년 11월 18일 · V8 재검증 확정본**

반복되는 지수 정기변경을 예측하는 능력과, 그 예측을 비용·유동성 제약 아래에서 수익으로 바꾸는 능력을 구분한다.

**이 노트북의 주 표와 그래프는 모두 V8 입력으로 재학습·재계산한 결과다.** 이전 발표 정리본은 백업했으며, 기존 204.49억원 통합계좌와는 IS·선택 모델·표본 설정이 다르다. 같은 실험의 수익률 정정으로 해석하지 않는다. 새 미래 실적이 아닌, 이미 여러 번 확인한 역사적 OOS 재검증이다.''')
 code('''from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'04_results').exists())
R=ROOT/'04_results/index_engine_v8_revalidation_20261118'
REPORT=ROOT/'05_reports/index_engine_v8_revalidation_20261118';REPORT.mkdir(exist_ok=True)
read=lambda n:pd.read_csv(R/(n+'.csv'))
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':120})
NAMES={'baseline':'규정 기반 기본 엔진','boost_p50':'부스팅·확률 50% 이상','boost_count':'부스팅·기본 후보 수 유지','logistic_p50':'로지스틱·확률 50% 이상','logistic_count':'로지스틱·기본 후보 수 유지','strict_logistic_count':'재무 확인 후보·로지스틱'}
def table(q, pct=(), numeric=()):
 fm={c:'{:.2%}' for c in pct if c in q};fm.update({c:'{:.3f}' for c in numeric if c in q})
 display(q.style.format(fm,na_rep='—').hide(axis='index'))
accuracy=read('accuracy_summary');performance=read('portfolio_roles');choices=read('performance_choices')
daily=pd.read_parquet(R/'daily_accounts.parquet')
def performance_table(idx):
 q=performance[performance.index_id.eq(idx)&performance.role.eq('IS_performance_selected')].copy()
 q['model']=q.model.map(NAMES)
 cols=['index_id','phase','model','lead','n_events','mean_event_net','win_rate','payoff','cagr','volatility','sharpe','mdd','year_cluster_t','year_cluster_p']
 table(q[cols],pct=['mean_event_net','win_rate','cagr','volatility','mdd'],numeric=['payoff','sharpe','year_cluster_t','year_cluster_p'])
 display(read('selected_periods').query('index_id==@idx'))
def chart(idx):
 r=choices[choices.index_id.eq(idx)].iloc[0]
 q=daily[daily.index_id.eq(idx)&daily.phase.eq('OOS')&daily.model.eq(r.model)&daily.lead.eq(r.lead)].sort_values('date')
 fig,axs=plt.subplots(2,1,figsize=(11,5.5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
 x=pd.to_datetime(q.date);nav=q.equity/3e9
 axs[0].plot(x,nav,color='#175c94',lw=2);axs[0].set(title=f'{idx} | IS 운용 선택 · OOS 계좌 NAV',ylabel='최초 원금 대비 자산');axs[0].axhline(1,color='gray',lw=.7)
 axs[1].fill_between(x,nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1,0,color='#bd4a4a',alpha=.6);axs[1].yaxis.set_major_formatter(PercentFormatter(1));axs[1].set_ylabel('고점 대비 낙폭')
 for ax in axs:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(REPORT/(idx+'_OOS_NAV.png'));plt.show()
''')
 md(old.cells[2].source)
 md('''**발표 대본:** “매년 반복되는 변경에서 실제 편입 명단을 미리 추정합니다. 편입 예측이 맞아도 이미 가격이 올랐다면 수익은 작을 수 있어, 예측 정확도와 투자 성과를 따로 보겠습니다.”''')
 md(old.cells[5].source)
 # Retain reviewed literature and illustrative formula, replace obsolete model-role assignments.
 literature=old.cells[6].source.split('### 3-3.')[0]
 md(literature)
 md('''### 3-3. 이번에 비교하는 엔진은 무엇인가?

| 엔진 | 이해하기 쉬운 설명 | 고정한 설정 |
|:--|:--|:--|
| 규정 기반 기본 엔진 | 예상 평균 시총과 규정으로 새 편입 명단을 계산 | V8 적격성 제외 적용 |
| 로지스틱 | 순위·시총·수급·자료 확인 상태를 과거 정답과 연결해 점수를 계산 | 정규화 C=0.1 |
| 작은 부스팅 모형 | 여러 개의 작은 의사결정 규칙을 순서대로 결합 | 100회 학습, 최대 7개 잎, 학습률 0.05 |

확률 점수 **50% 이상**을 고르는 방식과 **기본 엔진의 후보 수만큼** 상위 점수를 고르는 방식을 비교한다. KRX300은 재무자료가 확인된 후보만 허용하는 로지스틱 조건도 비교한다. 모형 출력 확률이 실제 빈도와 완전히 일치한다는 뜻은 아니다.

V8은 원문 재무 근거와 당시 공표된 제외 사유를 반영한다. 시총 경로 시뮬레이션을 새 방식으로 교체한 실험은 아니다. 대형 딥러닝도 사용하지 않았다.

### 3-4. 두 가지 목표를 IS에서 따로 선택

- **예측 목적:** IS 순차검증 F1 최대. 동률이면 정밀도·단순성·더 긴 선행일 순서.
- **운용 목적:** 각 선행일의 IS F1 1위 모형과 기본 엔진으로 후보를 제한. 평균 회차 순수익이 양수인 유효 계좌 중 IS 일별 샤프 최대.
- 진입 시점 후보는 심사마감일 **20·40·60·80거래일 전**이다. 이 날짜는 보유일 수가 아니며 실제 보유는 정기변경 관련 청산일까지 이어진다.
- 각 지수의 가용 과거부터 **2022년까지**만 학습한다. 최소 4개 과거 회차가 확정된 뒤 IS 순차평가를 시작한다. 같은 회차의 모든 종목·관측시점을 학습·평가 사이에 나누지 않는다.
- OOS는 **2023-01-02~2026-06-30**. OOS 수익률·t·p로 모델을 선택하지 않는다. 여러 차례 관찰한 역사적 연구 구간이라는 한계는 남는다.

아래 IS 범위는 실제 순차검증에 사용한 공통 유효 회차다. 초기 학습 회차나 입력 공백 회차를 모두 거래했다는 뜻은 아니다.''')
 code("table(read('frozen_choices'),pct=['IS_f1','IS_precision','IS_recall'])\ntable(choices,pct=['IS_mean_event_net'],numeric=['IS_sharpe'])\ndisplay(read('selected_periods'))")
 md('''## 4. 엔진의 예측력과 증권사 비교

정밀도는 **예측한 신규 편입 중 실제로 맞힌 비율**, 재현율은 **실제 신규 편입 중 찾아낸 비율**이다. 기존 구성종목을 적중 건수에 더하지 않는다. 예측 목적으로 IS에서 선택한 모델의 결과다.''')
 code('''q=accuracy[accuracy.selected&accuracy.phase.isin(['IS_forward','OOS'])].copy();q['model']=q.model.map(NAMES)
table(q[['index_id','phase','model','lead','n_events','hits','predicted','actual','precision','recall','f1']],pct=['precision','recall','f1'])
q=q[q.phase.eq('OOS')].set_index('index_id').reindex(['KOSPI200','KOSPI100','KRX300'])
fig,ax=plt.subplots(figsize=(10,4));x=np.arange(3)
for j,(c,l) in enumerate([('precision','정밀도'),('recall','재현율'),('f1','F1')]):ax.bar(x+(j-1)*.23,q[c],width=.23,label=l)
ax.set_xticks(x,q.index);ax.set_ylim(0,1.12);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=3);ax.set_title('V8 · 예측 목적 선택 모델의 OOS');fig.tight_layout();fig.savefig(REPORT/'prediction_OOS.png');plt.show()
display(read('forecast_availability'))
''')
 md('''**예측 생성 실패도 별도 공개한다.** KOSPI100 20일 입력은 2024년 6월에 예상 구성명단이 정원 100개를 채우지 못해 산출이 중단된 사례가 있다. 위 유효 회차 표와, 해당 회차를 무예측으로 포함한 `forecast_availability` 재현율을 함께 읽는다. 빈칸을 실제 명단으로 채우지 않았다.

증권사와는 같은 정보일에 엔진을 계산하고 동일한 진입·청산·비용·원금으로 비교한다. 최근 KOSPI200 비교는 전체 명단과 입력을 갖춘 보고서에 한정된다. KOSPI100은 전체 예측 리포트가 없고, KRX300은 과거 전체 리포트 3건과 일부 종목 의견이 있으나 현재 OOS와 동일한 비교 표본을 확보하지 못했다. 전체 수집 리포트 수와 실제 비교 수는 다르다.''')
 code("table(read('broker_summary'),pct=['precision','recall','f1','mean_net','win_rate'],numeric=['payoff'])\ntable(read('broker_paired_inference'),pct=['mean_net_difference'],numeric=['paired_signflip_p'])")
 md('''**발표 대본:** “공정한 비교를 위해 같은 날짜의 예측을 비교했습니다. 최근 표본만으로 증권사보다 우수하다고 단정하지 않습니다. 명단 정확도와 비용 반영 수익률은 서로 다른 평가 축이고, 두 지수는 비교 리포트 부족 자체를 공개합니다.”''')
 md(old.cells[14].source.replace('성과표는 비교 가능한 공통 회차만 남긴 연구계좌다.','IS 성과표는 선행일 간 비교 가능한 공통 회차로 제한한 연구계좌이며, OOS는 각 선택 모델의 유효 회차를 사용한다.'))
 for num,idx,idea in [(5,'KOSPI200','산업군 내 평균 시총 순위와 유동성을 예측해 신규 정기편입 후보를 선매수한다. 규정 경계를 넘을 가능성과 공표 전 가격 반영의 균형이 중요하다.'),(6,'KOSPI100','KOSPI200 내부에서 상위 대형주로 올라갈 후보를 예측한다. 후보 수가 적어 한 종목의 성공·실패가 계좌를 크게 움직일 수 있다.'),(7,'KRX300','KOSPI와 KOSDAQ을 합친 산업군·규모·유동성·재무 적격성을 평가한다. 더 넓은 후보군을 다루지만 부모 유니버스와 당시 재무 공개시점의 불확실성이 크다.')]:
  md(f'''## {num}-1. {idx} 단독 전략

{idea}

선택된 후보를 동일 비중 목표로 매수하고, 30억원의 고정 기준원금과 체결 가능한 거래대금 범위 안에서만 주문한다. 틀린 편입 예측의 손익도 그대로 남긴다.

## {num}-2. {idx} 성과

아래는 **운용 목적의 IS 선택** 결과다. 앞 절 예측 목적 모델의 성과와 혼동하지 않는다. 회차 수익률과 현금을 포함한 달력상 계좌 지표를 함께 제시한다.''')
  code(f"performance_table('{idx}')\nchart('{idx}')")
 md('''## 8. 세 전략 통합: 각 30억원, 총 90억원

각 전략의 IS 선택 모델을 그대로 가져온다. 전략별 현금은 따로 관리하고, 같은 종목을 같은 날 주문하면 **세 계좌의 매수·매도 거래대금 한도를 공유**한다. 한도가 부족하면 주문 수요에 비례해 나눈다. 청산을 우선하고 수익을 다음 회차 목표 원금에 재투자하지 않는다.

통합 수익률은 단독 수익률을 단순 평균한 값이 아니라, 공유 한도를 적용한 원화 주문 원장과 일별 현금·주식 평가액에서 계산한다.''')
 code('''display(read('integration_coverage'))
q=read('integration_performance').copy()
for c in ['budget_krw','ending_equity_krw','profit_krw','total_fees']:q[c]=q[c]/1e8
table(q[['index_id','budget_krw','ending_equity_krw','profit_krw','total_return','cagr','volatility','sharpe','mdd','initial_fill_ratio','position_days','max_exit_delay_sessions']],pct=['total_return','cagr','volatility','mdd','initial_fill_ratio'],numeric=['budget_krw','ending_equity_krw','profit_krw','sharpe'])
print('금액 단위: 억원 / 체결률: 실제 매수금액 ÷ 최초 목표 주문금액')
jd=pd.read_parquet(R/'integration_daily.parquet');fig,ax=plt.subplots(figsize=(11,4.4))
for idx,z in jd.groupby('index_id'):
 ax.plot(pd.to_datetime(z.date),z.equity/(9e9 if idx=='COMBINED' else 3e9),label='90억원 통합' if idx=='COMBINED' else idx,lw=2.5 if idx=='COMBINED' else 1.2)
ax.set(title='V8 통합계좌 OOS | 최초 원금 대비 자산',ylabel='NAV');ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(REPORT/'combined_NAV.png');plt.show()
z=jd[jd.index_id.ne('COMBINED')].pivot(index='date',columns='index_id',values='held_value').fillna(0)/9e9
fig,ax=plt.subplots(figsize=(11,3.5));ax.stackplot(pd.to_datetime(z.index),z.T,labels=z.columns,alpha=.8);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(title='언제 어떤 전략에 포지션이 있는가?',ylabel='총 원금 대비 주식 평가액');ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(REPORT/'combined_exposure.png');plt.show()
table(read('integration_event_inference'),pct=['mean_event_net','win_rate'],numeric=['payoff','year_cluster_t','year_cluster_p'])
table(read('integration_monthly_inference'),pct=['mean_monthly','win_rate'],numeric=['t','p'])
''')
 md('''회차 검정과 월별 검정은 서로 다른 질문이다. 회차는 이벤트 자체의 손익, 월별은 현금을 포함한 계좌의 시간 흐름을 평가한다. 월별 HAC 시차 1·3·6 중 가장 좋은 p값을 골라 전략을 채택하지 않는다. 고정 원금 운용이어도 평가액은 주가 상승으로 원금을 넘을 수 있다. CAGR은 계좌 자산 변화의 기하 연율 환산이며 주문 원금 재투자 가정이 아니다.''')
 md('''### V7과 V8의 차이: 같은 조건에서 자료 보완 후 재선택

아래 표는 V7 대비 V8이다. 기존 발표 노트의 다른 장기 IS 선택이나 과거 204.49억원 계좌와의 순수 자료 효과 비교가 아니다. IS 최적 모델·시점이 바뀌었다면 그 변화까지 포함된다.

**KOSPI200의 선택이 80일에서 40일로 바뀐 이유:** V7에서는 A008320의 보유가격 공백으로 40일 기본·부스팅 후보 계좌를 계산할 수 없어 IS 선택 대상에서 빠졌다. V8에서는 당시 공개된 부적격 근거를 예측일에 연결해 해당 후보를 차단했고, 40일 계좌가 유효해졌다. 그 결과 IS 샤프 1.243의 부스팅·40일이 기본·80일의 1.092보다 높아 선택됐다. 그러나 그 선택의 OOS 샤프는 0.614로, 이전 V7 선택의 0.994보다 낮다. **자료 보완이나 IS 성과 개선이 OOS 성과 개선을 보장하지 않는 사례**이며, 더 좋았던 OOS에 맞춰 선택을 되돌리지 않았다.''')
 code("q=read('V7_V8_performance_bridge');table(q[['version','index_id','model','lead','n_events','mean_event_net','cagr','sharpe','mdd','year_cluster_t','year_cluster_p']],pct=['mean_event_net','cagr','mdd'],numeric=['sharpe','year_cluster_t','year_cluster_p'])")
 md('''## 검증 한계 — 발표에서는 이 한 장을 반드시 함께 제시

| 구분 | 확인한 것 | 아직 주장할 수 없는 것 |
|:--|:--|:--|
| 자료 | 원문 재무 근거 574건, 당시 공시일 연결, 확인된 부적격 후보 차단 | 과거 BMI·2026 TMI 전체 명단, 모든 과거 적격성의 인증 |
| 규정 | 공개 규정·역사 개정사항의 구현과 근거 보존 | 초기 KOSPI100 완충규정·모든 개별 심사 결정의 완전 재현 |
| 예측 | 동일 규칙의 IS 선택과 역사적 OOS 정밀도·재현율 | 증권사보다 우수한 실전 엔진이라는 일반적 결론 |
| 성과 | 비용·거래대금·현금 제약을 반영한 모의 집행 | 실제 호가·체결 원장으로 입증한 실행 가능성과 시장충격 |
| 통계 | 회차·연도 의존성을 고려한 t·p, 민감도와 실패 사례 공개 | 반복 탐색에 따른 과적합이 사라졌다는 결론 |
| 미래 검증 | 전진 프로토콜과 근거 해시 등록 | 아직 0건인 새 전진 회차의 성과 |

**발표 결론:** 공개 규정과 시점 정보를 연결해 편입 예측에서 원화 집행까지 검증하는 연구 엔진을 만들었다. 예측력이 높은 조건과 운용 성과가 높은 조건을 별도로 비교했다. 남은 데이터 공백과 작은 표본을 공개하며 다음 미관측 회차에서 확인한다.''')
 refs=old.cells[43].source.split('**프로젝트 자료**')[0]
 md(refs+'''\n**이번 발표의 재현 경로**

- 자료·모형 사전 고정: `04_results/index_engine_v8_revalidation_20261118/pre_run_freeze.json`
- 입력 및 결과: 같은 폴더의 `panel_enriched.parquet`, `forecasts.csv`, `portfolio_roles.csv`, `integration_ledger.parquet`
- 재현 코드: `02_code/modules/index_engine_v8_revalidation_20261118/`
- 순서: `research.py` → `evaluate.py` → `validate.py` → `combined.py` → `publish.py`
- 노트북은 저장된 확정 결과로 표와 그래프를 재생성한다. 코드를 실행해 OOS에 맞춰 조건을 바꾸지 않는다.
''')
 code("checks=pd.DataFrame(json.loads((R/'checks.json').read_text(encoding='utf8')));display(checks);assert checks.passed.all()\ndisplay(pd.Series(json.loads((R/'integration_checks.json').read_text(encoding='utf8'))).to_frame('통합 검사'))")
 md('''## 10. Q&A

**Q. 편입을 잘 맞히는데 왜 수익이 가장 높지는 않나요?**  
이미 예상된 종목은 매수 전 가격에 반영됐을 수 있고, 성공 종목의 상승폭과 실패 종목의 손실폭도 다르다. F1은 명단의 정확성을, 운용 샤프는 시간에 따른 계좌 손익과 위험을 평가한다.

**Q. 지수별로 다른 모델을 선택해도 되나요?**  
규정 경계, 후보 수, 업종 구조, 유동성, 표본 기간이 다르므로 IS에서 선택되는 조건이 다를 수 있다. 이것만으로 경제적 원인이 확정된 것은 아니다.

**Q. 지금 바로 실전 운용 가능한가요?**  
연구 발표와 모의 전진검증이 가능한 단계다. 과거 규정·명단 공백과 실제 주문자료가 남아 있어 완전 인증된 실전 성과로 소개하지 않는다.

**Q. 예전 성과 숫자와 왜 다른가요?**  
엔진 입력·IS 범위·모델 선택·체결 제약·회차가 같은지 먼저 확인해야 한다. 이번 본문의 수치는 V8으로 통일했고 이전 발표본은 별도 백업했다.
''')
 nb=nbf.v4.new_notebook(cells=cells);nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
 runtime=OUT/'runtime';kernel=runtime/'kernels/python3';kernel.mkdir(parents=True,exist_ok=True)
 (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python 3','language':'python'}),encoding='utf8')
 os.environ['JUPYTER_DATA_DIR']=str(runtime);os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(runtime/'ipython')
 from jupyter_client import KernelManager
 from jupyter_client.kernelspec import KernelSpecManager
 km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
 NotebookClient(nb,timeout=240,km=km,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.validate(nb)
 path=ROOT/'03_notebooks/지수편입예측_3전략_발표용통합노트_20261118.ipynb';nbf.write(nb,path)
 ex=HTMLExporter();ex.exclude_input=True;html,_=ex.from_notebook_node(nb)
 report=ROOT/'05_reports/index_engine_v8_revalidation_20261118';report.mkdir(exist_ok=True);(report/'3전략_발표용통합노트_V8.html').write_text(html,encoding='utf8')
 dump('publication_manifest',dict(notebook=str(path.relative_to(ROOT)),executed_cells=sum(c.cell_type=='code' for c in nb.cells),errors=sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 print(path,flush=True)
if __name__=='__main__':prepare_tables();publish()
