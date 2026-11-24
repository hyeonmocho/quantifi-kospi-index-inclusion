from context import ROOT,OUT
import os,sys,json
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# 세 지수 편입 예측 전략: 80·60·40·20거래일 진입 비교
2026-11-18 · V8 고정 입력 · KOSPI200 / KOSPI100 / KRX300

**질문:** 심사마감일에 더 가까워진 후 예측해서 사면 수익과 위험이 어떻게 달라지는가?

- 지수별 기존 IS 선택 모형 계열을 유지한다. KOSPI200은 부스팅 점수 상위 후보(기본 엔진 후보 수 유지), KOSPI100·KRX300은 부스팅 확률 50% 이상이다. 각 시점의 IS 학습 결과를 사용하므로 계수와 예상 명단은 달라질 수 있다. 같은 종목을 단순히 다른 날짜에 사는 실험은 아니다.
- 원금 각 30억원, 정보일 다음 거래일 종가부터 최대 10거래일 집행. 하루 한도는 직전 ADV20의 10%와 당일 거래대금의 10% 중 작은 값. 매수·매도 각 25bp, 미체결 현금, 청산 완료까지 같은 한도 적용. 이익 재투자 없음.
- 20·40·60·80은 **심사마감일까지의 거리**다. 보유일 수나 확률 추적 분할매수 일정이 아니다. 청산 일정은 기존 전략을 유지한다.
- IS는 기존 공통 유효 회차, OOS는 2023-01-02~2026-06-30. 이번 OOS 민감도 결과로 기존 선택을 변경하지 않는다. 이미 관찰한 역사적 OOS이며 독립적인 신규 검증이 아니다.

**표본 통제:** KOSPI200·KRX300은 OOS 7회 공통. KOSPI100은 20일 예측에서 2024년 6월 명단을 생성하지 못해 네 시점 모두에서 그 회차를 제외한 **6회 공통**으로 비교한다. 따라서 KOSPI100 주표는 이전 7회 성과와 다를 수 있다. 제외된 회차는 무신호가 아니며, 아래 보조표에서 각 시점의 가용 표본 결과도 따로 공개한다. 공통 표본 선정 자체도 사후 자료 완전성에 따른 연구 제한이다.
''')
code('''from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'04_results').exists())
R=ROOT/'04_results/index_engine_v8_revalidation_20261118/timing_comparison'
REPORT=ROOT/'05_reports/index_engine_v8_revalidation_20261118/timing';REPORT.mkdir(exist_ok=True)
p=pd.read_csv(R/'performance.csv');daily=pd.read_parquet(R/'daily.parquet')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':130})
columns=['phase','lead','n_events','mean_event_net','cagr','volatility','sharpe','mdd','win_rate','payoff','year_cluster_t','year_cluster_p']
labels=['구간','심사마감 전 거래일','회차','회차 평균 순수익','CAGR','연율 변동성','샤프','MDD','승률','손익비','연도군집 t','연도군집 p']
def table(q):
 z=q[columns].copy();z.columns=labels
 pct=['회차 평균 순수익','CAGR','연율 변동성','MDD','승률']
 fmt={x:'{:.2%}' for x in pct};fmt.update({x:'{:.3f}' for x in ['샤프','손익비','연도군집 t','연도군집 p']})
 display(z.style.format(fmt,na_rep='—').hide(axis='index'))
display(pd.read_csv(R/'coverage.csv'))
''')
md('''## 1. 현재 선택 모형을 유지한 진입 시점 비교

회차 평균은 회차 손익/고정 원금의 산술평균이다. CAGR·샤프·변동성·MDD는 대기 현금까지 포함한 일별 계좌 지표다. 샤프의 무위험수익률은 0, 연율화는 252일이다. t·p는 회차 수익에 대한 연도군집 양측검정이며, OOS는 4개 연도뿐이어서 불확실성이 크다. 여러 시점 비교에 대한 다중검정 보정은 아니다.
''')
for idx in ['KOSPI200','KOSPI100','KRX300']:
 md('### '+idx)
 code(f'''idx='{idx}'
q=p[p.index_id.eq(idx)&p.family.eq('selected_family')&p.cohort.eq('common')].sort_values(['phase','lead'],ascending=[True,False])
table(q)
fig,axes=plt.subplots(1,2,figsize=(12,3.5))
for phase,ax in zip(['IS_forward','OOS'],axes):
 for lead,color in zip([80,60,40,20],['#2459A6','#168A78','#D28A31','#b54464']):
  z=daily[daily.index_id.eq(idx)&daily.family.eq('selected_family')&daily.cohort.eq('common')&daily.phase.eq(phase)&daily.lead.eq(lead)].sort_values('date')
  ax.plot(pd.to_datetime(z.date),z.equity/3e9,label=f'{{lead}}거래일',color=color)
 ax.set_title(idx+' '+phase);ax.grid(alpha=.2);ax.set_ylabel('최초 원금 대비 NAV');ax.legend(ncol=2)
fig.tight_layout();fig.savefig(REPORT/(idx+'_timing_NAV.png'));plt.show()
''')
md('''## 2. 기본 규정 엔진 비교

부스팅 모형과 기본 엔진의 차이를 함께 확인한다. 모델·진입일 두 가지를 동시에 바꾼 이전 결과를 한 가지 원인으로 해석하지 않는다. 아래 역시 같은 지수 내 공통 회차다.
''')
code('''for idx in ['KOSPI200','KOSPI100','KRX300']:
 print(idx+' / 규정 기반 기본 엔진');table(p[p.index_id.eq(idx)&p.family.eq('baseline')&p.cohort.eq('common')].sort_values(['phase','lead'],ascending=[True,False]))
q=p[p.phase.eq('OOS')&p.cohort.eq('common')].pivot(index=['index_id','family'],columns='lead',values='sharpe').reindex(columns=[80,60,40,20])
fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(q.values,cmap='RdYlGn',vmin=-1,vmax=1.5,aspect='auto')
ax.set_xticks(range(4),['80일','60일','40일','20일']);ax.set_yticks(range(len(q)),[a+' / '+('선택 모형' if b=='selected_family' else '기본 엔진') for a,b in q.index])
for i in range(len(q)):
 for j in range(4):ax.text(j,i,f'{q.iloc[i,j]:.3f}',ha='center',va='center')
ax.set_title('OOS 샤프 민감도 · 같은 지수 내 공통 회차');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(REPORT/'timing_sharpe.png');plt.show()
''')
md('''## 3. KOSPI100 가용 표본 보조 결과

40·60·80일은 7회 결과도 계산 가능하지만, 20일은 6회다. 아래 7회 표와 주표 6회를 섞어서 우열을 판단하지 않는다.
''')
code("table(p[p.index_id.eq('KOSPI100')&p.family.eq('selected_family')&p.cohort.eq('available_supplement')].sort_values('lead',ascending=False))")
md('''## 4. 해석과 재현

네 시점의 차이에는 매수 가격과 보유기간뿐 아니라 **당시 정보로 선택되는 종목과 체결 가능량의 차이**가 함께 들어간다. 주가가 오른 종목을 사후에 골라 같은 바스켓으로 비교한 결과가 아니다. OOS에서 가장 좋아 보이는 시점은 사후 관찰값으로만 다루며, 이번 민감도 분석으로 기존 `performance_choices.csv`를 바꾸지 않았다.

재현: `02_code/modules/index_engine_v8_revalidation_20261118/timing_comparison.py` 실행 후 `publish_timing.py`. 입력 해시, 계좌별 거래 원장, 현금·수수료·거래대금 한도·손익 대사는 결과 폴더에 저장했다.
''')
code("checks=json.loads((R/'checks.json').read_text());display(pd.Series(checks).to_frame('통과'));assert all(checks.values())")
nb=nbf.v4.new_notebook(cells=cells);nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
runtime=OUT/'runtime';kernel=runtime/'kernels/python3'
os.environ['JUPYTER_DATA_DIR']=str(runtime);os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(runtime/'ipython')
km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
NotebookClient(nb,timeout=240,km=km,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.validate(nb)
path=ROOT/'03_notebooks/지수편입예측_V8_80_60_40_20일_진입비교_20261118.ipynb';nbf.write(nb,path)
ex=HTMLExporter();ex.exclude_input=True;html,_=ex.from_notebook_node(nb)
(ROOT/'05_reports/index_engine_v8_revalidation_20261118/timing/진입시점비교.html').write_text(html,encoding='utf8')
print(path)
