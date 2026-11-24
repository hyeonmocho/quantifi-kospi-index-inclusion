"""Executed teaching notebook and matched complexity audit."""
from context import ROOT,OUT
import os,json
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md(r'''# 예측 엔진을 복잡하게 만들면 실제로 더 좋아지는가?
2026-11-18 · V8 동일 입력 · 세 지수 롱온리 연구

## 1. 모든 시총 추정의 공통 출발점

이미 관측한 심사기간 시총은 바꾸지 않고, 남은 구간만 추정한다.

$$\widehat{\bar M}_i=\frac{S_{i,관측}+D_i\widehat M_{i,미관측평균}}{N_i}$$

여기서 $D$는 남은 거래일 수, $N$은 전체 유효 심사일 수다. 최종 하루의 시총을 맞히는 것과 남은 기간의 **평균 시총**을 맞히는 것은 다르다. 종목별 기업행위에 따른 심사기간 처리는 별도 입력으로 유지한다.

### 현재 시총 유지와 20일 이동평균

기본 가정은 남은 매일의 시총이 오늘과 같다는 것이다. 이동평균 방식은 최근 20거래일 시총 평균을 남은 기간의 평균으로 사용해 하루의 급등락 영향을 완화한다. 실제 코드에서는 `log(MA20 / 현재시총)`을 ±0.4로 제한한다. 즉 미래 평균/현재시총의 비율이 약 0.670~1.492 범위를 넘지 않게 한다. 이동평균 계산값이 없으면 이 조정값을 0으로 두어 현재 시총 유지로 돌아간다.

예시: 120일 중 관측한 80일의 평균이 2조, 현재 시총 2.2조, MA20이 2.1조라면 기본 예상 심사평균은 2.067조, MA20 방식은 2.033조다. 설명용 숫자이며 실측 종목 사례가 아니다.

### 완만한 추세 보정

실제 구현은 최근 20거래일 **주가수익률** $r_{20}$으로 다음 값을 계산한다.

$$z=\operatorname{clip}\left(0.25\log(1+r_{20})\frac D{40},-0.4,0.4\right),\quad \widehat M_{미관측평균}=M_{현재}e^z$$

20일 추세를 남은 D일로 연장한 변화의 약 절반을 평균 구간으로 보고, 그 효과를 다시 25%만 남긴 근사식이다. 엄밀한 지수경로 평균 공식은 아니다. 0.25는 연구 설정이며 학술적으로 보장된 최적값이 아니다. 최근 20일 +10%, 남은 40일이면 예상 미래 평균은 현재 시총보다 약 +2.41% 높아진다. 주식 수 변화가 있으면 주가 추세와 시총 추세가 다를 수 있다. 공시된 주식 수 변화를 연결하는 인터페이스는 있지만, 알려지지 않은 미래 발행주식 수를 만들어 넣지 않는다.

`anchored_rank_update`는 이 식과 다른 모델이다. 20일 전 예상 심사평균 시총 순위에 현재 시총 순위의 변화량을 더해 업종 내 순서를 바꾸는 한국 자료 기반 근사 방식이다. 공식 거래소 순위표를 재현한 모델은 아니다.

## 2. 시나리오 방식은 무엇을 바꾸는가?

한 가지 미래 시총 대신 여러 미래 경로를 만든다. 실제 사용 설정은 과거 최대 126일 로그수익률, 최소 60개 관측, 과거 수익률 1~99% 윈저라이징이다. 업종 공통 움직임과 개별 종목 움직임을 분리하고 업종 간 상관관계도 반영한다. 업종 베타는 1 쪽으로 50% 축소, 공분산은 대각행렬 쪽으로 25% 축소한다. 부족한 위험 추정 자료에는 코드의 횡단면 변동성 대체 규칙을 사용한다.

$$M_{i,d+1}^{(s)}=M_{i,d}^{(s)}\exp\{\beta_i u_{g,d}^{(s)}+\epsilon_{i,d}^{(s)}-\tfrac12v_i\}$$

정규 충격 가정하에 현재 시총을 평균적으로 유지하도록 보정한 경로다. **과거 상승률이 계속된다는 기대수익을 넣지 않는다.** 최근 가격으로 변동성과 공통 충격을 추정하는 것과 과거 주식 고유 알파를 정상수익으로 차감하는 것은 다르다.

각 경로의 남은 시총 합을 관측 구간 합에 더한 뒤 지수 선정 규칙을 다시 실행한다. 256개 중 1개는 결정론적 현재 시총 유지 경로이고 나머지 255개가 확률 경로다. 미래 주식 수는 고정, 유동성·적격성은 해당 시점 기본 입력을 유지한다. 따라서 미래 기업행위·위원회 재량·자료 누락을 모두 시뮬레이션한 것은 아니다.

$$q_i=\frac{\text{256개 중 신규 편입 후보로 선정된 횟수}}{256}$$

예를 들어 205번 선정됐다면 약 80.1%다. 이것은 **설정한 시나리오 안에서의 선정빈도**이며 실제 시장에서의 편입 확률 80.1%를 보장하지 않는다. 시나리오 256개는 과거 정기변경 표본 256회가 아니다.

## 3. 로지스틱 회귀로 빈도를 보정하는 방법

로지스틱 회귀는 입력 점수를 0~1 확률로 바꾸는 S자 함수를 과거 정답에 맞춰 추정한다. 여기서는 선정빈도 하나를 입력으로 쓴다.

$$x_i=\log\frac{\tilde q_i}{1-\tilde q_i},\quad \tilde q_i=\operatorname{clip}(q_i,0.005,0.995)$$
$$p_i=\frac{1}{1+\exp[-(a+bx_i)]}$$

과거 실제 편입을 1, 비편입을 0으로 두고 a와 b를 학습한다. 각 회차의 학습 가중치 합을 같게 해 후보 수가 많은 회차 하나가 압도하지 않게 했다. 원래 단변수 보정 코드는 C=1, 양의 선정빈도 후보, 최소 과거 3회와 두 종류 정답을 요구한다. 이번 비교는 V8의 최소 4회 완료 조건을 통과한 평가 회차에서만 한다. 선정빈도 0인 후보는 보정 후에도 0으로 두는 구현 정책이 있어 전체 유니버스 확률로 일반화하면 안 된다.

### 다변수 로지스틱과 부스팅은 순차 통과 단계가 아니다

다변수 로지스틱은 선정빈도 외에도 시총 순위·모멘텀·수급 등을 함께 쓰는 대안이다. 부스팅은 여러 작은 나무를 더하는 다른 대안이다. 현재 V8 모형에 단변수 보정 확률을 먼저 넣고 그 결과를 다시 넣는 파이프라인은 아니다. 기존 규칙·시나리오 결과를 각각 입력으로 사용한다. 이들 설정은 우리 연구 구현이며 문헌의 완전 재현이 아니다.
''')
code('''from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'04_results').exists())
R=ROOT/'04_results/index_engine_v8_revalidation_20261118/complexity_audit'
REPORT=ROOT/'05_reports/index_engine_v8_revalidation_20261118/complexity';REPORT.mkdir(exist_ok=True)
read=lambda n:pd.read_csv(R/(n+'.csv'))
perf=read('performance');acc=read('accuracy');fits=read('calibration_fits');pairs=read('paired_improvement')
NAMES={'baseline':'기본 규정','frequency_count':'선정빈도 순위','calibration_count':'단변수 보정·같은 수','logistic_count':'다변수 로지스틱·같은 수','boost_count':'부스팅·같은 수','frequency_p50':'선정빈도 50%','calibration_p50':'단변수 보정 50%','calibration_p60':'단변수 보정 60%','logistic_p50':'다변수 로지스틱 50%','boost_p50':'부스팅 50%'}
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':130})
f=fits[fits.phase.eq('OOS')].groupby('index_id').first().reset_index();f['빈도80%_보정확률']=1/(1+np.exp(-(f.intercept+f.slope*np.log(4))))
display(f[['index_id','lead','train_events','intercept','slope','빈도80%_보정확률']])
fig,ax=plt.subplots(figsize=(7,4));q=np.linspace(.005,.995,200)
ax.plot(q,q,ls='--',color='gray',label='보정하지 않음')
for r in f.itertuples():ax.plot(q,1/(1+np.exp(-(r.intercept+r.slope*np.log(q/(1-q))))),label=r.index_id)
ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(xlabel='시나리오 선정빈도',ylabel='단변수 보정 확률',title='2022년까지 학습한 실제 보정 곡선');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(REPORT/'calibration_curves.png');plt.show()
''')
md('''## 4. 복잡성의 성과 비교: 조건을 맞춘 실험

기존 IS에서 고른 진입일을 고정한다: KOSPI200 40일, KOSPI100 40일, KRX300 80일. 각 모형에 유리한 OOS 진입일로 바꾸지 않았다. 단, 이 날짜들도 이전 연구의 선택 결과여서 완전히 중립적인 새 실험은 아니다.

같은 V8 적격성 후보·정답·회차·비용·집행 규칙을 사용한다. 각 30억원, 편도25bp, 직전 ADV20과 당일 거래대금 각각 10% 중 작은 한도, 최대10일 매수, 청산 완료까지 동일 한도, 별도 단독 계좌이며 이익 재투자 없음. OOS는 2023-01-02~2026-06-30 각7회다. KOSPI100은 이번에 40일로 고정했으므로 이전 시점 비교의 6회 표와 다르다.

IS는 기존 공통 유효 회차: KOSPI200 17회(2009~2022), KOSPI100 14회(2010~2022), KRX300 5회(2020~2022). OOS는 반복 관찰한 역사적 구간이다. 결과로 기존 모델 선택 파일은 바꾸지 않는다.

**같은 수 비교**는 기본 엔진이 고른 개수만큼 각 점수 상위 종목을 고른다. **50% 비교**는 같은 확률 문턱을 적용하지만 선택 종목 수·체결량이 달라질 수 있다. 단변수60%는 예전 정책의 보조 비교다. 표의 t·p는 연도군집, 샤프는 RF=0 일별 계좌, MDD는 계좌 전체 기준이다.
''')
code('''same=['baseline','frequency_count','calibration_count','logistic_count','boost_count']
def show(idx,models,phase):
 q=perf[perf.index_id.eq(idx)&perf.model.isin(models)&perf.phase.eq(phase)].merge(acc,on=['index_id','model','phase'],suffixes=('','_accuracy'))
 q['model']=pd.Categorical(q.model,categories=models,ordered=True);q=q.sort_values('model');q['방식']=q.model.astype(str).map(NAMES)
 cols=['방식','n_events','mean_event_net','cagr','volatility','sharpe','mdd','precision','recall','f1','year_cluster_p']
 fmt={c:'{:.2%}' for c in ['mean_event_net','cagr','volatility','mdd','precision','recall','f1']};fmt.update(sharpe='{:.3f}',year_cluster_p='{:.3f}')
 display(q[cols].style.format(fmt,na_rep='—').hide(axis='index'))
for idx in ['KOSPI200','KOSPI100','KRX300']:
 print(idx+' / 동일 후보 수 / IS');show(idx,same,'IS_forward')
 print(idx+' / 동일 후보 수 / OOS');show(idx,same,'OOS')
''')
md('''### 종목 수를 달리 허용하면 어떤가?

50% 문턱은 모형마다 같은 확률 의미를 보장하지 않는다. 더 적은 종목에 집중하거나 더 많은 현금을 보유한 효과와 예측 개선을 함께 확인해야 한다. 이전에 선택된 모형의 성과도 포함하며 재현 검사를 수행했다.
''')
code('''for idx in ['KOSPI200','KOSPI100','KRX300']:
 print(idx+' / 확률 문턱 / OOS');show(idx,['baseline','frequency_p50','calibration_p50','logistic_p50','boost_p50','calibration_p60'],'OOS')
q=perf[perf.phase.eq('OOS')&perf.model.isin(same)].pivot(index='index_id',columns='model',values='sharpe').reindex(index=['KOSPI200','KOSPI100','KRX300'],columns=same)
fig,ax=plt.subplots(figsize=(10,3));im=ax.imshow(q.values,cmap='RdYlGn',vmin=-.5,vmax=1.5,aspect='auto');ax.set_yticks(range(3),q.index);ax.set_xticks(range(len(same)),[NAMES[x].replace(chr(183),chr(10)) for x in same])
for i in range(3):
 for j in range(len(same)):ax.text(j,i,f'{q.iloc[i,j]:.3f}',ha='center',va='center')
ax.set_title('동일 진입일·같은 종목 수 조건의 OOS 샤프');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(REPORT/'complexity_sharpe.png');plt.show()
''')
md('''## 5. 기본 엔진보다 좋아진 차이가 반복되는가?

전체 수익이 양수인지 검정하는 것과 **기본 엔진 대비 개선분**을 검정하는 것은 다르다. 동일 회차 손익 차이에 대한 평균·연도군집 t,p·대응 부호반전 검정을 공개한다. 부호반전은 차이의 대칭성을 가정한다. 모델 다중비교 보정은 하지 않았으며 7회·4년의 작은 표본이다. 새 OOS 없이 우월성을 확정하지 않는다.
''')
code('''q=pairs[pairs.model.isin(['calibration_count','logistic_count','boost_count','calibration_p50','logistic_p50','boost_p50'])].copy();q['방식']=q.model.map(NAMES)
display(q[['index_id','방식','n_events','mean_net_difference','better_events','year_cluster_t','year_cluster_p','paired_signflip_p']].style.format({'mean_net_difference':'{:+.2%}','year_cluster_t':'{:.3f}','year_cluster_p':'{:.3f}','paired_signflip_p':'{:.3f}'}).hide(axis='index'))
''')
md('''## 6. 확률을 더 잘 표현하는 것과 더 버는 것은 다르다

확률 보정은 Brier score(확률과 정답의 제곱오차)와 Log loss(확신한 오답에 더 큰 벌점)로도 본다. 둘 다 낮을수록 좋다. 여기서는 양의 선정빈도 후보의 동일 표본, 회차별 동일 가중치다. 전체 유니버스의 확률 정확도와 동일하지 않다. Log loss의 0·1은 수치 계산을 위해 1e-6 경계로 제한했다.

단변수 로지스틱 기울기 b가 양수면 선정빈도 순서가 보존된다. 그래서 같은 수를 사는 경우 순위가 거의 그대로일 수 있다. 다만 빈도 0.005~0.995 제한에서 생기는 동률과 정렬 정책 때문에 완전히 같다는 것은 실제 명단으로 확인해야 한다. 확률 오차가 줄어도 명단이 같다면 수익은 바뀌지 않는다.
''')
code("display(read('calibration_reliability'));display(fits.groupby('index_id').slope.agg(['min','max']))")
md('''## 7. 해석 기준과 재현

**이번 결과:** 동일 후보 수 조건에서 선정빈도와 단변수 확률 보정의 매수 명단은 IS·OOS 57개 사례 모두 같았다. 따라서 보정만으로 수익이 증가하지 않았다. 양의 선정빈도 후보의 OOS Brier 오차는 KOSPI200·KRX300에서 줄었지만 KOSPI100에서는 커졌다.

KOSPI100의 다변수 로지스틱·부스팅은 기본 대비 OOS 평균 회차수익이 약 7.92%p 높았지만, 차이는 사실상 2026년 6월 한 회차에서 발생했다. KOSPI200의 다변수 로지스틱 차이도 2025년 12월, 부스팅 차이도 2025년 6월 한 회차에서 발생했다. KRX300의 부스팅은 7회 중 4회에서 개선됐으나 MDD가 기본의 −22.30%에서 −27.11%로 악화됐다. 같은 수의 부스팅 기본 대비 평균 개선분의 연도군집 p값은 KOSPI200 0.375, KOSPI100 0.446, KRX300 0.385다. 추가 복잡성이 안정적인 초과 성과를 준다고 확정할 근거는 아직 부족하다.

IS에서 같은 수 부스팅의 샤프는 KOSPI200 1.229→1.243, KOSPI100 0.300→0.314로 소폭 개선됐고 KRX300은 0.201→0.181로 낮아졌다. OOS에서 좋아 보이는 모델을 사후 최적 모델로 바꾸지 않았다. **기본 엔진은 기준 모델로 유지하고, 확률 보정은 확률 표현의 개선 여부로, 다변수 모형은 추가 검증 후보로 분리하는 해석**이 적절하다.

복잡한 모델은 예측 지표, 확률 오차, 비용 반영 수익·위험에서 기본 엔진 대비 어떤 개선을 주는지 각각 제시한다. 특정 지수·기간에서만 좋은 모형을 모든 전략의 상위 버전으로 부르지 않는다. 후보 수를 맞춘 결과와 확률 문턱 결과를 섞지 않는다. 모든 비교에서 OOS를 이미 확인했다는 사실과 자료·규정 인증 공백은 그대로 남는다.

코드: `02_code/modules/index_engine_v8_revalidation_20261118/complexity_audit.py`와 `publish_complexity.py`. 입력 해시·선택 명단·확률 보정 계수·주문 원장·대응 차이 검정을 결과 폴더에 저장했다. 이 노트북은 기존 시총 경로를 새로 시뮬레이션하거나 이동평균·추세 계수를 재최적화한 실험은 아니다. 그 부분은 실제 구현을 설명하고, V8에 저장된 규정·시나리오 입력 위의 선정 방식 복잡성을 비교한 것이다.
''')
code("checks=json.loads((R/'checks.json').read_text());display(pd.Series(checks).to_frame('통과'));assert all(checks.values())")
nb=nbf.v4.new_notebook(cells=cells);nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
runtime=OUT/'runtime';kernel=runtime/'kernels/python3'
os.environ['JUPYTER_DATA_DIR']=str(runtime);os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(runtime/'ipython')
km=KernelManager(kernel_name='python3',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
NotebookClient(nb,timeout=240,km=km,resources={'metadata':{'path':str(ROOT)}}).execute();nbf.validate(nb)
path=ROOT/'03_notebooks/지수편입예측_V8_단순엔진과ML_복잡성검증_20261118.ipynb';nbf.write(nb,path)
ex=HTMLExporter();ex.exclude_input=True;html,_=ex.from_notebook_node(nb)
(ROOT/'05_reports/index_engine_v8_revalidation_20261118/complexity/단순엔진과ML_비교.html').write_text(html,encoding='utf8')
print(path)
