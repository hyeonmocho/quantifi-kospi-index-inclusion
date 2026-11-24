"""Update the existing final notebook, daily holdings, and manifests."""
from pathlib import Path
import json, shutil, hashlib, importlib.util
import pandas as pd
import numpy as np
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'최종'
OUT=ROOT/'04_results/kospi_pair_nov20_update_20261122'
DEST=PKG/'04_results'/OUT.name
NOTE=PKG/'03_notebooks/KOSPI200_KOSPI100_전략_제출용.ipynb'
FNGUIDE_MARKET=False

def export(verify_previous_snapshot=True):
    backup=OUT/'submission_backup';backup.mkdir(exist_ok=True)
    for p in [NOTE,*PKG.glob('*.csv'),*PKG.glob('*.md')]:
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    DEST.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
        elif p.name=='broker_comparison':shutil.copytree(p,DEST/p.name,dirs_exist_ok=True)
    data=PKG/'01_data/processed/kospi_pair_fnguide_update_20261122'
    shutil.copytree(ROOT/'01_data/processed/kospi_pair_fnguide_update_20261122',data,dirs_exist_ok=True)
    shutil.copy2(ROOT/'01_data/processed/fnguide/CD_91.csv',data/'CD_91.csv')
    shutil.copy2(ROOT/'01_data/processed/kospi_pair_nonfnguide_20261122/KRX_raw_market_observations.parquet',data/'KRX_raw_market_observations.parquet')
    exp=PKG/'02_code/export_daily_portfolio.py'
    text=exp.read_text(encoding='utf-8').replace('04_results/kospi_pair_policy_20261121','04_results/kospi_pair_nov20_update_20261122')
    if FNGUIDE_MARKET:
        text=text.replace('04_results/kospi_pair_nov20_update_20261122',f'04_results/{OUT.name}')
        text=text.replace('04_results/kospi_pair_fnguide_20261123',f'04_results/{OUT.name}')
    if 'fresh = pd.read_csv' not in text:
        text=text.replace("names = primary.combine_first(fallback)","fresh = pd.read_csv(ROOT / '01_data/processed/kospi_pair_fnguide_update_20261122/KOSPI200_names.csv').set_index('code')['name']\n    names = primary.combine_first(fallback).combine_first(fresh)")
    exp.write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('updated_export',exp);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if verify_previous_snapshot:
        mod.main()
    else:
        mod.main(verify_previous_snapshot=False)
    pos=pd.read_csv(PKG/'포트폴리오_일별_실제보유.csv',parse_dates=['date'],dtype={'종목코드':str})
    fc=pd.read_csv(DEST/'all_forecasts.csv',parse_dates=['entry_date'])
    sig=fc[fc.entry_date.between('2006-01-01','2026-11-20')][['entry_date','index_id','event','lead']].rename(columns={'entry_date':'date'})
    names={'KOSPI200':'KOSPI200 정기 신규편입 예측','KOSPI100':'KOSPI100 정기 신규편입 예측'}
    sig['전략신호']=sig.index_id.map(names)
    stock=pos[pos['자산구분'].eq('주식')&pos.date.isin(sig.date)].merge(sig,on=['date','전략신호'],how='left',validate='many_to_one')
    orders=pd.concat([pd.read_parquet(DEST/f'{p}_orders.parquet') for p in ['IS','OOS']])
    orders['전략신호']=orders.index_id.map(names)
    buys=orders[orders.side.eq('buy')].groupby(['date','전략신호','code']).notional.sum().rename('당일매수액원').reset_index().rename(columns={'code':'종목코드'})
    stock=stock.merge(buys,on=['date','전략신호','종목코드'],how='left',validate='one_to_one')
    stock['당일매수액원']=stock['당일매수액원'].fillna(0)
    stock['신호시점']=stock.lead.map(lambda x:f'T-{int(x)}' if pd.notna(x) else '이날 신규 신호 없음')
    stock=stock.rename(columns={'event':'신호회차'})
    cols=['date','종목명','종목코드','비중','전략신호','검증구간','신호회차','신호시점','종목평가액원','통합계좌자산원','당일매수액원']
    stock[cols].to_csv(PKG/'포트폴리오_신호일_실제보유.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    sig.drop(columns='전략신호').rename(columns={'index_id':'전략','event':'회차','lead':'남은거래일'}).to_csv(PKG/'신호일_목록.csv',index=False,encoding='utf-8-sig')
    pos[pos.date.eq('2026-11-20')].to_csv(PKG/'포트폴리오_20261120.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    print('portfolio export complete',flush=True)

def notebook():
    b=nbformat.read(NOTE,4)
    b.cells[0].source="""# KOSPI200·KOSPI100 신규 편입 예측 전략

2026년 11월 22일 갱신 · **평가 기준일 2026년 11월 20일**. 추세보정 엔진과 80·60·40·20거래일 전 4회 매수 원칙을 유지했습니다. 각 전략의 고정 주문 예산은 100억 원, 통합 초기 원금은 200억 원입니다.

공표 명단에서 빠진 종목은 다음 거래일 종가부터, 편입된 종목은 패시브 리밸런싱일부터 거래대금 한도 내에서 청산합니다. 공표 결과가 미확인인 종목은 실패로 간주하지 않습니다. 2026년 하반기 원장은 당시 이용 가능 정보를 기준으로 **지금 재구성한 모의 원장**이며 실제 증권계좌 또는 당시 저장한 전진 주문 기록이 아닙니다."""
    b.cells[1].source=b.cells[1].source.replace('kospi_pair_policy_20261121','kospi_pair_nov20_update_20261122')
    b.cells[6].source=b.cells[6].source.split('이번 갱신에서도')[0].rstrip()+'''

이번 갱신에서도 IS에서 선택한 4회 진입을 유지했습니다. OOS의 1~4회 진입 비교는 민감도 분석이며 주전략 재선택에 사용하지 않습니다. 단독 IS 시작은 KOSPI200 2004년, KOSPI100 2006년이며 종료는 2022년 말입니다. 공통 OOS는 2023-01-02~2026-11-20입니다.'''
    z=pd.read_csv(DEST/'metrics.csv').query("phase=='OOS'").set_index('index_id')
    table=['| OOS · 2023-01-02~2026-11-20 | KOSPI200 계정 | KOSPI100 계정 | 통합계좌 |','|---|---:|---:|---:|']
    for label,col,fmt in [('종료 자산(억원)','ending_equity_krw','eok'),('순이익(억원)','profit_krw','eok'),('누적 순수익률','total_return','pct'),('CAGR','cagr','pct'),('연율 변동성','volatility','pct'),('샤프(CD 차감)','sharpe','num'),('MDD','mdd','pct')]:
        vals=[z.at[i,col] for i in ['KOSPI200','KOSPI100','COMBINED']]
        ss=[f'{v/1e8:.2f}' if fmt=='eok' else f'{v:.2%}' if fmt=='pct' else f'{v:.3f}' for v in vals]
        table.append('| '+label+' | '+' | '.join(ss)+' |')
    b.cells[7].source='''### 11월 20일까지 확장한 통합 주문 원장

아래 두 계정은 동일 종목·동일 거래일의 거래대금 한도를 공유합니다. 뒤의 단독 성과는 각 전략을 별도로 실행한 값입니다. 매매비용·미체결 현금·열린 포지션의 평가손익을 포함하며 미래 청산 비용은 아직 발생하지 않았습니다.

'''+ '\n'.join(table)+'''

과거 KOSPI100 OOS 7회는 자체 공표일이 완전히 인증되지 않아 같은 회차 KOSPI200 공표일을 대용한 기존 가정이 남아 있습니다. 2026년 12월 회차는 이 가정을 소급 적용하지 않고, 11월 20일까지 확인된 공표 정보만 청산 판단에 사용합니다.'''
    b.cells[14].source=b.cells[14].source.split('아래 승률·손익비는')[0].rstrip()+'''

아래 승률·손익비는 **청산을 마친 7회**만 집계합니다. 2026년 12월은 진입 후 아직 청산 전인 8번째 회차입니다. 계좌 NAV에는 포함하지만 완료 회차 승률에는 넣지 않습니다.'''
    b.cells[18].source='''### 2026년 하반기 목표·주문·일별 보유 원장

새 FnGuide 구성종목 여부·유동주식비율과 KRX 시세로 7월 3일·8월 3일·9월 1일·10월 1일부터의 분할매수를 재계산했습니다. 신호는 각각 직전 거래일 정보로 산출합니다. 최신 구성종목 여부를 과거 예측의 정답으로 사용하지 않습니다.

사용자 지정 정보 정책에 따라 네 신호는 당시 GICS 분류를 사용했습니다. KRICS 공개 뒤에는 새 분류를 사용할 수 있으나 이미 끝난 신호를 소급 변경하거나 다섯 번째 매수를 만들지 않습니다.

KOSPI200은 11월 18일 공표 명단으로 11월 19일부터 탈락주 매도를 판단합니다. KOSPI100의 12월 변경 확정 명단은 11월 20일까지 확보하지 못해 미확정 상태로 유지합니다. 관리종목 상태의 9월 7일 이후 공백과 한화 계열 분할 종목의 심사기간 확인 공백은 남아 있습니다. 따라서 하반기는 자료 한계가 있는 연구 재구성입니다.

신규 파일의 과거 구성종목 여부 일부는 기존 당시 명단과 다릅니다. 예를 들어 2006년 3월 KOSPI200은 새 파일에 198개만 표시되어 과거 공식·기존 검증 명단을 일괄 덮어쓰지 않았습니다. 과거 확정 입력은 유지하고 하반기 자료를 연결했습니다. 유동비율 10% 경계 변화는 일성신약 2023년 네 신호에서 확인했으나 재선정 결과 최종 명단 변화는 없었습니다. 상세 대조는 historical_input_signal_audit.csv에 남겼습니다.'''
    b.cells[25].source='''## 전체 거래일 모의 포트폴리오

**2006-01-02~2026-11-20**의 매 거래일 종가 보유 원장입니다. 비중은 통합계좌 자산 대비 평가액이며 현금(CASH)을 포함하면 매일 합계가 1입니다. 같은 종목을 두 전략이 보유하면 전략별 행을 유지합니다. 주식이 없는 날도 현금 행을 저장합니다.

포트폴리오_일별_실제보유.csv가 전체 보유 DF이며 포트폴리오_신호일_실제보유.csv는 신호일만 추린 보조 자료입니다. 파일명의 실제보유는 모의 체결 후 남아 있는 포지션을 의미하며 실계좌 기록이 아닙니다. IS·OOS는 별도 초기화한 계좌입니다.'''
    for i in [8,10]:b.cells[i].source=b.cells[i].source.replace('strategy_events','review_events')
    b.cells=b.cells[:27]+[nbformat.v4.new_markdown_cell('''## 갱신 자료와 미청산 회차 점검

마지막 날 KOSPI100 100개, KOSPI200 201개가 관측됩니다. 하반기 손익에는 열린 포지션 평가손익이 들어갑니다. 차수별 목표 수, 체결금액과 잔여 현금은 아래 자료에서 확인합니다.'''),nbformat.v4.new_code_cell("""import json
display(pd.DataFrame(json.loads((OUT/'input_audit.json').read_text(encoding='utf-8')))[['field','first','last','codes','latest_members']])
display(pd.read_csv(OUT/'H2_contribution.csv')[['index_id','H2_pnl','H2_account_return','H2_return_per_fixed_budget','fill_ratio','distinct_codes']].style.format({'H2_account_return':'{:+.2%}','H2_return_per_fixed_budget':'{:+.2%}','fill_ratio':'{:.2%}'}))
display(pd.read_csv(OUT/'forecast_audit.csv')[['index_id','lead','information_date','selected','parent_selected','management_source_date']])
print('왕복 거래비용 민감도 (동일 종료일)')
show(pd.read_csv(OUT/'cost_sensitivity_oos.csv')[['index_id','roundtrip_bp','total_return','cagr','sharpe','mdd']])
""")]
    if FNGUIDE_MARKET:
        b.cells[0].source=b.cells[0].source.replace('11월 22일 갱신','11월 23일 갱신')
        for c in b.cells:
            c.source=c.source.replace('kospi_pair_nov20_update_20261122', OUT.name)
        b.cells[18].source=b.cells[18].source.replace('및 KRX 시세','및 FnGuide 시세').replace(
            '와 KRX 시세로','와 FnGuide 수정주가·종가·거래대금으로')
        b.cells[18].source+='\n\n2026년 하반기 시가총액은 같은 날의 수정 전 종가와 상장주식수를 곱해 계산했습니다. 수정주가는 20거래일 추세, 수정 전 종가와 거래대금은 체결·평가에 사용했습니다. 상반기까지는 기존에 검증한 입력과 기업행위 처리 이력을 연결했습니다. KRX 자료는 시장 구분, 공표, 업종 및 기업행위 대조에 사용합니다.'
        b.cells[-1].source+="\ndisplay(pd.DataFrame(json.loads((OUT/'market_input_audit.json').read_text(encoding='utf-8')))[['field','csv','last','last_valid']])"
        if (DEST/'four_factor_summary.csv').exists():
            b.cells.extend([
                nbformat.v4.new_markdown_cell('## 시장·규모·가치·모멘텀 조정 알파\n\n2026년 11월 20일까지 현금 보유일을 포함한 비용 차감 수익률을 분석합니다. CD91을 차감하고 HAC 20거래일 양측검정을 사용했습니다. 연율 알파는 일별 회귀 절편에 252를 곱한 값입니다. 전체기간은 IS·OOS의 일별 수익률을 이어 사용하며, 각 구간의 초기자금 재설정은 수익률에 포함하지 않습니다.'),
                nbformat.v4.new_code_cell("a=pd.read_csv(OUT/'four_factor_summary.csv')\ndisplay(a.style.format({'연율 알파':'{:.2%}','95% 하한':'{:.2%}','95% 상한':'{:.2%}','t값':'{:.3f}','p값':'{:.4f}'}))")
            ])
    nbformat.write(b,NOTE)
    NotebookClient(b,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(PKG)}}).execute()
    assert not any(o.get('output_type')=='error' for c in b.cells if c.cell_type=='code' for o in c.get('outputs',[]))
    nbformat.write(b,NOTE)
    print('notebook executed',flush=True)

def docs():
    s=json.loads((DEST/'daily_portfolio_export_audit.json').read_text(encoding='utf-8'))
    text=f"""# 11월 20일 데이터 갱신 · 2026-11-22

- 제출 노트북: 03_notebooks/KOSPI200_KOSPI100_전략_제출용.ipynb (전체 셀 실행 완료)
- 성과·체결·일별 원장: 04_results/{OUT.name}/
- 전체 보유: 포트폴리오_일별_실제보유.csv (현금 포함), 포트폴리오_일별_주식보유.csv (주식만)
- 최신 스냅샷: 포트폴리오_20261120.csv
- 기간: {s['first']}~{s['last']}, {s['trading_days']:,}거래일, 현금 포함 {s['all_rows']:,}행.
- 종목별 비중은 통합계좌 순자산 대비입니다. 두 전략의 동일 종목은 전략별로 행을 나눕니다. IS·OOS는 독립 초기화 계좌입니다.
- 전략당 고정 주문 예산 100억원, 총 200억원. 이익은 주문 예산에 재투자하지 않습니다.
- 4회 분할, 최대 10거래일 매수, 당일·직전 20일 평균 거래대금의 각 10% 중 작은 한도 공유, 편도 25bp 유지.
- 단독 IS: KOSPI200 2004년부터, KOSPI100 2006년부터 2022년 말. OOS: 2023-01-02~2026-11-20.
- 완료 7회와 미청산 2026년 12월 회차를 구분합니다. 현재 성과에는 평가손익이 포함되고 미래 청산비용은 미포함입니다.
- 하반기 원장은 사후 재구성한 모의 기록입니다. KOSPI100 12월 공표, 최신 관리종목 상태, 일부 분할 심사 근거의 공백은 노트북에 명시했습니다.
- 새 파일의 과거 명단 불일치는 기록하고 기존 역사적 입력을 일괄 덮어쓰지 않았습니다. 유동비율 경계 변화의 재선정 영향은 없었습니다.
- PPT 파일은 이번 요청에서 수정하지 않았습니다. 최신 수치는 이 노트북·원장이 기준입니다.

제출 폴더에서 노트북 전체 실행으로 표·그래프를 재현할 수 있습니다. 일별 CSV 재생성은 python 02_code/export_daily_portfolio.py입니다.
전체 원장 재계산은 원본 프로젝트의 02_code/modules/update_pair_submission_20261122.py를 사용합니다. 제출 폴더는 결과표 재현·감사 범위이며 원본 전체 연구 환경과 동일한 독립 엔진 패키지는 아닙니다.
"""
    (PKG/'README_제출안내.md').write_text(text,encoding='utf-8')
    if FNGUIDE_MARKET:
        text=text.replace('2026-11-22','2026-11-23').replace('update_pair_submission_20261122.py를 사용합니다','refresh_pair_fnguide_20261123.py --replay를 사용합니다')
        text+='\n하반기 시장 데이터는 종목코드 기준 FnGuide 종가·수정주가·거래대금·상장주식수를 사용했습니다. 일별 시총은 수정 전 종가×상장주식수로 계산했습니다. 이전 구간 자료와 공식 공표·업종·기업행위 이력을 연결합니다.\n'
        (PKG/'README_제출안내.md').write_text(text,encoding='utf-8')
    name='작업업데이트_20261123_FnGuide반영.md' if FNGUIDE_MARKET else '작업업데이트_20261122_1120자료반영.md'
    (ROOT/'00_docs'/name).write_text(text,encoding='utf-8')
    rows=[]
    for p in sorted(PKG.rglob('*')):
        if p.is_file() and p.name!='파일목록_SHA256.csv' and not p.name.startswith('~$') and '__pycache__' not in p.parts:
            rows.append(dict(file=str(p.relative_to(PKG)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(PKG/'파일목록_SHA256.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    export();notebook();docs()

