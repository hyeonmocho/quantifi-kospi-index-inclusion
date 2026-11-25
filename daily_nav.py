"""발행주식 수 및 지수 편입 저장소의 일별 NAV를 기준일 종가 1로 표시한다."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import dates as md, font_manager

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '보고서'
BASE = pd.Timestamp('2026-09-01')
COLORS = ['#922d3e','#197867','#52658d']


def checked(data):
    if not data.index.is_monotonic_increasing or not data.index.is_unique or BASE not in data.index:
        raise ValueError('기준일이 없거나 거래일이 중복 또는 역순입니다.')
    if not np.isfinite(data).all().all() or not data.gt(0).all().all():
        raise ValueError('일별 평가액을 확인하세요.')
    return data.loc[BASE:]


def draw(values, phase, returns):
    values = checked(values)
    rel = values.div(values.loc[BASE])
    result = pd.concat([phase.rename('구간'),rel.add_suffix('_NAV'),returns.add_suffix('_일수익률')],axis=1).loc[rel.index]
    post = phase.index[phase.eq('후속 보유분 평가')]
    if len(post):
        result.loc[post,'후속첫날종가1_통합NAV'] = values.loc[post,'통합']/values.at[post[0],'통합']
    OUT.mkdir(exist_ok=True)
    for family in ('Malgun Gothic','Noto Sans CJK KR','AppleGothic'):
        if any(f.name == family for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family']=family
            break
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['svg.fonttype']='path'
    plt.rcParams['svg.hashsalt']='quantifi'
    fig,ax=plt.subplots(figsize=(10.5,4.8))
    fig.subplots_adjust(left=.10,right=.97,top=.77,bottom=.19)
    title='발행주식 수 롱숏' if len(values.columns)==1 else 'KOSPI200·100 편입 예측'
    fig.text(.10,.91,title+'   |   일별 모의 NAV',fontsize=18,fontweight='bold',color='#202738')
    fig.text(.10,.837,f'2026.09.01 종가 = 1     /     최종 관측 {values.index[-1]:%Y.%m.%d}',fontsize=10,color='#49566b')
    ax.axhline(1,color='#737982',lw=.8,ls=':')
    for i,col in enumerate(values):
        line=rel[col]
        end=post[0] if len(post) else line.index[-1]+pd.Timedelta(days=1)
        back=line[line.index<end]
        ax.plot(back.index,back,color=COLORS[i],lw=2.4 if i==0 else 1.5,label=col)
        if len(post):
            live=line.loc[back.index[-1]:]
            ax.plot(live.index,live,color=COLORS[i],ls='--',lw=2 if i==0 else 1.4)
        ax.scatter(line.index[-1],line.iloc[-1],s=18,color=COLORS[i],zorder=3)
    if len(post):
        ax.axvspan(post[0],rel.index[-1],color='#d9e0e7',alpha=.45)
    ax.xaxis.set_major_locator(md.AutoDateLocator(minticks=4,maxticks=8))
    ax.xaxis.set_major_formatter(md.DateFormatter('%m.%d'))
    ax.grid(axis='y',color='#e0e2e6')
    ax.spines[['top','right','left']].set_visible(False)
    ax.legend(frameon=False,loc='best',ncol=len(values.columns))
    ax.set_ylabel('NAV')
    fig.text(.10,.055,'점선은 마지막 보유분의 종가 평가 · 신규 주문과 청산은 다음 집행 원장 갱신 때 반영' if len(post)
        else '각 거래일 종가 평가 · 첫날 일수익률은 CSV에 별도로 표시',fontsize=9,color='#536070')
    pic=OUT/'일별_NAV.svg'
    tmp=pic.with_suffix('.tmp')
    fig.savefig(tmp,format='svg',metadata={'Date':None})
    plt.close(fig)
    tmp.replace(pic)
    target=OUT/'일별_NAV.csv'
    tmp=target.with_suffix('.tmp')
    result.to_csv(tmp,index_label='날짜',encoding='utf-8-sig',float_format='%.10f')
    tmp.replace(target)
    print(result.tail(1).to_string())

def evaluate():
    from market_close import market_dates
    import sys
    folder=ROOT/'04_results/kospi_pair_optimized_trend_20261124'
    records=pd.read_parquet(folder/'OOS_daily.parquet')
    names={'COMBINED':'통합','KOSPI200':'KOSPI200','KOSPI100':'KOSPI100'}
    values = records.pivot(index='date',columns='index_id',values='equity')[list(names)].rename(columns=names)
    returns = records.pivot(index='date',columns='index_id',values='return_daily')[list(names)].rename(columns=names)
    boundary=values.index[-1]
    phase=pd.Series('기존 OOS',index=values.index)
    saved=OUT/'일별_NAV.csv'
    if '--daily' not in sys.argv and saved.exists():
        old=pd.read_csv(saved,index_col='날짜',parse_dates=True)
        if boundary in old.index and np.allclose(
            old.loc[boundary,[c+'_NAV' for c in values]].astype(float),
            values.loc[boundary]/values.loc[BASE],rtol=1e-8):
            for day,row in old.loc[old.index>boundary].iterrows():
                if row['구간']!='후속 보유분 평가':
                    raise ValueError('기존 후속 구간의 출처를 확인하세요.')
                values.loc[day]=[row[c+'_NAV']*values.at[BASE,c] for c in values]
                returns.loc[day]=[row[c+'_일수익률'] for c in returns]
                phase.at[day]=row['구간']
    if '--daily' in sys.argv:
        cache=ROOT/'비공개자료/종가캐시'
        market=market_dates(boundary-pd.offsets.BDay(10),os.environ['KRX_KEY_FILE'],cache)
        if boundary not in market:
            raise ValueError('마지막 원장의 공식 기준 종가 미확인')
        held=pd.read_parquet(folder/'OOS_holdings.parquet')
        amounts=held.loc[held.date.eq(boundary)].pivot_table(index='code',columns='index_id',values='market_value',aggfunc='sum',fill_value=0)
        amounts.index=amounts.index.str.removeprefix('A')
        cash=records.loc[records.date.eq(boundary)].set_index('index_id').cash
        lots=pd.read_parquet(folder/'OOS_lots.parquet')
        exits=pd.to_datetime(lots.loc[lots.actual_exit.isna(),'planned_exit']).dropna()
        for col in amounts:
            if not np.isclose(amounts[col].sum()+cash[col],values.at[boundary,col],rtol=1e-9):
                raise ValueError('포지션과 현금이 기말 평가액과 일치하지 않습니다.')
        prev=market[boundary].TDD_CLSPRC.reindex(amounts.index)
        if prev.isna().any() or not prev.gt(0).all():
            raise ValueError('보유분 기초 종가 누락')
        for d,m in market.items():
            if d<=boundary:
                continue
            if not exits.empty and d>=exits.min():
                print('예정 청산일 도달 · 갱신된 집행 원장이 필요합니다.')
                break
            current=m.reindex(amounts.index)
            ref=current.TDD_CLSPRC-current.CMPPREVDD_PRC
            if not current.TDD_CLSPRC.gt(0).all() or not np.allclose(ref,prev):
                raise ValueError(f'{d.date()} 보유분 가격 또는 기업행위 확인 필요')
            equity=amounts.mul(current.TDD_CLSPRC/prev,axis=0)
            daily=equity.sum()+cash.reindex(equity.columns)
            daily['통합']=daily.sum()
            values.loc[d]=daily
            returns.loc[d]=values.iloc[-1]/values.iloc[-2]-1
            phase.at[d]='후속 보유분 평가'
            amounts,prev=equity,current.TDD_CLSPRC
    draw(values,phase,returns)

if __name__=='__main__':
    evaluate()
