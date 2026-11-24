"""전략 A 진입 필터. 미래 가격/실제 편입 결과를 진입 판단에 사용하지 않는다."""
from events_data import *

LISTINGS = pd.read_csv(PROC/'kind_listing.csv', parse_dates=['list_date'])
LISTINGS = LISTINGS[(LISTINGS.market=='유가증권시장') & LISTINGS.list_type.isin(['신규상장','이전상장'])]
TV = D['tv'].reindex(S)
MC = mc.reindex(S)
MARKET = pd.read_pickle(PROC/'market_monthly.pkl').reindex(S,method='ffill').shift(1)
_calendar=pd.to_datetime(pd.read_excel(CAL,sheet_name='거래일').iloc[:,0],errors='coerce').dropna()
def scope_expiry(y,m):
    """미래 만기를 가격자료 마지막 날로 잘라내지 않는다."""
    d=pd.Timestamp(dt.date(y,m,1))
    d+=pd.Timedelta(days=(3-d.weekday())%7+7)
    if d<=S[-1]: return on_or_before(d)
    if len(_calendar) and _calendar.min()<=d<=_calendar.max():
        return _calendar[_calendar<=d].max()
    return d  # 거래일 캘린더 범위 밖: 명목 둘째 목요일, 실행 전 갱신 필요.
EXPIRIES=sorted({scope_expiry(y,m) for y in range(2000,S[-1].year+2) for m in (3,6,9,12)})
# 이전상장은 정기편입 배제 사유가 아니다. 아래 종목은 감사표 대상일 뿐이다.
TRANSFER_SCOPE = 'special_only'
TRANSFER_REVIEW_CODES = {
    'A034120':'SBS','A035250':'강원랜드','A024110':'기업은행','A032390':'KTF',
    'A035420':'NAVER','A035720':'카카오','A068270':'셀트리온','A022100':'포스코DX'}

# 출처와 정확도는 rebuild_a_scope가 final/special_rule_regimes.csv에도 기록한다.
# KRX KOSPI200 방법론(2021.07) 주요 연혁에서 30→15일 시행일을 확인했다.
# 기존 발표 다음날(2014-12-17) 잠정 경계는 폐기; 연구 감사표에 영향 보존.
OBSERVATION_CHANGE = pd.Timestamp('2015-01-26')
RANK_RULE_START = pd.Timestamp('2017-06-09')  # 2017-03-29 발표, 6월 정기변경부터
FLOAT_RULE_START = pd.Timestamp('2024-11-01')  # 2025-01-15 유진증권: 전년 11월 시행
RULE_REGIMES = [
    dict(start='2000-01-01', end='2015-01-25', days=30, size_rule='시장 평균 시가총액 1% 초과',
         source='https://kind.krx.co.kr/external/2012/02/07/000185/20120207000535/10131.htm',
         supplementary_source='https://www.etnews.com/200901140309',
         limitation='2012 공시 및 2009 당시 설명에 근거한 과거 규칙 복원; 2000~2008 원 규정 연혁 미완전'),
    dict(start='2015-01-26', end='2017-06-08', days=15, size_rule='시장 평균 시가총액 1% 초과',
         source='https://download.kiwoom.com/web/KiwoomEtnRuleBook2.pdf',
         supplementary_source='KRX KOSPI200 방법론 2021.07 주요 연혁: 2015-01-26',
         limitation='시행일 확인; 당시 시장 구성·상장 구분 자료의 불완전성은 별개'),
    dict(start='2017-06-09', end='2024-10-31', days=15, size_rule='보통주 평균 시가총액 상위 50위',
         source='https://www.yna.co.kr/amp/view/AKR20170329162400008',
         supplementary_source='https://www.ktb.co.kr/common/download.jspx?cmd=viewPDF&path=%2Fattach_file%2FRESEARCH%2F63811%2F1%2F20170712_035720_leemina_75.pdf',
         limitation='예상 순위는 확정 편입이 아님; 산업군·유동성 및 거래소 재량의 완전 복원 아님'),
    dict(start='2024-11-01', end='', days=15, size_rule='상위 50위 및 유동시총 >= 50위 시총의 절반',
         source='https://kind.krx.co.kr/external/2024/09/30/000092/20240930000179/68323.htm',
         supplementary_source='https://stock.pstatic.net/stock-research/market/21/20241016_market_572060000.pdf',
         limitation='유동시총 15일 평균의 당시 확정 자료 미구축: 50위 충족 시 거래 보류·현금, 검토필요 상태')]

def next_regular_te(t):
    """t 다음의 정기 리밸런싱 종가일. 2019년까지 6월, 이후 6·12월."""
    for y in range(t.year,t.year+2):
        for m in ((6,) if y<2020 else (6,12)):
            d=scope_expiry(y,m)
            if d>t: return d
    return pd.NaT

def _observation_ready_date(listed, days):
    """관찰기간 마지막 날 이후 2거래일. 미래 가격이 아닌 거래일 달력만 사용."""
    sessions=S.union(pd.DatetimeIndex(_calendar)).sort_values().unique()
    future=sessions[sessions>=listed]
    if len(future)>days+1: return future[days+1]
    return pd.NaT

def special_signal_details(code, idx, ip, observation_change=None):
    """특례 '예측' 근거를 반환한다. 미래 편입 여부는 읽지 않는다.

    이전상장 자체나 최초 K200 편입이라는 이유로 A에서 제외하지 않는다.
    특례 관찰요건을 예상 충족하고 그 만기가 다음 정기변경보다 늦지 않은 경우만
    B_1로 예약한다. 과거 제도는 당시 시장시총 1% 기준으로 분리한다.
    """
    t=S[ip]
    result=dict(reason='', status='not_a_new_listing', signal_date=t,
                observation_days=np.nan, observed_days=np.nan, rank=np.nan,
                market_share=np.nan, market_members=np.nan, valid_market_members=np.nan,
                market_coverage=np.nan, expected_special_te=pd.NaT,
                regular_te=next_regular_te(t), list_date=pd.NaT, list_type='')
    if idx!='KOSPI200': return result
    known=LISTINGS[(LISTINGS.code==code)&(LISTINGS.list_date<=t)].sort_values('list_date')
    if known.empty: return result
    r=known.iloc[-1]
    result.update(list_date=r.list_date,list_type=r.list_type)
    boundary=OBSERVATION_CHANGE if observation_change is None else pd.Timestamp(observation_change)
    days=30 if t<boundary else 15
    i1=int(S.searchsorted(r.list_date)); ik=min(ip,i1+days-1); k=ik-i1+1
    result.update(observation_days=days, observed_days=k)
    ready=_observation_ready_date(r.list_date,days)
    exp=next((d for d in EXPIRIES if pd.notna(ready) and d>=ready),pd.NaT)
    result['expected_special_te']=exp
    if pd.isna(exp):
        result.update(status='special_calendar_unverified',reason='needs_special_calendar_review'); return result
    if exp<=t:
        result['status']='special_window_expired_regular_eligible'; return result
    if pd.isna(result['regular_te']) or exp>result['regular_te']:
        result['status']='no_special_window_before_regular'; return result
    if code not in MC or k<1:
        result.update(status='special_size_data_missing',reason='needs_special_size_data_review'); return result
    observed=MC.iloc[i1:ik+1][code]
    if len(observed)!=k or not (np.isfinite(observed)&observed.gt(0)).all():
        result.update(status='special_size_data_missing',reason='needs_special_size_data_review'); return result
    own=(observed.sum()+(days-k)*observed.iloc[-1])/days
    others=MC.iloc[i1:ik+1].mean()
    eligible=MARKET.iloc[ik].reindex(MC.columns).eq('유가증권시장')
    eligible.loc[code]=True
    if not np.isfinite(own) or own<=0:
        result.update(status='special_size_data_missing',reason='needs_special_size_data_review'); return result
    comparison=others[eligible]
    valid_comparison=np.isfinite(comparison)&comparison.gt(0)
    result.update(market_members=int(eligible.sum()),valid_market_members=int(valid_comparison.sum()),
                  market_coverage=float(valid_comparison.mean()) if len(comparison) else np.nan)
    # K200 구성 수보다도 작은 시장 단면은 순위/전체시총 판정용으로 사용할 수 없다.
    if valid_comparison.sum()<200:
        result.update(status='special_market_data_insufficient',reason='needs_special_size_data_review'); return result
    rank=int((comparison>own).sum())+1
    total=others[eligible].sum(min_count=1)-others.get(code,0)+own
    share=own/total if pd.notna(total) and total>0 else np.nan
    result.update(rank=rank,market_share=share)
    if not np.isfinite(share):
        result.update(status='special_market_data_insufficient',reason='needs_special_size_data_review'); return result
    if t<RANK_RULE_START:
        if share<=0.01:
            result['status']='legacy_size_threshold_not_met_regular_eligible'; return result
        result.update(reason='B_1_pending_market_share_gt1pct',status='legacy_special_size_candidate')
    elif rank<=50:
        if t>=FLOAT_RULE_START:
            # 현재 float_shares와 반기 상장주식수만으로 확정 15일 유동시총을 재현했다고
            # 주장하지 않는다. 불완전한 50위 조건만으로 정기후보를 지우지 않는다.
            result.update(status='post2024_float_requirement_unverified',reason='needs_special_rule_review'); return result
        result.update(reason='B_1_pending_rank_le50',status='special_rank_candidate')
    else:
        result['status']='rank_threshold_not_met_regular_eligible'
    return result

def strategy_a_exclusion(code, idx, ip):
    return special_signal_details(code,idx,ip)['reason']

def exclusion(code, idx, ip, ib):
    """종가 체결 모형: 신호일 자료와 주문 당일 가격·거래대금 존재 확인.

    이전상장 정기편입 후보는 유지. 당시 특례 시총요건을 예상 충족하고
    다음 정기 리밸런싱까지 도래하는 특례 만기를 기다리는 후보만 B_1로 예약한다.
    필요한 특례 판단 자료가 불충분한 경우 후보를 남기고 거래는 현금 보류한다.
    """
    j=CI.get(code)
    if j is None or ib>=len(S): return 'missing_entry_data'
    vals=[PXV[ip,j],PXV[ib,j],MC.iloc[ip].get(code),TV.iloc[ip].get(code),TV.iloc[ib].get(code)]
    if not all(pd.notna(v) and np.isfinite(v) and v>0 for v in vals):
        return 'missing_or_nontradable_entry'
    return strategy_a_exclusion(code,idx,ip)

def apply_eligibility(book):
    b=book.copy(); reasons=[]
    for r in b.itertuples():
        reasons.append(exclusion(r.code,r.idx,int(r.ib)-1,int(r.ib)))
    b['exclusion_reason']=reasons; b['traded']=b.exclusion_reason.eq('')
    b['original_r']=b.r
    b.loc[~b.traded,'r']=0.0
    # 보유 중 결측은 미래를 보고 진입 제외하지 않고 미평가로 남긴다.
    b['valuation_missing']=False
    for r in b[b.traded].itertuples():
        if not np.isfinite(RT[int(r.ib)+1:int(r.sell)+1,CI[r.code]]).all():
            b.loc[r.Index,'valuation_missing']=True
            b.loc[r.Index,'r']=np.nan
    b['cost']=np.where(b.traded,COST,0.)
    b['r_net']=b.r-b.cost
    b['weight']=1/b.groupby(['idx','시기']).code.transform('size')
    return b

def aggregate(book):
    rows=[]
    for (idx,e,s),g in book.groupby(['idx','시기','표본']):
        active=g[g.traded]
        rows.append(dict(idx=idx,시기=e,표본=s,ret=g.r_net.mean() if g.r_net.notna().all() else np.nan,
                         n=len(active),candidate_n=len(g),cash_weight=1-g.loc[g.traded,'weight'].sum(),
                         적중률=active.hit.mean(),unpriced=int(g.valuation_missing.sum())))
    return pd.DataFrame(rows)
