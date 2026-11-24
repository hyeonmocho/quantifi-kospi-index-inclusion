"""KOSPI200·KOSDAQ150 정기변경 규칙 재현 엔진 — 심사기준일까지의 데이터로 KRX 선정 규칙을 적용해 편입·편출 종목을 예측.

KOSPI200 (KRX 방법론, 거래소 2019-12 보도자료, 메리츠 2023-03-29)
  심사 대상: 유가증권시장 보통주. 관리종목·정리매매, 펀드·외국주권, 리츠·선박·인프라 투자회사·스팩, 유동주식비율 10% 미만 제외
             상장 6개월(2019년까지 1년) 미만 제외 — 코스닥 이전상장은 코스닥 기간 합산, 기존 구성종목(분할 신설법인 등)은 예외
  산업군: GICS 참고 10개 (에너지, 소재, 산업재, 자유소비재, 필수소비재, 헬스케어, 금융/부동산, 정보기술, 커뮤니케이션서비스, 유틸리티)
          → 거래소 공표 GICS(S&P 제공, KRX 인덱스 수집, 2010-10~)를 기본으로 사용(scheme='GICS'). 비교용으로 WICS·FnGuide IG27
          그날 이전에 공표된 분류만 사용: 거래소 GICS 없으면 WICS(2010~), 그것도 없으면 FnGuide 업종을 미리 고정한 대응표로 변환
          (2010년 10월 전 회차는 FnGuide 업종 기반이라 정확도가 낮음)
  미래 참조 차단: 관리종목·상장폐지·소속 시장(KIND 상장일로 월중 반영)·유동비율(유동주식수 ÷ 직전 거래소 스냅샷 상장주식수)·
          종목명(리츠·스팩 판정)·현재 구성종목은 모두 예측일까지 알려진 값 (i_info)
          (2017년 이전 KRX는 한국표준산업분류 기반 산업군을 써서 IS 초기 기간은 오차가 더 큼)
  심사 기간: 2020~ 직전 6개월(6월 변경 = 11~4월, 12월 변경 = 5~10월) / 2019년까지 직전 1년. 일평균 시가총액·거래대금
  1차: 산업군별 시총 큰 순으로 누적이 처음 85%(2019년까지 80%)에 닿는 종목까지, 거래대금 순위가 산업군 대상 수의 85% 밖이면 제외
  2차: 기존 종목 = 산업군 순위가 기존 구성종목 수의 110% 이내면 잔류 / 신규 = 1차 선정 + 90% 이내
  개수: 200 미달 → 빠진 기존 종목 중 시총 큰 순 / 초과 → 시총 작은 순 제외
  대형주 특례: 최근 15거래일 평균 시총 50위 이내는 편입 (가장 작은 종목 제외로 개수 맞춤)
KOSDAQ150 (공개 자료로 확인된 구조 + 불확실한 기준값은 데이터로 맞춤)
  2020~: 산업군별 같은 구조 (에너지·유틸리티 제외), 누적 비중 C·거래대금 L·버퍼·신규상장 특례 순위는
         IS 회차(2016-06~2022-12)로만 선택 → OOS 2023-06~ 검증 (kosdaq150_calib.py)
  2019년까지: 비기술주(소재·산업재·필수소비재·자유소비재) 산업군별 선정 후 남은 자리를 기술주(IT·헬스케어·커뮤니케이션)로 시총 순 채움
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from events_data import *

RAW = ROOT / '01_data/raw/fnguide'
names = D['names']

# ---- 입력 데이터 (모두 '그날까지 알 수 있었던 값'만 쓰도록: bfill·종목별 마지막 값·현재 이름 사용 금지) ----
MKT = pd.read_pickle(PROC / 'market_monthly.pkl')                       # 상장된 시장 (월말)
sec = pd.read_pickle(PROC / 'sector_ig27.pkl')
back = np.where(np.diff(sec.index.values.astype('int64')) < 0)[0]; sec = sec.iloc[:back[0] + 1]
SEC = sec.ffill()                                                         # FnGuide IG27, 빠진 달은 직전 값 (앞으로 채우기만)
MGT = pd.read_pickle(PROC / 'mgmt_flag.pkl')
FIRST_PX = px.notna().idxmax()                                            # KOSPI·KOSDAQ 첫 거래일 (이전상장은 코스닥 기간 합산과 같음)
MCD, TVD = mc.reindex(S), D['tv'].reindex(S)
FLOAT = pd.read_pickle(PROC / 'float_shares.pkl').reindex(index=S, columns=px.columns) if (PROC / 'float_shares.pkl').exists() else None

# KIND 신규상장기업현황: 상장일·상장유형(신규상장·이전상장·재상장)·증권구분. 상장일 당일부터 공개 정보로 사용
KL = pd.read_csv(PROC / 'kind_listing.csv', parse_dates=['list_date'])
KL = KL[KL.code.isin(px.columns)].sort_values('list_date')

# 거래소 GICS 스냅샷(4·10월 말): 상장주식수(유동비율 분모)·그날 종목명
_kg = pd.read_csv(ROOT / '01_data/raw/krx/gics/krx_gics_by_date.csv', dtype=str)
_kg = _kg[_kg.sel_code.str.len() == 2]
_kg['code'] = 'A' + _kg.isu_cd.str.zfill(6); _kg['date'] = pd.to_datetime(_kg.date)
SHR = _kg.assign(sh=pd.to_numeric(_kg.listshr_cnt.str.replace(',', ''), errors='coerce')).pivot_table(
    index='date', columns='code', values='sh', aggfunc='first').reindex(columns=px.columns)
EXCL = re.compile(r'리츠$|리츠부동산|스팩|기업인수목적|인프라투융자|맥쿼리인프라|선박투자|^바다로')   # '메리츠'·'현대인프라코어'는 걸리지 않게
NON_STOCK = {'투자회사', '선박투자회사', '부동산투자회사', '사회간접자본투융자회사', '외국주권', '주식예탁증권'}
_nm_hist = pd.concat([KL[['list_date', 'code', 'name']].rename(columns={'list_date': 'date'}),
                      _kg[['date', 'code', 'isu_abbr']].rename(columns={'isu_abbr': 'name'})]).sort_values('date')
_nm_hist['ex'] = _nm_hist.name.astype(str).str.contains(EXCL)
EX_HIST = _nm_hist.pivot_table(index='date', columns='code', values='ex', aggfunc='last').reindex(columns=px.columns)
EX_NOW = pd.Series([bool(EXCL.search(str(names.get(c, '')))) for c in px.columns], index=px.columns)
NONSTK_FROM = KL[KL.secu.isin(NON_STOCK)].groupby('code').list_date.min()  # 펀드·리츠 등: 상장일부터 제외
CODE_OK = pd.Series([c[-1] == '0' and not c.startswith('A9') for c in px.columns], index=px.columns)

def base_ok(t):
    """t 시점 보통주 여부. 이름 판정은 t 이전 마지막으로 알려진 이름(KIND 상장 시 이름·거래소 스냅샷), 기록이 없을 때만 현재 이름."""
    rows = EX_HIST.index[EX_HIST.index <= t]
    ex = (EX_HIST.loc[:t].ffill().iloc[-1] if len(rows) else pd.Series(np.nan, index=px.columns)).astype(object)
    ex = ex.where(ex.notna(), EX_NOW).astype(bool)
    nonstk = pd.Series(px.columns.map(lambda c: c in NONSTK_FROM.index and NONSTK_FROM[c] <= t), index=px.columns)
    return CODE_OK & ~ex & ~nonstk

def market_at(t):
    """t 시점 소속 시장. 월말 자료(직전 월말)에 KIND 상장일(신규·이전·재상장)을 날짜 단위로 덮어씀 — 월중 이전상장·신규상장 반영."""
    base_rows = MKT.index[MKT.index <= t]
    m = MKT.loc[base_rows[-1]].reindex(px.columns).copy() if len(base_rows) else pd.Series(np.nan, index=px.columns)
    k = KL[(KL.list_date <= t) & (KL.list_date > (base_rows[-1] if len(base_rows) else pd.Timestamp('1900')))]
    if len(k): m[k.code.values] = k.market.values
    return m

def float_ratio(i):
    """i일 유동비율 = 유동주식수 ÷ (그날 이전 마지막 거래소 스냅샷의 상장주식수). 액면분할 영향 없음, 미래 값 사용 안 함."""
    rows = SHR.index[SHR.index <= S[i]]
    if FLOAT is None or not len(rows): return pd.Series(np.nan, index=px.columns)
    return FLOAT.iloc[i] / SHR.loc[rows[-1]]

# WICS 대분류 (wiseindex 수집, 2010~)
WF = ROOT / '01_data/raw/external/wics/wics_sector.csv'
if WF.exists():
    w = pd.read_csv(WF, dtype={'code': str}).dropna(subset=['code'])
    w['date'] = pd.to_datetime(w.date.astype(str))
    WICS = w.pivot_table(index='date', columns='code', values='sec_nm', aggfunc='first').reindex(columns=px.columns)
else:
    WICS = None
# FnGuide 업종 → 산업군 대응표: 데이터로 맞추지 않고 GICS 정의대로 미리 고정 (미디어는 2018-09 GICS 개편 뒤 커뮤니케이션서비스)
IG_MAP = {'에너지': '에너지', '화학': '소재', '금속및광물': '소재', '기타 소재': '소재',
          '건설': '산업재', '기타자본재': '산업재', '조선': '산업재', '상업서비스': '산업재', '운송': '산업재',
          '자동차및부품': '경기관련소비재', '내구소비재및의류': '경기관련소비재', '소비자서비스': '경기관련소비재', '유통': '경기관련소비재',
          '음식료및담배': '필수소비재', '생활용품': '필수소비재', '의료': '건강관리',
          '은행': '금융', '증권': '금융', '보험': '금융', '기타금융': '금융',
          '반도체': 'IT', '디스플레이': 'IT', '하드웨어': 'IT', '소프트웨어': 'IT',
          '통신서비스': '커뮤니케이션서비스', '유틸리티': '유틸리티'}

# 거래소 공표 GICS (KRX 인덱스 'GICS 산업분류 > 산업별 종목현황', S&P 제공, 2010-10~ 4·10월 말). 이름은 WICS식으로 맞춤
GF = PROC / 'krx_gics.csv'
GICS_NAME = {'자유소비재': '경기관련소비재', '헬스케어': '건강관리', '정보기술': 'IT'}
if GF.exists():
    g = pd.read_csv(GF, dtype=str); g['date'] = pd.to_datetime(g.date)
    GICS = g.pivot_table(index='date', columns='code', values='gics_sec', aggfunc='first').replace(GICS_NAME).reindex(columns=px.columns)
else:
    GICS = None

def sector_at(t, scheme):
    """심사 산업군 라벨 — t 이전에 공표된 분류만 사용.
    GICS: t 이전 마지막 거래소 GICS → 없으면 t 이전 마지막 WICS → 없으면 t 시점 FnGuide 업종을 고정 대응표로 변환."""
    ig = (SEC.loc[:t].iloc[-1] if (SEC.index <= t).any() else pd.Series(np.nan, index=SEC.columns)).reindex(px.columns)
    if scheme == 'IG27': return ig
    ig_m = ig.map(IG_MAP)
    if t >= pd.Timestamp('2018-09-28'): ig_m[ig == '미디어'] = '커뮤니케이션서비스'
    else: ig_m[ig == '미디어'] = '경기관련소비재'
    w = WICS.loc[WICS.index[WICS.index <= t][-1]] if (WICS is not None and (WICS.index <= t).any()) else pd.Series(np.nan, index=px.columns)
    w = w.fillna(ig_m)
    if scheme == 'WICS' or GICS is None: return w
    s = GICS.loc[GICS.index[GICS.index <= t][-1]] if (GICS.index <= t).any() else pd.Series(np.nan, index=px.columns)
    return s.fillna(w)

def review_window(e, idx='KOSPI200', months_pre2020=12):
    y, mo = int(e[:4]), int(e[5:])
    end = S[(S.year == y) & (S.month == mo - 2)][-1]
    if y <= 2019: start = end - pd.DateOffset(months=months_pre2020) + pd.Timedelta(days=1)
    elif mo == 6: start = pd.Timestamp(y - 1, 11, 1)
    else: start = pd.Timestamp(y, 5, 1)
    return ix[S[S >= start][0]], ix[end]

K150 = pd.read_csv(PROC / 'kosdaq150_snapshots.csv', parse_dates=['date'])
K150_DATES = np.sort(K150.date.unique())
NOT_PIT = set()                                                          # 그 시점 구성종목 자료가 없어 미래 스냅샷을 쓴 (지수, 회차) — 백테스트에서 제외
def current_members(e, idx, t):
    """t 시점에 알려진 구성종목. KOSPI200 = t 이전 마지막 월말, KOSDAQ150 = t 이전 마지막 거래소 스냅샷(직전 회차 변경 후)
    에서 그 뒤 상장폐지·코스피 이전으로 수시 제외된 종목을 뺀 것 (t까지 알려진 사실만 사용).
    수시 제외 종목은 거래소 예비종목이 대신 들어가지만 예비종목 명단 자료가 없어 채우지는 않는다."""
    if idx == 'KOSPI200':
        return mem.loc[mem.index[mem.index <= t][-1]].reindex(px.columns, fill_value=False)
    d = K150_DATES[K150_DATES <= np.datetime64(t)]
    if len(d): d = d[-1]
    else: d = K150_DATES[0]; NOT_PIT.add((idx, e))                      # 첫 회차(2015-12): 이전 스냅샷 없음
    cur = pd.Series(px.columns.isin(K150[K150.date == d].code), index=px.columns)
    i = ix[S[S <= t][-1]]
    gone = cur & (~market_at(S[i]).eq('코스닥') | ~pd.Series(np.isfinite(PXV[i]), index=px.columns))
    return cur & ~gone

def universe(e, iT0, idx, cur, i_info=None):
    """i_info: 예측하는 날. 관리종목·상장폐지·소속 시장·유동비율은 그날까지 알려진 값으로 본다 (기본 = 심사기준일).
    상장 기간은 날짜로 미리 계산할 수 있으므로 늘 심사기준일 기준."""
    t = S[iT0]; y = int(e[:4])
    iu = iT0 if i_info is None else min(i_info, iT0); tu = S[iu]
    mkt = market_at(tu)
    age_days = (t - FIRST_PX.reindex(px.columns)).dt.days
    age_ok = age_days >= (365 if (y <= 2019 and idx == 'KOSPI200') else 182)
    mgt = MGT.loc[:tu].iloc[-1].reindex(px.columns).astype(str).eq('관리')
    alive = pd.Series(np.isfinite(PXV[iu]), index=px.columns)
    fl_ok = pd.Series(True, index=px.columns)
    if FLOAT is not None and tu >= FLOAT.dropna(how='all').index[0]:
        # 유동주식비율 10% 미만 제외 (신규 후보만, 값 없으면 통과). 분모는 그날 이전 마지막 거래소 스냅샷의 상장주식수
        fl_ok = ~(float_ratio(iu) < 0.10) | cur
    m = '유가증권시장' if idx == 'KOSPI200' else '코스닥'
    return mkt.eq(m) & base_ok(tu) & ~mgt & alive & fl_ok & (age_ok | cur)

DEFAULT = {'KOSPI200': dict(N=200, C_pre=0.80, C_post=0.85, L=0.85, buf=(1.1, 0.9), top_special=50, young_special=None),
           'KOSDAQ150': dict(N=150, C_pre=0.60, C_post=0.60, L=0.80, buf=(1.1, 0.9), top_special=None, months_pre2020=6, young_special=None)}
# young_special (KOSDAQ150): 상장 6개월 미만 종목이라도 최근 15거래일 평균 시총이 코스닥 상위 N위 이내면 편입 (N은 IS 회차로 선택)
TECH = {'IT', '건강관리', '커뮤니케이션서비스'}; NONTECH = {'소재', '산업재', '필수소비재', '경기관련소비재'}

def _step1(df, C, L):
    out = pd.Series(False, index=df.index); rk = pd.Series(np.nan, index=df.index); ncur = pd.Series(0, index=df.index)
    for _, g in df.groupby('grp'):
        g = g.sort_values('mc', ascending=False)
        cum = g.mc.cumsum() / g.mc.sum()
        in_cum = cum.shift(fill_value=0) < C
        liq = g.tv.rank(ascending=False) <= L * len(g)
        out[g.index] = in_cum & liq
        rk[g.index] = np.arange(1, len(g) + 1); ncur[g.index] = int(g.cur.sum())
    return out, rk, ncur

def run_rules(e, scheme='WICS', i_end=None, mc_override=None, idx='KOSPI200', params=None, i_special=None, i_info=None, strategy_a=False):
    """예상 구성종목 집합, 현재 구성종목 집합, 상세 표. i_end·mc_override는 심사 기간 중간 예측용.
    i_special: 신규상장 특례를 판단하는 날 (기본 = i_end). 심사기준일 뒤 예측이면 그 사이 상장한 종목도 포함.
    i_info: 심사기준일 전 예측이면 그날 (관리종목·상폐·시장·유동비율·현재 구성종목을 그날 기준으로, 미래 정보 차단)."""
    P = {**DEFAULT[idx], **(params or {})}
    y = int(e[:4]); i0, iT0 = review_window(e, idx, P.get('months_pre2020', 12))
    i_end = iT0 if i_end is None else i_end
    t = S[iT0] if i_info is None else S[min(i_info, iT0)]
    cur = current_members(e, idx, t)
    u = universe(e, iT0, idx, cur) if i_info is None else universe(e, iT0, idx, cur, i_info=i_info)
    avg_mc = MCD.iloc[i0:i_end + 1].mean() if mc_override is None else mc_override
    avg_tv = TVD.iloc[i0:i_end + 1].mean()
    grp = sector_at(S[i_end], scheme)
    if idx == 'KOSPI200': grp = grp.replace('부동산', '금융')              # KOSPI200 산업군 = 금융/부동산
    df = pd.DataFrame({'mc': avg_mc, 'tv': avg_tv, 'grp': grp.fillna('미분류'), 'cur': cur})[u & avg_mc.notna()]
    df['liq'] = df.groupby('grp').tv.rank(ascending=False) <= P['L'] * df.groupby('grp').tv.transform('size')
    b_keep, b_new = P['buf'] if P['buf'] else (None, None)
    C = P['C_pre'] if y <= 2019 else P['C_post']

    if idx == 'KOSDAQ150':
        df = df[~df.grp.isin(['에너지', '유틸리티', '부동산'])]
        if y <= 2019:
            df = df[df.grp != '금융']                                       # 2019년까지 금융 제외
    if idx == 'KOSDAQ150' and y <= 2019:
        nt = df[df.grp.isin(NONTECH)].copy()
        s1, rk, ncur = _step1(nt, C, P['L'])
        keep = nt.cur & ((rk <= b_keep * ncur) | s1) if b_keep else s1
        new = ~nt.cur & s1 & ((rk <= b_new * np.maximum(ncur, 1)) if b_new else True)
        sel = pd.Series(False, index=df.index); sel[nt.index] = keep | new
        tech = df[df.grp.isin(TECH) & df.liq].sort_values('mc', ascending=False)
        k = P['N'] - int(sel.sum())
        if b_keep:                                                         # 기술주에도 같은 버퍼: 기존은 k×110% 안, 신규는 k×90% 안
            trk = pd.Series(np.arange(1, len(tech) + 1), index=tech.index)
            cand = tech.index[(tech.cur & (trk <= b_keep * k)) | (~tech.cur & (trk <= b_new * k))]
            sel[cand] = True
        else:
            sel[tech.index[:k]] = True
    else:
        s1, rk, ncur = _step1(df, C, P['L'])
        keep = df.cur & ((rk <= b_keep * ncur) | s1) if b_keep else df.cur & s1
        new = ~df.cur & s1 & ((rk <= b_new * np.maximum(ncur, 1)) if b_new else True)
        sel = keep | new
        df['rk'], df['n_cur'], df['sel1'] = rk, ncur, s1
    top = pd.Series(False, index=df.index)
    if P['top_special']:
        mc15 = MCD.iloc[max(i_end - 14, 0):i_end + 1].mean().reindex(df.index)
        top = mc15.rank(ascending=False) <= P['top_special']
        sel |= top
        # 대형 신규상장 특례: 상장 6개월 미만이라 심사 대상에서 빠진 유가증권 보통주라도
        # 최근(상장 후) 15거래일 평균 시총이 시장 50위 이내면 편입 (카카오페이·HD현대중공업·하이브·SK바이오사이언스 사례)
        i_s = i_end if i_special is None else i_special
        ts = S[i_s]
        mk = market_at(ts)
        fp = FIRST_PX.reindex(px.columns)
        young = (mk.eq('유가증권시장') & base_ok(ts) & fp.notna() & (fp <= ts) & ((S[iT0] - fp).dt.days < 182)
                 & ~cur & ~pd.Series(px.columns.isin(df.index), index=px.columns))
        if young.any():
            m15 = MCD.iloc[max(i_s - 14, 0):i_s + 1].mean()
            pool = pd.concat([mc15, m15[young.index[young]]]).dropna()
            sp = [c for c in young.index[young] if c in pool.index and pool.rank(ascending=False)[c] <= P['top_special']]
            if sp:
                add = pd.DataFrame({'mc': m15[sp], 'tv': TVD.iloc[max(i_s - 14, 0):i_s + 1].mean()[sp], 'grp': grp.reindex(sp).fillna('미분류'),
                                    'cur': False, 'liq': True, 'special_new': True}, index=sp)
                df = pd.concat([df, add])
                sel = sel.reindex(df.index, fill_value=False); top = top.reindex(df.index, fill_value=False)
                sel[sp] = True; top[sp] = True
    if idx == 'KOSDAQ150' and P.get('young_special'):
        # 코스닥 신규·재상장 대형주 특례: 상장 6개월 미만(심사기준일 기준)이라 심사에서 빠진 종목 중
        # 최근 15거래일 평균 시총(상장 후 날만)이 코스닥 보통주 전체에서 N위 이내면 편입
        i_s = i_end if i_special is None else i_special
        ts = S[i_s]
        kq = market_at(ts).eq('코스닥') & base_ok(ts) & pd.Series(np.isfinite(PXV[i_s]), index=px.columns)
        m15 = MCD.iloc[max(i_s - 14, 0):i_s + 1].mean()[kq].dropna()
        fp = FIRST_PX.reindex(px.columns)
        excl_grp = {'에너지', '유틸리티', '부동산'} | ({'금융'} if y <= 2019 else set())
        g_all = grp.reindex(px.columns)
        young = kq & fp.notna() & (fp <= ts) & ((S[iT0] - fp).dt.days < 182) & ~cur & ~g_all.isin(excl_grp)                 & ~pd.Series(px.columns.isin(df.index), index=px.columns)
        rk = m15.rank(ascending=False)
        sp = [c for c in young.index[young] if c in rk.index and rk[c] <= P['young_special']]
        if sp:
            add = pd.DataFrame({'mc': m15[sp], 'tv': TVD.iloc[max(i_s - 14, 0):i_s + 1].mean()[sp], 'grp': g_all[sp].fillna('미분류'),
                                'cur': False, 'liq': True, 'special_new': True}, index=sp)
            df = pd.concat([df, add])
            sel = sel.reindex(df.index, fill_value=False); top = top.reindex(df.index, fill_value=False)
            sel[sp] = True; top[sp] = True
    n = int(sel.sum())
    if n < P['N']:
        add = df[~sel & df.cur & df.liq].sort_values('mc', ascending=False).index[:P['N'] - n]; sel[add] = True
        if sel.sum() < P['N']:
            add2 = df[~sel & df.liq].sort_values('mc', ascending=False).index[:P['N'] - int(sel.sum())]; sel[add2] = True
    elif n > P['N']:
        drop = df[sel & ~top].sort_values('mc').index[:n - P['N']]; sel[drop] = False
    df['pred'] = sel
    if strategy_a and idx=='KOSPI200':
        # 전체 구성 예측은 유지한 뒤 A의 신규 매수 후보만 분리한다.
        # 제외 빈자리를 후순위 후보로 채우지 않는다(현금 배정 유지).
        from eligibility import strategy_a_exclusion
        info=i_end if i_info is None else min(i_info,iT0)
        df['index_pred']=df.pred
        df['a_candidate_before_filter']=df.pred & ~df.cur
        df['a_exclusion_reason']=[strategy_a_exclusion(c,idx,info) if flag else ''
                                  for c,flag in df.a_candidate_before_filter.items()]
        df['a_requires_review']=df.a_exclusion_reason.str.startswith('needs_')
        df.loc[df.a_exclusion_reason.ne(''),'pred']=False
    return set(df.index[df.pred]), set(cur.index[cur]), df

def evaluate(scheme, idx='KOSPI200', params=None, events=None):
    rows = []
    E = ev[ev.idx == idx]
    for e in (events or sorted(E.event.unique())):
        pred, cur, _ = run_rules(e, scheme, idx=idx, params=params)
        a = set(E[(E.event == e) & (E.type == '편입')].code); d = set(E[(E.event == e) & (E.type == '편출')].code)
        pa, pdl = pred - cur, cur - pred
        rows.append(dict(event=e, smp='IS' if int(e[:4]) <= 2019 else 'OOS', 실제편입=len(a), 예측편입=len(pa), 편입적중=len(pa & a),
                         실제편출=len(d), 예측편출=len(pdl), 편출적중=len(pdl & d)))
    return pd.DataFrame(rows)

def score(r, by='smp'):
    g = r.groupby(by)[['실제편입', '예측편입', '편입적중', '실제편출', '예측편출', '편출적중']].sum()
    g['편입 정밀도(%)'] = g.편입적중 / g.예측편입 * 100; g['편입 재현율(%)'] = g.편입적중 / g.실제편입 * 100
    g['편출 정밀도(%)'] = g.편출적중 / g.예측편출 * 100; g['편출 재현율(%)'] = g.편출적중 / g.실제편출 * 100
    g['F1 평균'] = (2 * g.편입적중 / (g.예측편입 + g.실제편입) + 2 * g.편출적중 / (g.예측편출 + g.실제편출)) / 2 * 100
    return g.round(1)

if __name__ == '__main__':
    pd.set_option('display.width', 230)
    nm = lambda x: ', '.join(sorted(str(names[c]) for c in x))
    if True:
        print('== KOSPI200 (심사기준일 데이터, 공개 규칙 그대로)')
        for scheme in ('IG27', 'WICS', 'GICS'):
            r = evaluate(scheme); print(f'-- 산업군 = {scheme}'); print(score(r).to_string())
            r.to_csv(OUT / f'rule_engine_kospi200_{scheme}.csv', index=False, encoding='utf-8-sig')
        r = evaluate('GICS')
        print('\nGICS 회차별 (2010~)'); print(r[r.event >= '2010'].to_string(index=False))
        E = ev[ev.idx == 'KOSPI200']
        for e in sorted(r.event.unique())[-2:]:
            pred, cur, df = run_rules(e, 'GICS')
            a = set(E[(E.event == e) & (E.type == '편입')].code); d = set(E[(E.event == e) & (E.type == '편출')].code)
            print(f'\n{e} 예측 편입: {nm(pred - cur)}')
            print(f'{e} 실제 편입: {nm(a)}')
            print(f'{e} 예측 편출: {nm(cur - pred)}')
            print(f'{e} 실제 편출: {nm(d)}')

    # KOSDAQ150 기준값 선택(IS 회차로만)은 kosdaq150_calib.py
