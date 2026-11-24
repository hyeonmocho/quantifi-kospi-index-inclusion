"""신호 A 공통 데이터: 수익률·벤치마크·KOSPI200/KOSDAQ150 정기변경 이벤트·구간 수익률 함수.
event_windows.py, is_oos_covid.py가 `from events_data import *`로 쓴다. 날짜·초과수익 정의는 event_windows.py 머리말 참고."""
import datetime as dt, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # 개인프로젝트/
PROC = ROOT / '01_data/processed'; OUT = ROOT / '04_results/signal_a'; OUT.mkdir(parents=True, exist_ok=True)
CAL = ROOT / '01_data/calendar/korea_flow_calendar_data.xlsx'
COST = 0.005

D = pd.read_pickle(PROC / 'signal_a_cache.pkl')
px, mc, mem = D['px'], D['mc'], D['mem']
S = px.index[px.notna().sum(axis=1) > 100]
ret = px.reindex(S).pct_change(fill_method=None)
RT, PXV = ret.values, px.reindex(S).values
CI = {c: j for j, c in enumerate(ret.columns)}
ix = {t: i for i, t in enumerate(S)}
def on_or_before(t): return S[S <= t][-1]
def second_thu(y, mo):
    d = dt.date(y, mo, 1); d += dt.timedelta(days=(3 - d.weekday()) % 7 + 7)
    return on_or_before(pd.Timestamp(d))

# ---- 벤치마크 ----
memd = mem.reindex(S, method='ffill').shift(1).fillna(False).astype(bool)
w = (mc.reindex(S).shift(1) * memd).fillna(0).astype(float)
b200 = (ret.fillna(0) * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
k = pd.read_excel(ROOT / '01_data/raw/fnguide/코스닥150_수정주가.xlsx', header=None)
k = k.iloc[14:, :2].set_index(0)[1]; k.index = pd.to_datetime(k.index); k = pd.to_numeric(k, errors='coerce')
b150 = k.reindex(S).pct_change(fill_method=None)
BENCH = {'KOSPI200': b200.values, 'KOSDAQ150': b150.values}

# ---- 이벤트 ----
cal = pd.read_excel(CAL, sheet_name='A_지수정기변경')
ta_col = [c for c in cal.columns if c.startswith('공표일')][0]
TA = {(r['지수'], r['정기변경']): pd.Timestamp(r[ta_col]) for _, r in cal.iterrows() if pd.notna(r[ta_col])}
for _, r in pd.read_excel(CAL, sheet_name='A_공표일_이력').iterrows():
    TA.setdefault(('KOSPI200', r['정기변경']), pd.Timestamp(r['공표일']))
TA[('KOSDAQ150', '2018-12')] = pd.Timestamp('2018-11-23')   # 데일리안 2018-11-23 (캘린더 미반영)

rows = []
for y in range(2001, 2027):                                  # KOSPI200: 월말 구성종목 변화 (노트북과 같음)
    for mo in ([6] if y < 2020 else [6, 12]):
        key = f'{y}-{mo:02d}'; me = mem.index[(mem.index.year == y) & (mem.index.month == mo)]
        if len(me) == 0 or ('KOSPI200', key) not in TA: continue
        cur, prev = mem.loc[me[0]], mem[mem.index < me[0]].iloc[-1]
        for typ, flag in (('편입', cur & ~prev), ('편출', ~cur & prev)):
            rows += [dict(idx='KOSPI200', event=key, code=c, type=typ, TE=second_thu(y, mo)) for c in flag[flag].index]
k150 = pd.read_csv(PROC / 'kosdaq150_events.csv', parse_dates=['TE'])
rows += [dict(idx='KOSDAQ150', event=r.event, code=r.code, type={'add': '편입', 'del': '편출'}[r.type], TE=r.TE)
         for r in k150.itertuples()]
ev = pd.DataFrame(rows)
ev['y'] = ev.event.str[:4].astype(int)
ev['iTE'] = ev.TE.map(ix)
ev['iTA'] = [ix[on_or_before(TA[(a, e)])] if (a, e) in TA else np.nan for a, e in zip(ev.idx, ev.event)]
# 매수 가능한 가장 이른 날 = 공표 다음날 종가 (장 마감 후 발표).
# 공표일 미확인(KOSDAQ150 2015-12·2016-12·2017-12·2019-12)은 지금까지 관측된 가장 짧은 공표~매매일 간격(5거래일, KOSPI200 2006-06)을
# 가정 → 실제 공표보다 늦게 안다고 보는 쪽이라 미래 참조가 생기지 않음 (오답 손절이 늦어지는 만큼 성과는 보수적)
MIN_GAP = 5
ev['iEarliest'] = np.where(ev.iTA.notna(), ev.iTA + 1, ev.iTE - MIN_GAP + 1).astype(int)

# ---- 계산 도구 ----
BETA_WIN = (150, 30)    # 베타 추정: 공표 150거래일 전 ~ 30거래일 전 (심사·발표 영향 전 평소 기간)

def stock_beta(code, ind, i_ref):
    """종목 일별 수익률을 지수에 회귀한 베타. 관측 60일 미만이면 1, 극단값은 0~3으로 자름."""
    a, b = max(i_ref - BETA_WIN[0], 1), i_ref - BETA_WIN[1]
    y, x = RT[a:b, CI[code]], BENCH[ind][a:b]
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 60: return 1.0
    return float(np.clip(np.cov(y[m], x[m])[0, 1] / np.var(x[m], ddof=1), 0, 3))

def excess(code, i0, i1, B, hedge=True, beta=1.0):
    """i0 종가 → i1 종가 buy&hold 수익률. hedge=True면 지수 헤지: 종목 − beta × 지수 (beta=1이면 단순 초과)."""
    if i1 >= len(S) or i0 < 0: return np.nan
    j = CI[code]
    if not np.isfinite(PXV[i0, j]): return np.nan
    r = np.prod(1 + np.nan_to_num(RT[i0 + 1:i1 + 1, j])) - 1
    return r - beta * (np.prod(1 + np.nan_to_num(B[i0 + 1:i1 + 1])) - 1) if hedge else r

def window(d, a0, k0, a1, k1, hedge=True):
    """회차별 평균 수익률 Series. a0/a1 = 'iTA' 또는 'iTE'."""
    out = {}
    for (ind, e), g in d.groupby(['idx', 'event']):
        if g[[a0, a1]].isna().any().any(): continue
        i0, i1 = int(g[a0].iloc[0]) + k0, int(g[a1].iloc[0]) + k1
        rs = [excess(c, i0, i1, BENCH[ind], hedge) for c in g.code]
        if np.isfinite(rs).any(): out[e] = np.nanmean(rs)
    return pd.Series(out)

def tstat(x):
    x = pd.Series(x).dropna(); return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan

def summ(x):
    x = pd.Series(x).dropna()
    return pd.Series({'회차': len(x), '평균(%)': x.mean() * 100, 't': tstat(x), '+회차(%)': (x > 0).mean() * 100}).round(2)

