"""외국인·기관은 편입 종목을 언제 얼마나 사고 파나 — 선취매(헤지펀드)와 패시브(인덱스 펀드)의 흔적.

자료 (FnGuide DataGuide, 일간, 코스피·코스닥 전 종목, 만원)
  외국인: '외국인등록 순매수대금(일간)' (CI20093020), 2000-01 ~ 2026-10-14  (01_data/raw/fnguide/등록외국인순매수대금(일간).xlsx)
  투신  : '투신 순매수대금(일간)' (CI20033020), 2000-01 ~ 2026-10-14       (투신_순매수대금.xlsx) — 공모·인덱스 펀드
  연기금: '연기금 순매수대금(일간)' (CI20063020), 2000-01 ~ 2026-10-14     (연기금순매수대금.xlsx)
  기관  : '기관 순매수대금(일간)' (CI20003020), 2000-01 ~ 2026-09-04       (기관_순매수대금.xlsx)
  기관 기타 = 기관 − 투신 − 연기금 (금융투자·보험·은행·사모 등. 국내 증권사 스와프(TRS)를 통한 헤지펀드 매매, ETF 유동성 공급 물량이 주로 여기)
  순매수 = 매수금액 − 매도금액. 만원 → 원으로 바꿔 씀
대상: 실제 편입 종목 (KOSPI200 2001~, KOSDAQ150 2016-06~)
대조군: 같은 회차·같은 시장에서 편입되지 않은 비구성종목 중 시총이 편입 종목 범위(0.5~2배) 안인 종목 — 평소 흐름을 빼기 위함
지표
  구간 합: ① 3개월 전(T0−60) → 공표일, ② 공표 다음날 → 매매일 전날, ③ 매매일 당일, ④ 매매일 후 20일 — 누적 순매수 ÷ 시총(T0−60) (%)
  일별 경로: 매매일 기준 −80 ~ +20거래일 누적 순매수 ÷ 시총 (%), 편입·대조군 평균
  금액: 한 종목당 누적 순매수 (억 원, 중앙값). 가정(패시브 자금 규모 등) 없이 관측값만 사용
결과: 04_results/signal_a/foreign_flow_stocks.csv, foreign_flow_paths.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_engine import *

def load_flow(fname, key):
    f = PROC / f'flow_{key}.pkl'
    if f.exists(): return pd.read_pickle(f)
    m = pd.read_excel(ROOT / '01_data/raw/fnguide' / fname, header=None)
    x = m.iloc[14:, :].copy(); x.columns = ['date'] + m.iloc[8, 1:].tolist()
    x['date'] = pd.to_datetime(x['date'])
    x = x.set_index('date').apply(pd.to_numeric, errors='coerce') * 1e4          # 만원 → 원
    x.to_pickle(f); return x

K0, K1 = -80, 20                                                                # 매매일 기준 경로 범위

if __name__ == '__main__':
    FLOW = {k: load_flow(f, k).reindex(index=S, columns=px.columns) for k, f in
            (('외국인', '등록외국인순매수대금(일간).xlsx'), ('투신', '투신_순매수대금.xlsx'), ('연기금', '연기금순매수대금.xlsx'),
             ('기관', '기관_순매수대금.xlsx'))}
    last = min(v.dropna(how='all').index[-1] for v in FLOW.values())
    FLOW['기관 기타'] = (FLOW['기관'] - FLOW['투신'].fillna(0) - FLOW['연기금'].fillna(0)).where(FLOW['기관'].notna())
    for k in FLOW: FLOW[k] = FLOW[k].loc[:last].reindex(S)
    first = {k: v.dropna(how='all').index[0] for k, v in FLOW.items()}
    rows, paths = [], []
    for idx in ('KOSPI200', 'KOSDAQ150'):
        mk_name = '유가증권시장' if idx == 'KOSPI200' else '코스닥'
        for e, g in ev[ev.idx == idx].groupby('event'):
            if idx == 'KOSDAQ150' and e < '2016-06': continue
            _, iT0 = review_window(e, idx); i0 = iT0 - 60
            iA = int(g.iEarliest.iloc[0]) - 1; iE = int(g.iTE.iloc[0])
            if iE + K1 >= len(S) or i0 < 0: continue
            adds = sorted(set(g[g.type == '편입'].code))
            mc0 = MCD.iloc[i0] * 1e6
            cur = current_members(e, idx, S[i0]); mkt = market_at(S[i0])
            lo, hi = mc0.reindex(adds).min() * 0.5, mc0.reindex(adds).max() * 2
            ctrl = [c for c in px.columns if mkt.get(c) == mk_name and not cur.get(c, False) and c not in adds and lo <= mc0.get(c, np.nan) <= hi]
            segs = {'①3개월 전→공표': (i0 + 1, iA), '②공표→매매일 전날': (iA + 1, iE - 1), '③매매일': (iE, iE), '④매매일 후 20일': (iE + 1, iE + K1)}
            for grp, codes in (('편입', adds), ('대조군', ctrl)):
                codes = [c for c in codes if np.isfinite(mc0.get(c, np.nan)) and mc0[c] > 0]
                if not codes: continue
                for who, F in FLOW.items():
                    if S[iE + K0] < first[who]: continue
                    blk = F.iloc[iE + K0:iE + K1 + 1][codes].fillna(0.0)
                    cum = blk.cumsum().div(mc0[codes], axis=1) * 100
                    paths.append(pd.DataFrame({'idx': idx, 'event': e, '그룹': grp, '주체': who, 'k': np.arange(K0, K1 + 1),
                                               '누적순매수_시총대비': cum.mean(axis=1).values, '종목수': len(codes), 'TA_k': iA - iE}))
                    for c in codes:
                        r = dict(idx=idx, event=e, 그룹=grp, 주체=who, code=c, name=names.get(c), 시총억=mc0[c] / 1e8)
                        for k, (a, b) in segs.items(): r[k] = F.iloc[a:b + 1][c].sum() / 1e8
                        rows.append(r)
        print(idx, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / 'foreign_flow_stocks.csv', index=False, encoding='utf-8-sig')
    Pth = pd.concat(paths, ignore_index=True); Pth.to_csv(OUT / 'foreign_flow_paths.csv', index=False, encoding='utf-8-sig')
