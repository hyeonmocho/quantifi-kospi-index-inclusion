"""신호 A 전략 — 단순 규칙: "편입될 종목을 예측해서 미리 사 둔다" (2026-10-15 확정)

규칙 (기존 3개월·동일가중 유지. KOSDAQ150은 기록용이며 현재 운용 대상 아님)
  1) 심사기준일 60거래일(약 3개월) 전 종가까지의 자료로 엔진(rule_engine) 예측 → 다음날 종가에 예측 편입 종목을 한 번에 동일 금액 매수
     (이전 연구의 선택 이력이 있는 고정 규칙. 새 기간에서 재최적화하지 않음)
  2) 편입 종목은 매매일(패시브가 사는 날) 종가에 매도, 거래소 공표에서 빠진 종목은 공표 다음날 종가에 매도
  3) 시장 위험은 지수 선물로 헤지: 손익 = 종목 − 베타 × 지수 (베타 = 매수 전 150~30거래일 일별 회귀)
  4) eligibility.py: 진입 자료 결측/거래불가 및 당시 시총순위 기준 B_1 특례 후보 제외.
     제외분은 원래 배정비중 현금, 비용·헤지 0. 보유 중 결측은 미평가로 표시.
미래 참조 차단: 엔진 입력은 모두 예측일까지 알려진 값 (rule_engine 참고), 매수는 예측 다음날
표본 구분: sample_policy.py 기준. KOSPI200 2002-06~2016-06 IS 15회 / 2017-06~2026-06 역사적 OOS 16회.
  2001 HISTORICAL, 2026-06 이후 FORWARD. 2002-06은 ETF 개설 전 회차임에 유의.
  KOSDAQ150의 과거 분할은 기록용 유지.
  판정은 KOSPI200만 사용하며 과거 기준(IS t ≥ 2 & OOS 평균 > 0 & OOS t ≥ 1.65)의 참고 판정이다.
  최근 표본을 이미 관찰했으므로 미사용 OOS 검증이나 자동 운용 채택으로 해석하지 않는다.

결과: 04_results/signal_a/simple_rule_events.csv, simple_rule_summary.csv, simple_capital_book.csv(종목별 매매 기록)
"""
import sys
from pathlib import Path
from sample_policy import sample_label, relabel, main_sample
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
from rule_engine import *
import rule_engine as RE
from eligibility import apply_eligibility, aggregate

H = -60
FEE = 0.0025
P150 = json.load(open(OUT / 'rule_engine_kosdaq150_params_GICS.json', encoding='utf-8')); P150['buf'] = tuple(P150['buf']) if P150.get('buf') else None
TV20 = TVD.rolling(20, min_periods=10).mean()
VOL60 = pd.DataFrame(RT, index=S, columns=px.columns).rolling(60, min_periods=30).std()

def sample(idx, e):
    return sample_label(idx, e)

def predict(e, idx, include_excluded=False):
    """심사기준일 H거래일 전 예측: 심사 기간 평균 시총 = (그날까지 실제 + 그날 시총 × 남은 일수) ÷ 전체 일수."""
    p = P150 if idx == 'KOSDAQ150' else None
    i0, iT0 = review_window(e, idx, (p or {}).get('months_pre2020', 12))
    ip = iT0 + H
    n_all, n_obs = iT0 - i0 + 1, ip - i0 + 1
    proj = (MCD.iloc[i0:ip + 1].sum(min_count=1) + MCD.iloc[ip] * (n_all - n_obs)) / n_all if n_obs > 0 else MCD.iloc[ip].copy()
    if n_obs > 0: proj[MCD.iloc[i0:ip + 1].notna().sum() == 0] = np.nan
    pred, cur, detail = run_rules(e, 'GICS', i_end=ip, mc_override=proj, idx=idx, params=p, i_info=ip, strategy_a=True)
    if include_excluded and 'a_candidate_before_filter' in detail:
        # 감사 원장은 제외 후보의 원래 현금 배정 자리도 보존한다.
        return sorted(detail.index[detail.a_candidate_before_filter]), ip
    return sorted(pred - cur), ip

def event_book(indices=('KOSPI200', 'KOSDAQ150')):
    out = []
    for idx in indices:
        for e, g in ev[ev.idx == idx].groupby('event'):
            if idx == 'KOSDAQ150' and e < '2016-06': continue                 # 2015-12: 그 시점 구성종목 자료 없음
            _, iT0 = review_window(e, idx)
            if idx == 'KOSDAQ150' and S[iT0 + H] < RE.K150_DATES[0]: continue
            names_, ip = predict(e, idx, include_excluded=True); ib = ip + 1
            iA = int(g.iEarliest.iloc[0]) - 1; iE = int(g.iTE.iloc[0]); act = set(g[g.type == '편입'].code)
            for c in names_:
                hit = c in act; b = stock_beta(c, idx, ip); sell = iE if hit else iA + 1
                out.append(dict(시기=e, idx=idx, 표본=sample(idx, e), code=c, name=names.get(c), ib=ib, sell=sell, hit=hit, beta=b,
                                adv=TV20.iloc[ip].get(c, np.nan), sig=VOL60.iloc[ip].get(c, np.nan), r=excess(c, ib, sell, BENCH[idx], True, b)))
        print(idx, 'done', flush=True)
    return apply_eligibility(pd.DataFrame(out))

if __name__ == '__main__':
    Bk = event_book()
    Bk.to_csv(OUT / 'simple_capital_book.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 230)
    # 1) 규칙 성과 (회차 동일가중, 비용 왕복 0.5%, 매수 한도·충격 없음)
    E = aggregate(Bk)
    Bk[~Bk.traded].to_csv(OUT/'excluded_candidates.csv',index=False,encoding='utf-8-sig')
    E.to_csv(OUT / 'simple_rule_events.csv', index=False, encoding='utf-8-sig')
    rows = {}
    E_main = main_sample(E)
    for lab, d in (('KOSPI200', E_main[E_main.idx == 'KOSPI200']), ('KOSDAQ150', E_main[E_main.idx == 'KOSDAQ150']), ('두 지수 합산', E_main)):
        for s in ('IS', 'OOS', '전체'):
            x = d.ret if s == '전체' else d[d.표본 == s].ret
            rows[(lab, s)] = {'회차': len(x), '평균(%)': x.mean() * 100, '중앙값(%)': x.median() * 100, 't': tstat(x), '승률(%)': (x > 0).mean() * 100,
                              '최악(%)': x.min() * 100, '적중률(%)': d.적중률.mean() * 100 if s == '전체' else d[d.표본 == s].적중률.mean() * 100}
    T = pd.DataFrame(rows).T; T.to_csv(OUT / 'simple_rule_summary.csv', encoding='utf-8-sig')
    print('== 단순 규칙 성과 (회차당 %, 시장 영향 제거, 비용 0.5%)'); print(T.round(2).to_string())
    i_, o_ = T.loc[('KOSPI200', 'IS')], T.loc[('KOSPI200', 'OOS')]
    v = '채택' if (i_.t >= 2 and o_['평균(%)'] > 0 and o_.t >= 1.65) else ('보류' if o_['평균(%)'] > 0 else '기각')
    print(f"역사적 재검증(KOSPI200, 자동 운용 변경 아님): {v} — IS t {i_.t:.2f}, OOS 평균 {o_['평균(%)']:+.2f}%, OOS t {o_.t:.2f}")
