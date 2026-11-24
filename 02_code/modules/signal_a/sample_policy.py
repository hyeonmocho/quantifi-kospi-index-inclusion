"""전략 A 표본 정책. 사용자 요청에 따른 31회 시간순 15/16 균등 분할.

2026-10-28 변경: 2002-06~2016-06 IS 15회, 2017-06~2026-06 역사적 OOS 16회.
2019년까지 연 1회, 2020년부터 연 2회인 관측 캘린더에 따라 회차 수로 나눈다.
수익률을 보며 분할 경계를 선택하지 않는다. 이미 관찰한 자료의 재분할이다.
2001년만 HISTORICAL 비교군. 2002-06은 ETF 개설(2002-10-14) 전이므로
정확한 ETF 도입 후 표본은 2003-06부터이며 별도 민감도로 보고한다.
2026-06 이후는 FORWARD: 관측 전 확정한 규칙의 전진 검증용이다.
근거: 00_docs/signal_a_sample_period.md
"""
START_EVENT = '2002-06'
IS_END_EVENT = '2016-06'
OOS_START_EVENT = '2017-06'
HISTORY_END_EVENT = '2026-06'
IS_LABEL = f'IS {START_EVENT}~{IS_END_EVENT}'
OOS_LABEL = f'역사적 OOS {OOS_START_EVENT}~{HISTORY_END_EVENT}'
MAIN_LABEL = f'주 표본 {START_EVENT[:4]}~{HISTORY_END_EVENT[:4]}'
EXPECTED_IS_EVENTS = 15
EXPECTED_OOS_EVENTS = 16


def sample_label(index, event):
    if index != 'KOSPI200':
        return 'IS' if event <= '2021-12' else 'OOS'
    if event < START_EVENT:
        return 'HISTORICAL'
    if event <= IS_END_EVENT:
        return 'IS'
    if event <= HISTORY_END_EVENT:
        return 'OOS'
    return 'FORWARD'


def relabel(frame, event_column='시기'):
    """캐시의 오래된 IS/OOS 라벨도 재계산. 수익률/종목 선정은 변경하지 않는다."""
    result = frame.copy()
    indexes = result['idx'] if 'idx' in result else ['KOSPI200'] * len(result)
    result['표본'] = [sample_label(i, str(e)) for i, e in zip(indexes, result[event_column])]
    return result


def main_sample(frame, event_column='시기'):
    result = relabel(frame, event_column)
    return result[result['표본'].isin(['IS', 'OOS'])].copy()
