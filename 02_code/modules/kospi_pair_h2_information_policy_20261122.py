"""Information-time policy for reconstructing the 2026 H2 paper account.

This supplies dates and decision states, not forecasts or simulated fills.
Missing announced outcomes do not prevent entry or mark-to-market. They never
become a negative inclusion label. Classification switches do not create orders.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / '01_data/processed/kospi_pair_nonfnguide_20261122'
OUT = ROOT / '04_results/kospi_pair_nonfnguide_20261122'
PUBLICATION = pd.Timestamp('2026-10-30')
AVAILABLE = pd.Timestamp('2026-11-02')


def sector_policy(asof):
    day = pd.Timestamp(asof).normalize()
    # The PDF has a publication date, not a certified intraday timestamp.
    # Do not assume it was accessible before the Oct-30 closing signal.
    new = day >= AVAILABLE
    return {
        'scheme': 'KRICS' if new else 'GICS',
        'classification_basis': ('official_final_classification' if new
                                 else 'previous_classification_available_at_signal'),
        'source_file': ('KRICS_official_final_20261030.csv' if new
                        else 'official_sector_query_observations.csv'),
        'source_publication_date': '2026-10-30' if new else None,
        'available_from': '2026-11-02' if new else None,
        'creates_additional_order': False,
    }


def announcement_state(index_id, code, asof):
    """Three-state result: unknown must not trigger a failed-prediction sale."""
    if index_id not in ('KOSPI200', 'KOSPI100'):
        raise ValueError(index_id)
    day = pd.Timestamp(asof).normalize()
    if index_id == 'KOSPI100' or day < pd.Timestamp('2026-11-19'):
        return 'unknown'
    if day > pd.Timestamp('2026-11-20'):
        raise ValueError('Collected announcement coverage ends at 2026-11-20')
    roster = pd.read_csv(DATA / 'KOSPI200_announced_planned_members_202612.csv',
                         dtype={'code': str})
    code = str(code)
    code = code if code.startswith('A') else 'A' + code
    return 'confirmed_in' if code in set(roster.code) else 'confirmed_out'


def main():
    signals = pd.read_csv(DATA / 'H2_signal_dates.csv')
    rows = []
    for index_id in ('KOSPI200', 'KOSPI100'):
        for row in signals.to_dict('records'):
            rows.append(dict(index_id=index_id, **row,
                             **sector_policy(row['information_date']),
                             entry_requires_announced_outcome=False,
                             entry_requires_valid_forecast_and_market_data=True,
                             forecast_generated=False, execution_replayed=False))
    frame = pd.DataFrame(rows)
    assert len(frame) == 8 and frame.scheme.eq('GICS').all()
    assert sector_policy('2026-10-30')['scheme'] == 'GICS'
    assert sector_policy('2026-11-02')['scheme'] == 'KRICS'
    assert announcement_state('KOSPI100', 'A000100', '2026-11-20') == 'unknown'
    assert announcement_state('KOSPI200', 'A000100', '2026-11-18') == 'unknown'
    assert announcement_state('KOSPI200', 'A000100', '2026-11-19') == 'confirmed_in'
    assert announcement_state('KOSPI200', 'A0220W0', '2026-11-19') == 'confirmed_out'
    frame.to_csv(DATA / 'H2_information_policy.csv', index=False, encoding='utf-8-sig')
    protocol = {
        'requested_cutoff': '2026-11-20',
        'entry_requires_final_announcement': False,
        'valuation_requires_final_announcement': False,
        'unknown_outcome_action': 'continue marking existing holdings; do not infer failed prediction',
        'confirmed_out_action': 'sell from next close after public announcement, subject to liquidity limits',
        'confirmed_in_action': 'retain scheduled rebalance exit, subject to liquidity limits',
        'no_KOSPI100_outcome_proxy_from_KOSPI200': True,
        'missing_KOSPI100_outcome_caveat': 'holding-path reconstruction remains provisional if a missed public announcement would have triggered exit',
        'KRICS_publication_date': str(PUBLICATION.date()),
        'KRICS_first_conservative_use': str(AVAILABLE.date()),
        'historical_four_entry_signals_use': 'GICS',
        'new_classification_does_not_generate_unscheduled_orders': True,
        'methodology_effective_date_distinct_from_classification_publication': True,
        'historical_core_thresholds_not_overwritten': True,
        'coverage_status': 'policy_ready_forecasts_and_execution_not_reconstructed',
        'remaining_inputs': ['FnGuide price/adjustment/turnover/capitalization refresh',
                             'date-available eligibility and corporate-action reconciliation',
                             'both-index four-tranche targets and shared-liquidity execution replay'],
        'tests': 'passed: four-signal classification, publication boundary, unknown and confirmed outcomes',
    }
    (OUT / 'H2_information_policy.json').write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame[['index_id', 'lead', 'information_date', 'scheme']].to_string(index=False))
    print(protocol['tests'])


if __name__ == '__main__':
    main()
