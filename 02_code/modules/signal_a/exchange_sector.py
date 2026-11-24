"""Dated exchange-sector input for Signal A research, separate from GICS.

CP10001090 is a vendor series, not an original historical KSIC publication.
Mappings below are an explicit legacy-eight-group proxy only. In particular,
the broad '제조' label cannot distinguish historical fishery/mining membership.
The frozen baseline does not import this module or change its candidate rules.
"""
from functools import lru_cache
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / '01_data/processed/exchange_sector_daily.pkl'
RULE_SOURCE = 'https://kind.krx.co.kr/external/2011/06/03/000103/20110603000235/10131.htm'

MANUFACTURING = {'음식료·담배','섬유·의류','종이·목재','화학','제약','비금속',
                 '금속','기계·장비','전기·전자','의료·정밀기기','운송장비·부품'}
SERVICE = {'유통','운송·창고','일반서비스','IT 서비스','오락·문화','부동산'}
FINANCE = {'금융','증권','보험','은행(단절)'}


def map_legacy8(label):
    """Return a proxy group and explicit qualification; never guess unknowns."""
    if pd.isna(label): return None, 'missing_label'
    if not label.startswith('코스피 '): return None, 'not_kospi_or_unverified_legacy_label'
    short=label.removeprefix('코스피 ')
    if short in MANUFACTURING: return '제조업', 'named_manufacturing_proxy'
    if short in SERVICE: return '서비스업', 'service_aggregation_proxy'
    if short in FINANCE: return '금융업', 'finance_aggregation_proxy'
    if short=='전기·가스': return '전기가스업', 'named_group_proxy'
    if short=='건설': return '건설업', 'named_group_proxy'
    if short=='통신': return '통신업', 'named_group_proxy'
    return None, 'ambiguous_broad_manufacturing' if short=='제조' else 'unmapped_label'


@lru_cache(maxsize=1)
def load_panel():
    if not PANEL.exists(): raise FileNotFoundError('Run 02_code/preprocess/exchange_sector_build.py first')
    return pd.read_pickle(PANEL)


def snapshot(t, codes=None, panel=None):
    """Latest *row strictly before* signal date; no per-stock carry/backfill.

    One-row lag is a conservative information-timing convention, not proof that
    the vendor did not revise older observations. Future rows are not consulted.
    """
    data=load_panel() if panel is None else panel
    t=pd.Timestamp(t)
    j=data.index.searchsorted(t,side='left')-1
    if j < 0:
        raw=pd.Series(index=data.columns,dtype=object)
        observed=pd.NaT
    else:
        raw=data.iloc[j].astype(object)
        observed=data.index[j]
    if codes is not None: raw=raw.reindex(codes)
    mapped=[map_legacy8(x) for x in raw]
    result=pd.DataFrame({'exchange_label':raw,'legacy8_proxy':[x[0] for x in mapped],
                         'mapping_status':[x[1] for x in mapped]},index=raw.index)
    result.index.name='code'
    result['observation_date']=observed
    result['signal_date']=t
    result['historical_membership_verified']=False
    return result
