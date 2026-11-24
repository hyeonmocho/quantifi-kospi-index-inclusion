"""Supplement verified exchange ratios; keep raw/adjusted units consistent."""
from context import ROOT,OUT,RAW,dump
import pandas as pd,numpy as np,requests,json,hashlib,importlib.util
from bs4 import BeautifulSoup

SOURCES={
 'meritz_securities_exchange':'https://kind.krx.co.kr/external/2022/12/05/000477/20221205000999/11347.htm',
 'samsung_imaging_distribution':'https://www.samsung.com/global/ir/reports-disclosures/public-disclosure-view.70398/',
 'hd_mipo_hhi_merger':'https://kind.krx.co.kr/external/2025/08/28/000553/20250828001355/10081.htm'}

def collect():
    records=[]
    for name,url in SOURCES.items():
        p=RAW/(name+'.html')
        if not p.exists():
            r=requests.get(url,timeout=(10,40));r.raise_for_status();p.write_bytes(r.content)
        p.with_suffix('.txt').write_text(BeautifulSoup(p.read_bytes(),'html.parser').get_text(' ',strip=True),encoding='utf8')
        records.append(dict(name=name,url=url,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    spec=importlib.util.spec_from_file_location('market_api',ROOT/'02_code/modules/combo_team/update_market_daily.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for date in ['20230331','20230425']:
        p=RAW/(date+'_stk.json')
        if not p.exists():
            r=requests.get(m.BASE+'stk_bydd_trd',params={'basDd':date},headers={'AUTH_KEY':m.private_key()},timeout=(10,30))
            if r.status_code!=200:raise RuntimeError('KRX HTTP '+str(r.status_code))
            j=r.json();assert isinstance(j.get('OutBlock_1'),list);p.write_text(json.dumps(j,ensure_ascii=False),encoding='utf8')
        records.append(dict(date=date,source='KRX API stk_bydd_trd',path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump('corporate_supplement_sources',records)

def supplement(px,tv,actions):
    raw=pd.read_parquet(ROOT/'01_data/processed/index_data_extension_20261117/raw_quotes_with_turnover.parquet')[['date','code','close']]
    rows=[]
    for path in RAW.glob('*_stk.json'):
        for r in json.loads(path.read_text(encoding='utf8'))['OutBlock_1']:
            rows.append(dict(date=pd.Timestamp(r['BAS_DD']),code='A'+str(r['ISU_CD']).zfill(6),close=float(str(r['TDD_CLSPRC']).replace(',',''))))
    raw=pd.concat([raw,pd.DataFrame(rows)]).drop_duplicates(['date','code'],keep='last').set_index(['date','code'])
    extra=[dict(old='A008560',new='A138040',halt='2023-04-03',effective='2023-04-05',delivery='2023-04-25',old_ref='2023-03-31',ratio=.1607327,source=SOURCES['meritz_securities_exchange'],delivery_basis='announced_new_listing_date'),
           dict(old='A108070',new='A005930',halt='2010-03-23',effective='2010-04-01',delivery='2010-04-16',old_ref='2010-03-22',ratio=.0577663,source=SOURCES['samsung_imaging_distribution'],delivery_basis='conservative_last_day_of_announced_Apr13_16_distribution_window',halt_basis='KRX daily zero volume after last trading day'),
           dict(old='A010620',new='A329180',halt='2025-11-27',effective='2025-12-01',delivery='2025-12-15',old_ref='2025-11-26',ratio=.4059146,source=SOURCES['hd_mipo_hhi_merger'],delivery_basis='KRX filing 2025-08-28: 0.4059146 new HHI shares per old Mipo share; listing 2025-12-15',halt_basis='KRX filing: old share trading halt 2025-11-27 through 2025-12-14')]
    for a in extra:
        for c in ['halt','effective','delivery','old_ref']:a[c]=pd.Timestamp(a[c])
        if a['old']=='A010620':
            # These two observed closes in the FnGuide raw and simulation panels
            # use the same share units. Fail if the input vintage changes.
            assert np.isclose(px.loc[a['old_ref'],a['old']],223000.)
            assert np.isclose(px.loc[a['delivery'],a['new']],551000.)
            of=nf=1.
        else:
            of=px.loc[a['old_ref'],a['old']]/raw.loc[(a['old_ref'],a['old']),'close']
            nf=px.loc[a['delivery'],a['new']]/raw.loc[(a['delivery'],a['new']),'close']
        a['conversion']=of*a['ratio']/nf;a['old_adjusted_to_raw_factor']=of;a['new_adjusted_to_raw_factor']=nf
        assert np.isfinite(a['conversion']) and a['conversion']>0
        if a['old']!='A010620':
            assert np.isclose(a['conversion']*px.loc[a['delivery'],a['new']],of*a['ratio']*raw.loc[(a['delivery'],a['new']),'close'])
        tv.loc[tv.index>=a['halt'],a['old']]=0
    dump('corporate_supplement_actions',extra)
    return px,tv,actions+extra

if __name__=='__main__':collect()
