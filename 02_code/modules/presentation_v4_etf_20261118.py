"""Same-date historical ETF cross-sections; exact tracker names, no survivor list."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,re,hashlib
import pandas as pd,requests
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'04_results/presentation_v4_20261118';OUT.mkdir(exist_ok=True)
RAW=ROOT/'01_data/raw/presentation_etf_aum_20261118';RAW.mkdir(exist_ok=True)
URL='https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd'
def main():
 text=(ROOT/'KRX_API_KEY.txt').read_text(encoding='utf-8-sig');keys=[]
 for line in text.splitlines():
  line=line.strip()
  if not line or line.startswith('#'):continue
  if '=' in line and line.split('=',1)[0].strip() in ('KRX_API_KEY','KRX_AUTH_KEY','AUTH_KEY'):keys.append(line.split('=',1)[1].strip().strip('"\''))
  elif re.fullmatch(r'[A-Za-z0-9_+./=-]{20,}',line):keys.append(line)
 assert len(set(keys))==1
 def fetch(year):
  target=pd.Timestamp(f'{year}-12-30') if year<2026 else pd.Timestamp('2026-11-17')
  for days in range(7):
   date=(target-pd.Timedelta(days=days)).strftime('%Y%m%d');f=RAW/(date+'.parquet')
   if f.exists():d=pd.read_parquet(f)
   else:
    try:
     r=requests.get(URL,params={'basDd':date},headers={'AUTH_KEY':keys[0]},timeout=(8,30))
     if r.status_code!=200:return dict(year=year,status='http_'+str(r.status_code)),None
     data=r.json().get('OutBlock_1',[])
     if not data:continue
     d=pd.DataFrame(data)
     if set(d.BAS_DD.astype(str))!={date}:continue
     d.to_parquet(f,index=False)
    except (requests.RequestException,ValueError):return dict(year=year,status='network_or_schema_error'),None
   values=pd.to_numeric(d.INVSTASST_NETASST_TOTAMT.astype(str).str.replace(',',''),errors='coerce')
   if values.isna().any():continue
   return dict(year=year,status='ok',date=date,rows=len(d),file=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()),d
  return dict(year=year,status='empty'),None
 audits=[];summary=[];products=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for a,d in pool.map(fetch,range(2010,2027)):
   audits.append(a);print(a['year'],a['status'],a.get('date'),flush=True)
   if d is None:continue
   d['aum_krw']=pd.to_numeric(d.INVSTASST_NETASST_TOTAMT.astype(str).str.replace(',',''),errors='coerce');assert d.aum_krw.notna().all() and d.ISU_CD.is_unique
   key=d.IDX_IND_NM.astype(str).str.replace(r'\s+','',regex=True).str.upper()
   mapping={'코스피200':'KOSPI200','코스피200TR':'KOSPI200','KOSPI200':'KOSPI200','KOSPI200TR':'KOSPI200','코스피100':'KOSPI100','KOSPI100':'KOSPI100','KRX300':'KRX300','KRX300TR':'KRX300'}
   d['index_id']=key.map(mapping);plain=~d.ISU_NM.str.contains('액티브|레버리지|인버스|커버드콜|선물|합성',regex=True,na=False)
   q=d[d.index_id.notna()&plain].copy();q['year']=a['year'];products.append(q)
   summary.append(dict(year=a['year'],date=a['date'],index_id='전체 상장 ETF',aum_krw=d.aum_krw.sum(),products=len(d)))
   for idx in ['KOSPI200','KOSPI100','KRX300']:
    g=q[q.index_id.eq(idx)];summary.append(dict(year=a['year'],date=a['date'],index_id=idx,aum_krw=g.aum_krw.sum(),products=len(g)))
 pd.DataFrame(audits).to_csv(OUT/'etf_collection_audit.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(summary).to_csv(OUT/'etf_aum_yearly.csv',index=False,encoding='utf-8-sig')
 if products:pd.concat(products).to_csv(OUT/'etf_products_yearly.csv',index=False,encoding='utf-8-sig')
 (OUT/'etf_definition.json').write_text(json.dumps(dict(source=URL,retrieved=datetime.now(timezone.utc).isoformat(),measure='net assets INVSTASST_NETASST_TOTAMT, not secondary trading value or net inflow',universe='All ETFs at each requested date; no current product survivorship filter',target_groups='exact index name, KOSPI200 price and TR combined',excluded='active leveraged inverse covered-call futures synthetic names; non-ETF funds not covered',market_total='All listed ETFs, including foreign underlying, bonds and active; do not call all passive or all Korean stock demand',zeros='no classified products that date; KRX300 pre-launch unavailable/not applicable, not backfilled',yearend='last available date on or before Dec30; 2026 asof Nov17'),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
