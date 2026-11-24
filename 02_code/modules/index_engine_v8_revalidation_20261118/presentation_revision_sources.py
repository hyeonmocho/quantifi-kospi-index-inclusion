from pathlib import Path
import pandas as pd,requests,hashlib,json,concurrent.futures
import fitz
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'04_results/presentation_revision_20261118';OUT.mkdir(exist_ok=True)
RAW=ROOT/'01_data/raw/broker_reports/additions_20261118';RAW.mkdir(parents=True,exist_ok=True)
reg=pd.read_csv(ROOT/'04_results/index_forecast_broker_20261118_more/new_source_registry.csv')
def fetch(r):
 result=r.copy();path=RAW/(r['report_id']+'.pdf')
 try:
  if path.exists():b=path.read_bytes();url=r['source_url'];status=200
  else:
   resp=requests.get(r['source_url'],timeout=30,headers={'User-Agent':'Mozilla/5.0'});status=resp.status_code;b=resp.content;url=resp.url
  result.update(http_status=status,resolved_url=url)
  if b[:4]==b'%PDF':
   path.write_bytes(b);doc=fitz.open(path)
   text='\n'.join(f'\n--- PAGE {i+1} ---\n'+p.get_text() for i,p in enumerate(doc))
   path.with_suffix('.txt').write_text(text,encoding='utf8')
   result.update(local_pdf_downloaded='yes',local_path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),pages=len(doc),status='downloaded')
  else:
   (RAW/(r['report_id']+'.html')).write_bytes(b)
   result.update(status='html_or_error',local_pdf_downloaded='no')
 except Exception as ex:result.update(status='failed',reason=str(ex)[:220])
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(fetch,reg.to_dict('records')))
pd.DataFrame(rows).to_csv(OUT/'source_download_audit.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows)[['report_id','status']].to_string(index=False))
