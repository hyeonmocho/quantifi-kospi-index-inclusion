from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'04_results/index_engine_v8_revalidation_20261118'
RAW=ROOT/'01_data/raw/index_engine_v7_20261118'
# Historical adapters only; this directory remains first to prevent cross-version imports.
for name in ['index_engine_v7_20261118','index_engine_v6_20261118']:
 sys.path.append(str(ROOT/'02_code/modules'/name))
def dump(name,x):(OUT/(name+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
