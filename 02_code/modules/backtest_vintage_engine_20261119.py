"""Re-run dated mechanical inclusion forecasts on existing as-of review frames."""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"02_code/modules"))
import index_vintage_engine_20261119 as v

OUT=ROOT/"04_results/index_method_vintage_20261119"
OUT.mkdir(parents=True,exist_ok=True)
FRAME_ROOT={
 "KOSPI200":ROOT/"04_results/index_two_axes_20261117/k200_long/frames",
 "KOSPI100":ROOT/"04_results/k100_long_20261117/frames",
 "KRX300":ROOT/"04_results/index_two_axes_20261117/three_indices/frames",
}
START={"KOSPI200":"2020-06","KOSPI100":"2021-06","KRX300":"2025-06"}


def record(index_id,case,meta,parent=None,tmi_parent_override=None):
    row=meta.iloc[0]
    rule=v.vintage(index_id,row.event,row.information_date,row.effective_date)
    result={"index_id":index_id,"case_id":case,"event":row.event,"lead":int(row.lead),
            "signal_date":pd.Timestamp(row.information_date),"review_date":pd.Timestamp(row.effective_date),
            "rule_industry":rule.industry,"rule_parent":rule.parent,"rule_source":rule.source,
            "rule_status":rule.status,"status":"unavailable"}
    if rule.status.startswith(("uncertified","rule_not_known","index_not_launched")):
        result["status"]=rule.status;return result,[]
    path=FRAME_ROOT[index_id]/(case+".pkl")
    if not path.exists():result["status"]="missing_review_frame";return result,[]
    obj=pd.read_pickle(path);f=obj["frame"].copy()
    if index_id=="KOSPI100":
        observed=meta.set_index("code")
        f["cur"]=observed.current_member.reindex(f.index).fillna(False).astype(bool)
    if index_id=="KRX300":
        observed=meta.set_index("code")
        f["cur"]=observed.current_member.reindex(f.index).fillna(False).astype(bool)
        f["eligibility_state"]=observed.eligibility.reindex(f.index).fillna("unknown")
    if rule.industry.startswith("KRICS") and not f.sector_status.astype(str).str.contains("KRICS").all():
        result["status"]="KRICS_asof_sector_data_missing";return result,[]
    try:
        if index_id=="KOSPI200":full,audit=v.select_kospi200(f,rule)
        elif index_id=="KOSPI100":
            if parent is None:result["status"]="predicted_KOSPI200_parent_missing";return result,[]
            full,audit=v.select_kospi100(f,parent,rule)
        else:
            if rule.parent!="upcoming_midlarge_TMI":
                result["status"]="historical_BMI_parent_missing";return result,[]
            roster=tmi_parent_override if tmi_parent_override is not None else v.tmi_asof(row.information_date)
            if roster is None:result["status"]="asof_TMI_parent_missing";return result,[]
            full,audit=v.select_krx300_current_parent_only(f,rule,roster)
    except (ValueError,KeyError) as exc:
        result["status"]="selection_error:"+str(exc)[:90];return result,[]
    if index_id=="KOSPI100" and audit.get("selection_underfill",0):
        result.update(status="underfilled_KOSPI100_vintage",**audit)
        return result,[]
    cur=set(f.index[f.cur]);new=full-cur
    # Keep the original strategy's explicit regular-entry exclusions and
    # candidate gate; this is not an official index inclusion rule.
    opportunity=set(meta.loc[meta.monitor_candidate,"code"])
    trades=new & opportunity
    original=set(obj["baseline"])
    truth=set(meta.loc[meta.actual_addition.eq(1),"code"])
    result.update(status="mechanical_replayed" if index_id!="KRX300" else (
                  "projected_TMI_parent_research" if tmi_parent_override is not None else "asof_current_TMI_parent_sensitivity"),
                  full_selected=len(full),current=len(cur),mechanical_new=len(new),
                  predicted=len(trades),hits=len(trades&truth),actual=len(truth),
                  original_predicted=len(original),original_hits=len(original&truth),
                  selection_same_as_original=trades==original,
                  forecast_added=len(trades-original),forecast_removed=len(original-trades),
                  **audit)
    names=[{"index_id":index_id,"case_id":case,"event":row.event,"lead":int(row.lead),
            "code":c,"selected":True,"actual_addition":c in truth,
            "original_selected":c in original,"signal_date":row.information_date,
            "entry_date":row.entry_date,"exit_date":row.exit_date}
           for c in sorted(trades)]
    return result,names


def main():
    panel=pd.read_parquet(ROOT/"04_results/index_engine_v6_20261118/candidate_panel.parquet")
    denom=pd.read_csv(ROOT/"04_results/index_engine_v6_20261118/candidate_coverage.csv").set_index("case_id").actual.to_dict()
    panel=panel[panel.event.ge(panel.index_id.map(START))]
    rows=[];targets=[];parents={}
    cases={(r.index_id,r.event,int(r.lead)):cid for cid,r in
           ((cid,g.iloc[0]) for cid,g in panel.groupby("case_id"))}
    for idx in ["KOSPI200","KOSPI100","KRX300"]:
        groups=panel[panel.index_id.eq(idx)].groupby("case_id",sort=True)
        for case,g in groups:
            r=g.iloc[0];parent=None
            if idx=="KOSPI100":
                key=("KOSPI200",r.event,int(r.lead));pcase=cases.get(key)
                parent=parents.get(pcase)
            record_row,names=record(idx,case,g,parent)
            if "actual" in record_row:record_row["actual"]=int(denom.get(case,record_row["actual"]))
            if idx=="KOSPI200" and record_row["status"]=="mechanical_replayed":
                src=pd.read_pickle(FRAME_ROOT[idx]/(case+".pkl"))["frame"]
                spec=v.vintage(idx,r.event,r.information_date,r.effective_date)
                parents[case]=v.select_kospi200(src,spec)[0]
            rows.append(record_row);targets.extend(names)
    summary=pd.DataFrame(rows);t=pd.DataFrame(targets)
    summary.to_csv(OUT/"vintage_replay_case_audit.csv",index=False,encoding="utf-8-sig")
    t.to_parquet(OUT/"vintage_replay_targets.parquet",index=False)
    live=summary[summary.status.isin(["mechanical_replayed","asof_current_TMI_parent_sensitivity"])]
    aggregate=[]
    for (idx,status),g in live.groupby(["index_id","status"]):
        aggregate.append({"index_id":idx,"status":status,"cases":len(g),"events":g.event.nunique(),
                          "predicted":int(g.predicted.sum()),"hits":int(g.hits.sum()),
                          "actual_case_denominator":int(g.actual.sum()),
                          "precision":g.hits.sum()/g.predicted.sum() if g.predicted.sum() else np.nan,
                          "recall":g.hits.sum()/g.actual.sum() if g.actual.sum() else np.nan,
                          "changed_cases":int(g.selection_same_as_original.eq(False).sum())})
    pd.DataFrame(aggregate).to_csv(OUT/"vintage_replay_accuracy.csv",index=False,encoding="utf-8-sig")
    report={"status":"historical rule replay on existing as-of frames",
            "claims_not_supported":["complete KRX discretionary selection certification",
                "KRX300 upcoming TMI forecast from current parent only",
                "pre-2020 KOSPI200, pre-2021 KOSPI100, pre-2025 KRX300 exact vintage backtest"],
            "candidate_monitor_filter":"existing research gate, separate from official methodology",
            "input_panel":"04_results/index_engine_v6_20261118/candidate_panel.parquet",
            "TMI_evidence":"01_data/processed/index_membership_20261119/manifest.json"}
    (OUT/"vintage_replay_protocol.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(summary.groupby(["index_id","status"]).size().to_string())
    print(pd.DataFrame(aggregate).to_string(index=False))

if __name__=="__main__":main()
