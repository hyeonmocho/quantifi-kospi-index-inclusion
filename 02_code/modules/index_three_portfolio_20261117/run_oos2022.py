"""Rebuild the 30+30+30억원 account from the 2021-frozen KRX300 OOS split."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "04_results/index_forecast_development_20261117"
BASE = ROOT / "04_results/index_forecast_is_oos_20261117"
OUT = ROOT / "04_results/index_three_portfolio_oos2022_20261118"


def year_rows(daily):
    rows = []
    for name, account in daily.groupby("index_id", sort=False):
        account = account.sort_values("date")
        initial = 9e9 if name == "COMBINED" else 3e9
        prior = initial
        for year, g in account.groupby(account.date.dt.year):
            path = np.r_[prior, g.equity.to_numpy()]
            r = path[1:] / path[:-1] - 1
            rows.append(dict(index_id=name, year=int(year), start=str(g.date.iloc[0].date()),
                             end=str(g.date.iloc[-1].date()), start_equity_krw=prior,
                             end_equity_krw=float(g.equity.iloc[-1]),
                             profit_krw=float(g.equity.iloc[-1] - prior),
                             return_on_start_equity=float(g.equity.iloc[-1] / prior - 1),
                             sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) else np.nan,
                             mdd=float((path / np.maximum.accumulate(path) - 1).min()),
                             partial_year=bool(year == 2026)))
            prior = float(g.equity.iloc[-1])
    return pd.DataFrame(rows)


def plot(daily):
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    colors = {"COMBINED": "#1f2937", "KOSPI200": "#2563eb",
              "KOSPI100": "#e58b28", "KRX300": "#139b82"}
    labels = {"COMBINED": "통합 90억 원", "KOSPI200": "코스피200 30억 원",
              "KOSPI100": "코스피100 30억 원", "KRX300": "KRX300 30억 원"}
    for name in ["COMBINED", "KOSPI200", "KOSPI100", "KRX300"]:
        g = daily[daily.index_id.eq(name)].sort_values("date")
        principal = 9e9 if name == "COMBINED" else 3e9
        nav = g.equity / principal
        drawdown = nav / np.maximum.accumulate(np.r_[1., nav.to_numpy()])[1:] - 1
        top.plot(g.date, nav, label=f"{labels[name]}  ({nav.iloc[-1]:.2f}배)",
                 lw=2.8 if name == "COMBINED" else 1.6, color=colors[name])
        bottom.plot(g.date, 100 * drawdown, lw=2.5 if name == "COMBINED" else 1.2,
                    color=colors[name], alpha=1 if name == "COMBINED" else .8)
    top.axhline(1, color="#94a3b8", lw=.8)
    top.set_title("세 지수 통합전략 NAV | 2022-01-03~2026-06-30 OOS", fontsize=15, pad=14)
    top.set_ylabel("초기 원금 = 1")
    top.legend(ncol=2, loc="upper left", frameon=False)
    bottom.set_ylabel("고점 대비 낙폭 (%)")
    bottom.set_xlabel("날짜")
    for ax in [top, bottom]:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.01, .005, "2018~2021 IS 선택 고정 · 각 30억 원 독립 현금계좌 · 동일 종목 일일 유동성 한도 공유 · 비용 차감", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, .02, 1, 1])
    fig.savefig(OUT / "NAV_oos2022.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    sources = [DEV / name for name in ["selected_IS_only.csv", "accuracy.csv", "execution_targets.parquet"]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    spec = importlib.util.spec_from_file_location("oos2022_resources", ROOT / "02_code/modules/index_forecast_is_oos_20261117/evaluate.py")
    resources = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = resources
    spec.loader.exec_module(resources)
    resources.write = lambda name, data: None
    px, tv, actions = resources.resources()
    schedule = pd.read_csv(BASE / "case_schedule.csv", parse_dates=["entry_date", "exit_date", "information_date"])
    acc = pd.read_csv(DEV / "accuracy.csv", low_memory=False)
    acc = acc[acc.train_end.eq(2021)]
    targets = pd.read_parquet(DEV / "execution_targets.parquet")
    targets = targets[targets.train_end.eq(2021)]
    choices = pd.read_csv(DEV / "selected_IS_only.csv")
    choices = choices[choices.train_end.eq(2021) & choices.role.eq("IS_sharpe_selected")]
    assert set(choices.index_id) == {"KOSPI100", "KOSPI200", "KRX300"} and len(choices) == 3
    summary, daily, events, ledger, holdings, checks = engine.simulate(
        choices, schedule, acc, targets, px, tv, actions, shared=True,
        start="2022-01-03", end="2026-06-30")
    assert checks["all_events_closed"] and checks["shared_security_day_cap"]
    assert checks["min_cash"] > -.001 and checks["max_limit_excess"] < .001
    assert all(events.groupby("index_id").size().eq(9))
    assert np.isclose(summary.loc[summary.index_id.eq("COMBINED"), "profit_krw"].iloc[0],
                      summary.loc[summary.index_id.ne("COMBINED"), "profit_krw"].sum(), atol=.01)
    annual = year_rows(daily)
    summary.to_csv(OUT / "performance.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    daily.to_parquet(OUT / "daily.parquet", index=False)
    events.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    ledger.to_parquet(OUT / "ledger.parquet", index=False)
    plot(daily)
    protocol = dict(IS="2018-2021", OOS=["2022-01-03", "2026-06-30"],
                    train_end=2021, choice_role="IS_sharpe_selected", budget_per_sleeve_krw=3e9,
                    shared_liquidity_limit=True, separate_cash_accounts=True,
                    fee_per_side=engine.FEE, participation=engine.PARTICIPATION,
                    profits_not_reinvested_into_entry_cap=True, source_hashes=hashes,
                    checks=checks, note="Different frozen choices from the 2023-start IS 2018-2022 account. Previously inspected OOS, not new forward evidence.")
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf8")
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    print(summary.to_string(index=False), flush=True)
    print(annual[annual.index_id.eq("COMBINED")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
