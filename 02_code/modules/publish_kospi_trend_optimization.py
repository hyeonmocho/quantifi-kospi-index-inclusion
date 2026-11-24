"""Rebuild and publish the main v19 account with its frozen IS trend settings."""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

import compare_kospi_trend_parameters as experiment
import refresh_pair_fnguide_20261123 as latest
import update_pair_submission_20261122 as accounts

ROOT = latest.ROOT
OUT = ROOT / "04_results/kospi_pair_optimized_trend_20261124"
RESEARCH = experiment.OUT
BASE = latest.OUT
PARAMS = {"KOSPI200": (0.25, 0.2), "KOSPI100": (1.0, np.inf)}
PACKAGE = ROOT / "최종"
NOTEBOOK = PACKAGE / "03_notebooks/KOSPI200_KOSPI100_전략_제출용.ipynb"
AUDIT = OUT / "publication_audit"


def replay():
    OUT.mkdir(exist_ok=True)
    locked = json.loads((RESEARCH / "IS_selection.json").read_text())["parameters"]
    assert locked == {"KOSPI200": [0.25, 0.2], "KOSPI100": [1.0, "unbounded"]}
    schedule = pd.read_csv(BASE / "all_forecasts.csv", parse_dates=["entry_date", "exit_date"])
    train = schedule[schedule.event.lt("2023")]
    forecast = experiment.forecast_sets(train, list(PARAMS.values()))
    selected = {r.case_id: forecast[PARAMS[r.index_id]][r.case_id] for r in train.itertuples()}
    print("Optimized historical IS targets generated", flush=True)
    test = pd.read_parquet(RESEARCH / "IS_selected_targets.parquet")
    future_cases = schedule.loc[schedule.event.eq("2026-12"), "case_id"]
    test[test.case_id.isin(future_cases)].to_csv(OUT / "targets.csv", index=False)
    schedule[schedule.case_id.isin(future_cases)].to_csv(OUT / "forecasts.csv", index=False)
    historical = pd.concat([experiment.targets_from(selected), test[~test.case_id.isin(future_cases)]])
    from kospi_pair_h2_information_policy_20261122 import announcement_state
    out = []
    for r in test[test.case_id.isin(future_cases)].itertuples():
        state = announcement_state(r.case_id.split("_")[0], r.code, "2026-11-20")
        out.append({"case_id": r.case_id, "code": r.code, "outcome": state,
                    "first_sale": "2026-11-19" if state == "confirmed_out" else None})
    pd.DataFrame(out).to_csv(OUT / "H2_announcement_decisions.csv", index=False)
    for f in ["input_audit.json", "market_input_audit.json", "historical_input_signal_audit.csv",
              "historical_float_selection_changes.csv", "market_factor_extension.csv", "flow_t40_mean_path.csv"]:
        shutil.copy2(BASE / f, OUT / f)
    for f in ["IS_grid.csv", "IS_selection.json", "OOS_metrics.csv", "OOS_event_buy_changes.csv", "OOS_signal_changes.csv"]:
        shutil.copy2(RESEARCH / f, OUT / f)
    a = pd.read_csv(BASE / "forecast_audit.csv")
    selected_tail = test[test.case_id.isin(future_cases)].groupby("case_id").code.agg(list)
    for i, row in a.iterrows():
        codes = sorted(selected_tail.get(f"{row.index_id}_2026-12_L{row.lead}", []))
        a.loc[i, ["selected", "codes"]] = [len(codes), ";".join(codes)]
    a = a.drop(columns=["parent_selected"])
    a.to_csv(OUT / "forecast_audit.csv", index=False)
    accounts.OUT = OUT
    accounts.PRICE_PANELS = latest.load_panels()
    accounts.full_accounts(historical_targets=historical)
    accounts.full_accounts(cost_only=True, historical_targets=historical)
    m = pd.read_csv(OUT / "metrics.csv")
    expected = pd.read_csv(RESEARCH / "OOS_metrics.csv").query("variant == 'IS_selected'").set_index("index_id")
    keys = ["ending_equity_krw", "cagr", "mdd", "sharpe"]
    np.testing.assert_allclose(m.query("phase == 'OOS' and index_id == 'COMBINED'").iloc[0][keys].astype(float),
                               expected.loc["COMBINED", keys].astype(float), atol=.01, rtol=1e-10)
    s = pd.read_csv(OUT / "standalone_splits.csv").query("n_splits == 4")
    for idx, (alpha, cap) in PARAMS.items():
        p = pd.read_csv(OUT / "IS_grid.csv").query("index_id == @idx and alpha == @alpha and limit == @cap").iloc[0]
        actual = s.query("index_id == @idx and phase == 'IS'").iloc[0]
        np.testing.assert_allclose(actual[keys].astype(float), p[keys].astype(float), atol=.01, rtol=1e-10)
        actual = s.query("index_id == @idx and phase == 'OOS'").iloc[0]
        np.testing.assert_allclose(actual[keys].astype(float), expected.loc[idx, keys].astype(float), atol=.01, rtol=1e-10)
    finish_replay()


def finish_replay():
    tail_summary()
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    protocol.update(selection=json.loads((RESEARCH / "IS_selection.json").read_text())["parameters"],
                    parameter_selection="Standalone IS CD91-adjusted Sharpe, applied unchanged to OOS; four entries fixed")
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def tail_summary():
    names = pd.Series(pd.read_pickle(ROOT / "01_data/processed/signal_a_cache.pkl")["names"])
    names = names.combine_first(pd.read_csv(BASE / "H2_stock_contribution.csv").drop_duplicates("code").set_index("code").name)
    orders = pd.read_parquet(OUT / "OOS_orders.parquet")
    orders = orders[orders.event.eq("2026-12")]
    daily = pd.read_parquet(OUT / "OOS_daily.parquet")
    lots = pd.read_parquet(OUT / "OOS_lots.parquet").query("event == '2026-12'")
    held = pd.read_parquet(OUT / "OOS_holdings.parquet")
    held = held[held.date.eq(pd.Timestamp("2026-11-20"))]
    rows = []
    for idx, group in daily.groupby("index_id"):
        start, end = group[group.date.le("2026-06-30")].iloc[-1], group.iloc[-1]
        z = lots if idx == "COMBINED" else lots[lots.index_id.eq(idx)]
        h = held if idx == "COMBINED" else held[held.index_id.eq(idx)]
        rows.append(dict(index_id=idx, starting_equity=start.equity, ending_equity=end.equity,
                         H2_pnl=end.equity-start.equity, H2_account_return=end.equity/start.equity-1,
                         H2_return_per_fixed_budget=(end.equity-start.equity)/(2e10 if idx == "COMBINED" else 1e10),
                         ending_stock_value=end.held_value, distinct_codes=h.code.nunique(),
                         fill_ratio=z.entry_notional.sum()/z.initial_target_notional.sum()))
    pd.DataFrame(rows).to_csv(OUT / "H2_contribution.csv", index=False)
    group = ["index_id", "code"]
    buys = orders[orders.side.eq("buy")].groupby(group)[["notional", "fee"]].sum()
    sales = orders[orders.side.eq("sell")].groupby(group)[["notional", "fee"]].sum()
    stocks = buys.join(held.set_index(group).market_value).fillna({"market_value": 0})
    stocks["sale_proceeds"] = sales.notional.reindex(stocks.index, fill_value=0)
    stocks["sale_fees"] = sales.fee.reindex(stocks.index, fill_value=0)
    stocks["pnl"] = stocks.market_value + stocks.sale_proceeds - stocks.sale_fees - stocks.notional - stocks.fee
    stocks["net_return_on_paid_cash"] = stocks.pnl / (stocks.notional + stocks.fee)
    stocks = stocks.reset_index()
    stocks["name"] = stocks.code.map(names)
    assert stocks.loc[stocks.market_value.gt(0), "name"].notna().all(), stocks.loc[stocks.name.isna(), "code"].tolist()
    assert abs(stocks.pnl.sum()-lots.pnl.sum()) < .02
    stocks.to_csv(OUT / "H2_stock_contribution.csv", index=False, encoding="utf-8-sig")


def factor_results():
    from kospi_pair_full_oos_factor_20261122 import fit, NAMES

    columns = ["MKT_RF", "SMB", "HML"]
    factors = pd.read_csv(ROOT / "04_results/kospi_pair_full_oos_factor_20261123/factor_inputs.csv",
                          parse_dates=["date"])[["date", "RF", *columns]]
    rows = []
    for idx in NAMES:
        key = "OOS" if idx == "COMBINED" else f"{idx}_OOS"
        d = pd.read_parquet(OUT / f"{key}_daily.parquet").query("index_id == @idx")
        d = d.sort_values("date").merge(factors, on="date", how="left", validate="one_to_one")
        assert len(d) == 948 and d[["RF", *columns]].notna().all().all()
        d["excess_return"] = d.return_daily - d.RF
        fit_result, _ = fit(d, lag=20, columns=columns)
        rows.append({"전략": NAMES[idx], **fit_result})
    a = pd.DataFrame(rows)
    a = a[["전략", "start", "end", "n_days", "alpha_annual", "alpha_t", "alpha_p", "ci_low", "ci_high"]]
    a.columns = ["전략", "시작일", "종료일", "거래일 수", "연율 알파", "t값", "p값", "95% 하한", "95% 상한"]
    a.to_csv(OUT / "oos_ff3_robustness.csv", index=False, encoding="utf-8-sig")
    return a


def submission():
    import nbformat
    from nbclient import NotebookClient
    import broker_stock_trade_stats_20261121 as broker
    import package_pair_update_20261122 as package

    AUDIT.mkdir(exist_ok=True)
    for p in (NOTEBOOK, PACKAGE / "README_제출안내.md"):
        if not (AUDIT / p.name).exists():
            shutil.copy2(p, AUDIT / p.name)
    shutil.copytree(RESEARCH / "broker_comparison", OUT / "broker_comparison", dirs_exist_ok=True)
    broker.OUT = OUT / "broker_comparison"
    broker.main()
    # Require exact stock lists and reported results before preserving the worked comparison.
    for file in ["pairs.csv", "summary.csv", "stock_trade_summary.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(broker.OUT / file), pd.read_csv(BASE / "broker_comparison" / file))
    factor = factor_results()
    package.OUT = OUT
    package.DEST = PACKAGE / "04_results" / OUT.name
    package.FNGUIDE_MARKET = True
    package.export(verify_previous_snapshot=False)
    book = nbformat.read(AUDIT / NOTEBOOK.name, 4)
    for c in book.cells:
        c.source = c.source.replace(BASE.name, OUT.name)
    book.cells[0].source = book.cells[0].source.replace("2026년 11월 23일 갱신", "추세 파라미터 갱신")
    book.cells[0].source = book.cells[0].source.replace(
        "추세보정 엔진과 80·60·40·20거래일 전 4회 매수 원칙을 유지했습니다.",
        "인샘플 샤프로 고른 시총 추세보정 값은 코스피200 ¼·±0.2, 코스피100 1.0·한도 없음입니다. 심사마감 80·60·40·20거래일 전 4회 신호를 사용합니다.")
    intro = book.cells[4].source.split("신호일까지 관측한")[0].rstrip()
    book.cells[4].source = intro + """

신호일까지 관측한 시총과 거래일 수를 고정합니다. 최근 20거래일 수익률을 $r$, 남은 심사 거래일을 $D$라 하면 보정 전 로그값은 $g_0=\\alpha[\\ln(1+r)/20](D/2)$입니다.
KOSPI200은 $\\alpha=1/4$와 $g=\\operatorname{clip}(g_0,-0.2,0.2)$, KOSPI100은 $\\alpha=1.0$과 $g=g_0$를 사용합니다. 예상 잔여 평균 시총은 신호일 시총에 $e^g$를 곱하고, 심사기간 전체의 평균은 관측 구간과 잔여 구간을 거래일 수로 가중해 구합니다.

KOSPI200의 상·하한은 예측일 시총 대비 약 −18.1%~+22.1%의 잔여 평균 시총 보정 범위입니다. KOSPI100은 예상 모지수를 만드는 과정부터 자체 계수를 사용합니다. 이 계수와 한도는 거래소 선정 요건이나 논문의 추정값과 구별되는 프로젝트 설정입니다.
"""
    book.cells[5].source = book.cells[5].source.replace(
        "추세 계수 0.25와 네 차수의 주문은 국내 연구에서 설정했습니다.",
        "지수별 추세 계수·한도와 네 차수의 주문은 국내 연구에서 설정했습니다. 현대오토에버의 첫 신호에서는 새 ±0.2 한도가 작동하지 않았으며, OOS의 코스피200 종목과 체결 내역도 기존과 같습니다.")
    book.cells[5].source = book.cells[5].source.replace(
        "../04_results/common_index_engine_20261119/targets.csv", f"../04_results/{OUT.name}/all_targets.csv").replace(
        "../04_results/kospi_pair_policy_20261121/OOS_orders.parquet", f"../04_results/{OUT.name}/KOSPI200_OOS_orders.parquet")
    book.cells[6].source = book.cells[6].source.replace(" OOS는 2023년 1월~2026년 6월입니다.", "")
    book.cells[6].source += """

### 추세 계수와 한도의 인샘플 선택

각 지수의 4회 분할 단독 계좌에서 추세 계수 0·0.10·0.25·0.50·0.75·1.0과 한도 0.2·0.3·0.4·0.6·무제한을 비교했습니다. 계수 0에서는 한도가 영향을 주지 않아 한 번만 계산했습니다. 지수당 26개 조합 중 CD91 차감 샤프로 값을 고른 뒤 2023년 이후에는 고정했습니다. 아래 진입 횟수·거래비용 표에서도 두 값은 고정했습니다.

코스피100 2006~2022년 단독 IS는 기존 샤프 0.4383에서 0.4507로 올랐습니다. 예전 v18의 0.730은 2006~2016년 10회, 다른 후보 선정법과 거래 규칙, 무위험수익률 미차감 기준의 수치입니다.
"""
    text = book.cells[7].source
    first = text.index("| OOS")
    last = text.index("\n\n과거", first)
    z = pd.read_csv(OUT / "metrics.csv").query("phase == 'OOS'").set_index("index_id")
    table = ["| OOS · 2023-01-02~2026-11-20 | KOSPI200 계정 | KOSPI100 계정 | 통합계좌 |",
             "|---|---:|---:|---:|"]
    for title, key, scale, fmt in [
        ("종료 자산(억원)", "ending_equity_krw", 1e8, ".2f"), ("순이익(억원)", "profit_krw", 1e8, ".2f"),
        ("누적 순수익률", "total_return", 1, ".2%"), ("CAGR", "cagr", 1, ".2%"),
        ("연율 변동성", "volatility", 1, ".2%"), ("샤프(CD 차감)", "sharpe", 1, ".3f"), ("MDD", "mdd", 1, ".2%")]:
        table.append("| " + title + " | " + " | ".join(format(z.at[idx, key]/scale, fmt) for idx in (*PARAMS, "COMBINED")) + " |")
    book.cells[7].source = text[:first] + "\n".join(table) + text[last:]
    book.cells[28].source = book.cells[28].source.replace("'parent_selected',", "")
    book.cells[30].source = book.cells[30].source.replace(".read_csv(OUT/'oos_ff3_robustness.csv')",
        ".read_csv(OUT/'oos_ff3_robustness.csv').query(\"전략 == '두 전략 통합'\").copy()")
    book.cells[29].source = book.cells[29].source.replace(
        "계좌별 일일 NAV 수익률을 사용했습니다. KOSPI200·KOSPI100 단독 계좌와 거래대금 한도를 공유하는 통합계좌를 각각 회귀했습니다.",
        "통합계좌의 일일 NAV 수익률을 사용했습니다. 두 전략은 동일 종목·동일 거래일의 거래대금 한도를 공유합니다.")
    ff = factor.iloc[-1]
    book.cells[31].source = (
        "### 유의성 해석\n\n"
        f"통합계좌의 연율 알파는 **{ff['연율 알파']:+.2%}**이며 95% 신뢰구간은 "
        f"{ff['95% 하한']:+.2%}~{ff['95% 상한']:+.2%}입니다. "
        f"양측 t검정 결과 t={ff['t값']:.3f}, p={ff['p값']:.4f}입니다."
    )
    book.cells.insert(9, nbformat.v4.new_code_cell(
        """grid = pd.read_csv(OUT/'IS_grid.csv')
grid['선택'] = ((grid.index_id.eq('KOSPI200') & grid.alpha.eq(.25) & grid.limit.eq(.2)) |
              (grid.index_id.eq('KOSPI100') & grid.alpha.eq(1) & grid.limit.eq(float('inf'))))
display(grid.pivot(index=['index_id','limit'], columns='alpha', values='sharpe').round(4))
display(grid.loc[grid['선택'], ['index_id','alpha','limit','sharpe','mdd']].round(4))"""))
    NotebookClient(book, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(PACKAGE)}}).execute()
    nbformat.validate(book)
    nbformat.write(book, NOTEBOOK)
    print("Executed optimized strategy notebook", flush=True)


def presentation():
    import update_ppt_nov20_20261122 as ppt
    import finalize_ppt_nov20_20261122 as validation
    import win32com.client

    AUDIT.mkdir(exist_ok=True)
    ppt.SOURCE = OUT
    validation.src = OUT
    primary = ppt.PATHS[1]
    backup = AUDIT / primary.name
    ff = pd.read_csv(OUT / "oos_ff3_robustness.csv").iloc[-1]
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = next((p for p in app.Presentations if str(p.FullName).casefold() == str(primary).casefold()), None)
    opened = pres is None
    if opened:
        pres = app.Presentations.Open(str(primary), False, False, False)
    assert pres.Saved and pres.Slides.Count == 20
    if not backup.exists():
        shutil.copy2(primary, backup)
    affected = (8, 9, 12, 13, 14, 15, 16, 17, 18)
    for n in affected:
        file = AUDIT / f"before_{n}.png"
        if not file.exists():
            pres.Slides(n).Export(str(file), "PNG", 1200, 675)
    schedule_text = ppt.shape(pres.Slides(4), 3).TextFrame.TextRange.Text
    assert "2025" in schedule_text
    ppt.edit(pres, strategy_only=True)
    s = pres.Slides(8)
    ppt.text(s, 35, "KOSPI200  α=¼ · g=clip(g₀, ±0.2)\vKOSPI100  α=1.0 · g=g₀", 13)
    ppt.text(s, 40, "g₀ = α ×")
    ppt.text(s, 44, "")
    ppt.text(s, 51, "③ 지수별 추세 반영")
    ppt.note(s, "예측일 시총을 C, 최근 20거래일 수익률을 r, 남은 거래일을 D로 둡니다. 하루당 로그 추세에 남은 기간의 중간점 D 나누기 2와 추세 반영 계수를 곱합니다. 코스피200은 4분의 1만 반영한 뒤 로그 보정값을 마이너스 0.2에서 플러스 0.2로 제한합니다. 코스피100은 같은 중간점에 추세를 전부 반영하고 별도 상·하한은 두지 않습니다. 4회 분할을 고정한 상태에서 지수별 26개 조합을 비교해 표본내 CD91 차감 샤프가 가장 높은 값을 선택했습니다. 남은 기간 예상 평균은 C 곱하기 exp(g)입니다. 이미 관측한 시총 평균과 거래일 수로 가중해 전체 심사기간 평균을 계산합니다.")
    s = pres.Slides(9)
    ppt.text(s, 24, "남은 80일 평균 = 4.643 × exp(g) ≈ 5.079조 원  ·  g = ¼ × ln(1+19.65%) ÷ 20 × (80÷2)")
    for sh in s.NotesPage.Shapes:
        if sh.Type == 14 and sh.PlaceholderFormat.Type == 2:
            notes = sh.TextFrame.TextRange.Text.split("\r\n\r\n이 사례의")[0]
            if "±0.2 한도" not in notes:
                sh.TextFrame.TextRange.Text = notes + "\r\r이 사례의 첫 신호는 로그 보정값이 약 0.090으로 ±0.2 한도 이내라 계산값과 편입 후보 순위가 유지됩니다."
            break
    for n, detail in [(12, "추세 계수 ¼ · 로그 한도 ±0.2"),
                      (13, "추세 계수 1.0 · 로그 한도 없음")]:
        s = pres.Slides(n)
        ppt.text(s, 22, f"80·60·40·20거래일 전 후보 산출  |  {detail}", 12)
    s = pres.Slides(16)
    low, high, pvalue = ff["95% 하한"], ff["95% 상한"], ff["p값"]
    for col, value in enumerate([f"{ff['연율 알파']:+.2%}", f"{ff['t값']:.3f}", f"{pvalue:.4f}",
                                 f"{low:+.2%} ~ {high:+.2%}"], 2):
        ppt.cell(s, 11, 2, col, value)
    assert low < 0 < high and pvalue >= .05
    status = "10% 참고 기준 유의" if pvalue < .10 else "10% 참고 기준에서도 비유의"
    ppt.text(s, 12, f"양의 알파 추정 · 5% 양측검정에서 비유의 · {status}")
    ppt.note(s, f"2023년부터 11월 20일까지 현금을 포함한 통합계좌 일별 순수익률에 3팩터 회귀를 적용했습니다. "
                f"연율 알파는 {ff['연율 알파']:+.2%}, t값은 {ff['t값']:.3f}, 양측 p값은 {pvalue:.4f}였습니다. "
                f"95% 신뢰구간은 {low:+.2%}에서 {high:+.2%}로 0을 포함합니다. CD91을 차감하고 HAC 20거래일 표준오차를 사용했습니다.")
    s = pres.Slides(18)
    ppt.text(s, 3, "2026년 하반기 포트폴리오 · 11월 20일", 29)
    ppt.text(s, 7, "KOSPI200 · 11월 18일 명단 공표", 14)
    ppt.text(s, 11, "KOSPI100 · 확정 명단 미확인", 14)
    fit_snapshot_tables(s, ppt)
    assert ppt.shape(pres.Slides(4), 3).TextFrame.TextRange.Text == schedule_text
    pres.Save()
    for n in affected:
        pres.Slides(n).Export(str(AUDIT / f"after_{n}.png"), "PNG", 1200, 675)
    if opened:
        pres.Close()
    synchronize_decks(ppt, validation)


def fit_snapshot_tables(slide, ppt):
    left = ppt.shape(slide, 17).Table
    right = ppt.shape(slide, 20).Table
    for row in range(2, 10):
        for col in range(1, 6):
            existing = left.Cell(row, col).Shape
            updated = right.Cell(row, col).Shape
            updated.Fill.Solid()
            updated.Fill.ForeColor.RGB = existing.Fill.ForeColor.RGB
            source_frame, frame = existing.TextFrame, updated.TextFrame
            frame.MarginTop, frame.MarginBottom = source_frame.MarginTop, source_frame.MarginBottom
            frame.MarginLeft, frame.MarginRight = source_frame.MarginLeft, source_frame.MarginRight
            frame.VerticalAnchor = source_frame.VerticalAnchor
            t, e = frame.TextRange, source_frame.TextRange
            t.Font.Name, t.Font.Size = e.Font.Name, e.Font.Size
            t.Font.Bold, t.Font.Color.RGB = e.Font.Bold, e.Font.Color.RGB
            t.ParagraphFormat.Alignment = e.ParagraphFormat.Alignment
        right.Rows(row).Height = left.Rows(row).Height


def finish_presentation_layout():
    import update_ppt_nov20_20261122 as ppt
    import finalize_ppt_nov20_20261122 as validation
    import win32com.client

    validation.src = OUT
    app = win32com.client.Dispatch("PowerPoint.Application")
    primary = ppt.PATHS[1]
    pres = next((p for p in app.Presentations if str(p.FullName).casefold() == str(primary).casefold()), None)
    opened = pres is None
    if opened:
        pres = app.Presentations.Open(str(primary), False, False, False)
    assert pres.Saved
    ppt.text(pres.Slides(8), 35, "KOSPI200  α=¼ · g=clip(g₀, ±0.2)\vKOSPI100  α=1.0 · g=g₀", 13)
    fit_snapshot_tables(pres.Slides(18), ppt)
    pres.Save()
    for n in (8, 18):
        pres.Slides(n).Export(str(AUDIT / f"after_{n}.png"), "PNG", 1200, 675)
    if opened:
        pres.Close()
    synchronize_decks(ppt, validation)


def synchronize_decks(ppt, validation):
    primary = ppt.PATHS[1]
    validation.validate(primary, slide_count=20, monthly_slide=17)
    silent = primary.with_name("지수편입예측_발표_v19_대본없음.pptx")
    validation.without_notes(primary, silent)
    working = ppt.PATHS[0]
    design = working.with_name(working.stem + "_제안서디자인.pptx")
    for src, dest in [(primary, working), (primary, design), (silent, design.with_name(design.stem + "_대본없음.pptx"))]:
        shutil.copyfile(src, dest)
    validation.validate(silent, slide_count=20, monthly_slide=17)
    print("Verified optimized presentation and embedded chart data", flush=True)


def refresh_manifest():
    import hashlib

    audit = json.loads((PACKAGE / "04_results" / OUT.name / "daily_portfolio_export_audit.json").read_text(encoding="utf-8"))
    readme = PACKAGE / "README_제출안내.md"
    text = (AUDIT / readme.name).read_text(encoding="utf-8")
    text = text.replace("11월 20일 데이터 갱신 · 2026-11-23", "시총 추세 파라미터 재선택")
    text = text.replace(BASE.name, OUT.name).replace("44,075행", f"{audit['all_rows']:,}행")
    text = text.replace("refresh_pair_fnguide_20261123.py --replay", "publish_kospi_trend_optimization.py --replay")
    text += "\n주전략은 인샘플 CD91 차감 샤프 기준으로 선택한 코스피200 ¼·±0.2, 코스피100 1.0·한도 없음입니다. 두 설정을 고정해 2023년 이후 계좌를 재검증했습니다.\n"
    readme.write_text(text, encoding="utf-8")
    rows = []
    for p in sorted(PACKAGE.rglob("*")):
        if p.is_file() and p.name != "파일목록_SHA256.csv" and not p.name.startswith("~$") and "__pycache__" not in p.parts:
            with p.open("rb") as f:
                rows.append({"file": str(p.relative_to(PACKAGE)), "bytes": p.stat().st_size,
                             "sha256": hashlib.file_digest(f, "sha256").hexdigest()})
    pd.DataFrame(rows).to_csv(PACKAGE / "파일목록_SHA256.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if "--replay" in sys.argv:
        replay()
    elif "--finish-replay" in sys.argv:
        finish_replay()
    elif "--submission" in sys.argv:
        submission()
    elif "--ppt" in sys.argv:
        presentation()
    elif "--ppt-layout" in sys.argv:
        finish_presentation_layout()
    elif "--manifest" in sys.argv:
        refresh_manifest()
