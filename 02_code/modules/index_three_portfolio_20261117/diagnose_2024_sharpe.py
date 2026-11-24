"""Attribute 2024 daily combined NAV to security marks and trade cashflows."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "04_results/index_three_portfolio_20261117"
OUT = SRC / "2024_sharpe_diagnostic"


def metrics(equity, start_equity):
    path = np.r_[start_equity, equity]
    returns = path[1:] / path[:-1] - 1
    std = returns.std(ddof=1)
    return dict(annual_return=equity[-1] / start_equity - 1,
                annual_profit_eok=(equity[-1] - start_equity) / 1e8,
                sharpe=returns.mean() / std * np.sqrt(252) if std else np.nan,
                volatility=std * np.sqrt(252),
                mdd=(path / np.maximum.accumulate(path) - 1).min())


def main():
    OUT.mkdir(exist_ok=True)
    daily = pd.read_parquet(SRC / "daily.parquet")
    daily = daily[(daily.scenario == "sharpe_selected") & (daily.index_id == "COMBINED")].sort_values("date")
    current = daily[daily.date.dt.year.eq(2024)].copy()
    prior = daily[daily.date.lt("2024-01-01")].iloc[-1]
    dates = current.date
    holdings = pd.read_parquet(SRC / "holdings.parquet")
    holdings = holdings[holdings.scenario.eq("sharpe_selected")]
    marks = holdings.pivot_table(index="date", columns=["index_id", "code"], values="market_value", aggfunc="sum").fillna(0)
    marks = marks.reindex(index=pd.DatetimeIndex([prior.date]).append(pd.DatetimeIndex(dates)), fill_value=0)
    mark_change = marks.diff().iloc[1:]
    ledger = pd.read_parquet(SRC / "ledger.parquet")
    ledger = ledger[(ledger.scenario == "sharpe_selected") & ledger.date.dt.year.eq(2024)].copy()
    assert set(ledger.side) <= {"buy", "sell"}
    ledger["cashflow"] = np.where(ledger.side.eq("sell"), ledger.notional, -ledger.notional) - ledger.fee
    cashflow = ledger.pivot_table(index="date", columns=["index_id", "code"], values="cashflow", aggfunc="sum")
    by_security = mark_change.add(cashflow, fill_value=0).fillna(0).reindex(index=dates, fill_value=0)
    actual_daily_pnl = current.equity.to_numpy() - np.r_[prior.equity, current.equity.to_numpy()[:-1]]
    assert np.allclose(by_security.sum(axis=1).to_numpy(), actual_daily_pnl, atol=.01)
    start_equity = float(prior.equity)
    original = metrics(current.equity.to_numpy(), start_equity)
    aggregate = by_security.T.groupby(level="code").sum().T
    records = []
    for code in aggregate.columns:
        excluded_equity = current.equity.to_numpy() - aggregate[code].cumsum().to_numpy()
        result = metrics(excluded_equity, start_equity)
        records.append(dict(code=code, pnl_eok=aggregate[code].sum() / 1e8,
                            contribution_pp=aggregate[code].sum() / start_equity * 100,
                            sharpe_excluding_one=result["sharpe"],
                            annual_return_excluding_one=result["annual_return"],
                            delta_sharpe=result["sharpe"] - original["sharpe"]))
    pd.DataFrame(records).sort_values("pnl_eok").to_csv(OUT / "stock_attribution.csv", index=False, encoding="utf-8-sig")
    by_security.sum(axis=0).rename("pnl_krw").reset_index().sort_values("pnl_krw").to_csv(OUT / "position_attribution.csv", index=False, encoding="utf-8-sig")
    group = by_security.T.groupby(level="index_id").sum().T
    sleeve = {idx: float(group[idx].sum() / 1e8) for idx in group.columns}
    assert np.isclose(sum(sleeve.values()), (current.equity.iloc[-1] - start_equity) / 1e8, atol=1e-7)
    print("original", original)
    print("sleeve_pnl_eok", sleeve)
    print("worst_stock_diagnostics")
    print(pd.DataFrame(records).sort_values("pnl_eok").head(12).to_string(index=False))
    print("top_daily_losses")
    changes = by_security.sum(axis=1)
    for date in changes.nsmallest(5).index:
        print(str(date.date()), "total_eok", round(changes.loc[date] / 1e8, 3),
              "largest_losses", [(i, c, round(v / 1e8, 3)) for (i, c), v in by_security.loc[date].nsmallest(5).items()])


if __name__ == "__main__":
    main()
