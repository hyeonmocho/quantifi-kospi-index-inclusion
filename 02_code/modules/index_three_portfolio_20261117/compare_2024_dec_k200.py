"""Compare the 2024-12 cohort with KOSPI200 on its actual holding dates."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "04_results/index_three_portfolio_20261117"
OUT = SRC / "2024_dec_kospi200_comparison.csv"


def main():
    ledger = pd.read_parquet(SRC / "ledger.parquet")
    ledger = ledger[(ledger.scenario == "sharpe_selected") & ledger.event.eq("2024-12")]
    events = pd.read_csv(SRC / "events.csv")
    events = events[(events.scenario == "sharpe_selected") & events.event.eq("2024-12")]
    holdings = pd.read_parquet(SRC / "holdings.parquet")
    holdings = holdings[holdings.scenario.eq("sharpe_selected")]
    price = pd.read_parquet(ROOT / "01_data/processed/calendar_main/benchmark_fnguide_2000_20261029/kospi200_price.parquet").price
    tr = pd.read_parquet(ROOT / "01_data/processed/calendar_main/data_v2_csv_20261029/kospi200_tr.parquet").iloc[:, 0]
    index = pd.DataFrame({"price": price, "tr": tr}).dropna().sort_index()
    rows = []
    for sleeve in ["KOSPI100", "KOSPI200", "KRX300", "COMBINED"]:
        trades = ledger if sleeve == "COMBINED" else ledger[ledger.index_id.eq(sleeve)]
        first_buy = pd.Timestamp(trades[trades.side.eq("buy")].date.min())
        last_sell = pd.Timestamp(trades[trades.side.eq("sell")].date.max())
        assert first_buy in index.index and last_sell in index.index
        dates = index.loc[first_buy:last_sell].index
        h = holdings[holdings.date.between(first_buy, last_sell)]
        if sleeve != "COMBINED":
            h = h[h.index_id.eq(sleeve)]
        exposure = h.groupby("date").market_value.sum().reindex(dates, fill_value=0.)
        assert exposure.iloc[-1] < .01
        entry_principal = 9e9 if sleeve == "COMBINED" else 3e9
        actual_pnl = events.pnl.sum() if sleeve == "COMBINED" else events.loc[events.index_id.eq(sleeve), "pnl"].sum()
        item = dict(sleeve=sleeve, first_buy=first_buy.date(), last_sell=last_sell.date(),
                    buy_trading_days=int(trades[trades.side.eq("buy")].date.nunique()),
                    sell_trading_days=int(trades[trades.side.eq("sell")].date.nunique()),
                    principal_eok=entry_principal / 1e8,
                    strategy_net_return=actual_pnl / entry_principal,
                    strategy_profit_eok=actual_pnl / 1e8)
        for kind in ["price", "tr"]:
            r = index.loc[dates, kind].pct_change(fill_method=None).fillna(0)
            item[f"k200_{kind}_full_window"] = index.loc[last_sell, kind] / index.loc[first_buy, kind] - 1
            item[f"k200_{kind}_exposure_matched"] = (exposure.shift(1).fillna(0) * r).sum() / entry_principal
            item[f"excess_vs_{kind}_matched_pp"] = (item["strategy_net_return"] - item[f"k200_{kind}_exposure_matched"]) * 100
        rows.append(item)
    result = pd.DataFrame(rows)
    assert np.isclose(result.loc[result.sleeve.eq("COMBINED"), "strategy_profit_eok"].iloc[0], -11.10395243465564, atol=.001)
    assert np.isclose(result.loc[result.sleeve.ne("COMBINED"), "strategy_profit_eok"].sum(), result.iloc[-1].strategy_profit_eok)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
