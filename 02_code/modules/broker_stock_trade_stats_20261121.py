"""Calculate stock-level win rate/payoff from the dated broker execution ledger."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_results/kospi_pair_policy_20261121/broker_comparison"
KEY = ["design", "report_id", "event", "who", "code"]


def main():
    orders = pd.read_csv(OUT / "orders.csv")
    assert set(orders.side.unique()) == {"buy", "sell"}, "Corporate actions require separate lot matching"
    orders["cashflow"] = np.where(orders.side.eq("buy"),
                                  -orders.notional - orders.fee,
                                  orders.notional - orders.fee)
    orders["buy_cost"] = np.where(orders.side.eq("buy"),
                                  orders.notional + orders.fee, 0.)
    orders["sell_proceeds"] = np.where(orders.side.eq("sell"),
                                       orders.notional - orders.fee, 0.)
    stock_report = orders.groupby(KEY, as_index=False).agg(
        pnl=("cashflow", "sum"), buy_cost=("buy_cost", "sum"),
        sell_proceeds=("sell_proceeds", "sum"),
        first_trade=("date", "min"), last_trade=("date", "max"))
    assert stock_report.buy_cost.gt(0).all() and stock_report.sell_proceeds.gt(0).all()
    stock_report["return_on_filled_cost"] = stock_report.pnl / stock_report.buy_cost
    pairs = pd.read_csv(OUT / "pairs.csv")
    check = stock_report.groupby(KEY[:-1]).pnl.sum().div(1e10)
    expected = pairs.set_index(KEY[:-1]).net
    np.testing.assert_allclose(check.loc[expected.index], expected, rtol=0, atol=1e-10)
    stock_report.to_csv(OUT / "stock_report_trades.csv", index=False, encoding="utf-8-sig")

    # A security appearing in multiple reports for the same review is one
    # economic stock-event. Average its different report-date trade returns.
    stock_event = stock_report.groupby(["design", "event", "who", "code"], as_index=False).agg(
        return_on_filled_cost=("return_on_filled_cost", "mean"),
        reports=("report_id", "nunique"), pnl_mean=("pnl", "mean"),
        buy_cost_mean=("buy_cost", "mean"))
    stock_event.to_csv(OUT / "stock_event_trades.csv", index=False, encoding="utf-8-sig")

    rows = []
    for unit, frame in [("stock_report", stock_report), ("stock_event", stock_event)]:
        for (design, who), g in frame.groupby(["design", "who"]):
            r = g.return_on_filled_cost
            positive = r[r > 0]
            negative = r[r < 0]
            rows.append(dict(unit=unit, design=design, who=who,
                             observations=len(r), wins=len(positive), losses=len(negative),
                             win_rate=len(positive)/len(r),
                             payoff=positive.mean()/-negative.mean(),
                             mean_return=r.mean(), median_return=r.median(),
                             reviews=g.event.nunique(), stocks=g.code.nunique()))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "stock_trade_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "stock_trade_protocol.json").write_text(json.dumps({
        "stock_return": "Actual filled sell proceeds minus filled buy cost, divided by filled buy cost; both sides include 25bp cost and liquidity-constrained execution",
        "primary_unit": "one stock x one review event; average that stock's returns from repeated reports in the same event",
        "sensitivity_unit": "one stock x one report",
        "zero_fill": "none in declared_inputs sample",
        "corporate_actions": "none in trade orders",
        "portfolio_mean_net": "remains review-level 100억원 budget return, not comparable directly to equal-stock mean return",
        "independence": "stock observations within one review are correlated; observation count is not independent event count",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[summary.design.eq("declared_inputs")].to_string(index=False))


if __name__ == "__main__":
    main()
