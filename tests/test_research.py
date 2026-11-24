import numpy as np
import pandas as pd
import pytest
from nselab.costs import IndiaDeliveryCosts
from nselab.features import (
    training_before,
    build_features,
    PRICE_FEATURES,
    CONTEXT_FEATURES,
    point_in_time_join,
)
from nselab.backtest import simulate
from nselab.common import config


def fixture_bars(n=260):
    dates = pd.bdate_range("2020-01-01", periods=n)
    records = []
    for symbol in ["TEST", "INDEX_market", "INDEX_technology"]:
        for i, d in enumerate(dates):
            px = 100 + np.sin(i / 7) * 2 + i * 0.01
            records.append(
                dict(
                    date=d,
                    symbol=symbol,
                    open=px,
                    high=px + 2,
                    low=px - 2,
                    close=px + 1,
                    volume=1e6,
                    raw_open=px,
                    raw_high=px + 2,
                    raw_low=px - 2,
                    raw_close=px + 1,
                    raw_volume=1e6,
                    split=1.0,
                    dividend=0.0,
                )
            )
    cfg = {**config(), "universe": {"TEST": "technology"}}
    return pd.DataFrame(records), cfg


def test_features_do_not_use_future_prices():
    bars, cfg = fixture_bars()
    date = sorted(bars.date.unique())[180]
    a = build_features(bars, cfg)
    changed = bars.copy()
    changed.loc[changed.date > date, ["close", "open", "volume"]] *= 10
    b = build_features(changed, cfg)
    pd.testing.assert_frame_equal(
        a.loc[a.date <= date, PRICE_FEATURES + CONTEXT_FEATURES],
        b.loc[b.date <= date, PRICE_FEATURES + CONTEXT_FEATURES],
    )


def test_labels_use_next_open_to_sixth_open():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    assert f.target.iloc[140] == pytest.approx(f.open.iloc[146] / f.open.iloc[141] - 1)
    assert f.entry_date.iloc[140] == f.date.iloc[141]
    assert f.label_end.iloc[140] == f.date.iloc[146]


def test_purge_removes_overlap():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    boundary = f.date.iloc[220]
    t = training_before(f, boundary, 6)
    assert t.label_end.max() < boundary
    assert t.date.max() < f.date.iloc[214]


def test_fee_components_match_schedule():
    f = IndiaDeliveryCosts()
    b = f.components(100000, "BUY")
    s = f.components(100000, "SELL")
    assert b["stt"] == 100 and b["stamp"] == pytest.approx(15) and b["brokerage"] == 20
    assert s["dp"] == 20 and s["stamp"] == 0
    assert b["gst"] == pytest.approx(0.18 * (20 + 3.07 + 0.1 + 0.1))
    assert f.components(100000, "SELL", 2)["total"] == 2 * s["total"]
    assert f.components(100, "BUY")["brokerage"] == 2.5


def test_next_open_cash_constraints_and_settlement():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    s = f[["date", "symbol"]].copy()
    s["score"] = 1.0
    s["threshold"] = 0.0
    start = f.date.iloc[150]
    end = f.date.iloc[177]
    r = simulate(f, s, start, end, cfg)
    assert r["trades"] and min(r["curve"].cash) >= 0
    first = r["trades"][0]
    assert first["date"] == str(start.date())
    assert first["signal_date"] == str(f.date.iloc[149].date())
    assert first["price"] == pytest.approx(f.raw_open.iloc[150] * 1.001)
    assert r["curve"].unsettled_cash.max() > 0
    assert r["curve"].equity.iloc[-1] == pytest.approx(
        cfg["initial_cash"] + sum(x["pnl"] for x in r["closed_trades"])
    )
    assert not r["open_positions"]


def test_locked_bar_does_not_fill():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    d = f.date.iloc[150]
    f.loc[f.date == d, "raw_high"] = f.loc[f.date == d, "raw_low"]
    s = f[["date", "symbol"]].copy()
    s["score"] = 1.0
    s["threshold"] = 0.0
    r = simulate(f, s, d, f.date.iloc[155], cfg)
    assert not r["trades"]
    assert r["rejects"]


def test_cost_stress_on_fixed_orders_reduces_net():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    s = f[["date", "symbol"]].copy()
    s["score"] = 1.0
    s["threshold"] = 0.0
    a = simulate(f, s, f.date.iloc[150], f.date.iloc[160], cfg)
    b = simulate(f, s, f.date.iloc[150], f.date.iloc[160], cfg, cost_multiplier=3)
    assert a["curve"].equity.iloc[-1] > b["curve"].equity.iloc[-1]


def test_news_availability_not_period_date():
    prices = pd.DataFrame(
        {"symbol": ["TEST"] * 2, "date": pd.to_datetime(["2024-01-02", "2024-01-03"])}
    )
    events = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "available_at": ["2024-01-02T12:00:00Z"],
            "sentiment": [0.8],
        }
    )
    joined = point_in_time_join(prices, events, ["sentiment"])
    assert pd.isna(joined.sentiment.iloc[0]) and joined.sentiment.iloc[1] == 0.8
    with pytest.raises(ValueError):
        point_in_time_join(prices, events.drop(columns="available_at"), ["sentiment"])


def test_split_is_wealth_neutral():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    for c in ["raw_open", "raw_close"]:
        f[c] = 100.0
    f.raw_high = 101.0
    f.raw_low = 99.0
    d = f.date.iloc[152]
    for c in ["raw_open", "raw_high", "raw_low", "raw_close"]:
        f.loc[f.date >= d, c] /= 2
    f.loc[f.date == d, "split"] = 2
    s = f[["date", "symbol"]].copy()
    s["score"] = 1.0
    s["threshold"] = 0.0
    r = simulate(f, s, f.date.iloc[150], f.date.iloc[155], cfg, cost_multiplier=0)
    assert r["curve"].equity.iloc[-1] == pytest.approx(cfg["initial_cash"])
    assert r["trades"][-1]["quantity"] == 2 * r["trades"][0]["quantity"]


def test_dividend_receivable_not_reinvested():
    bars, cfg = fixture_bars()
    f = build_features(bars, cfg)
    f.loc[f.date == f.date.iloc[152], "dividend"] = 2
    s = f[["date", "symbol"]].copy()
    s["score"] = 1.0
    s["threshold"] = 0.0
    r = simulate(f, s, f.date.iloc[150], f.date.iloc[155], cfg)
    assert r["curve"].dividend_receivable.iloc[-1] == 2 * r["trades"][0]["quantity"]
    assert r["curve"].equity.iloc[-1] == pytest.approx(
        cfg["initial_cash"] + sum(t["pnl"] for t in r["closed_trades"])
    )
