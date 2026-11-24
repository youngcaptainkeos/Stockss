"""Close-time features and executable next-open labels. No backward fills."""

import numpy as np
import pandas as pd
from .common import config

PRICE_FEATURES = [
    "ret1",
    "ret5",
    "ret10",
    "ret20",
    "ret60",
    "vol10",
    "vol20",
    "range14",
    "gap",
    "body",
    "ema10_gap",
    "ema20_gap",
    "ema50_gap",
    "rsi14",
    "volume_ratio",
    "volume_change",
    "drawdown60",
]
CONTEXT_FEATURES = [
    "market_ret5",
    "market_ret20",
    "market_vol20",
    "market_trend",
    "sector_ret5",
    "sector_ret20",
    "relative20",
    "beta60",
]


def build_features(bars, cfg=None):
    cfg = cfg or config()
    market = bars[bars.symbol == "INDEX_market"].set_index("date").sort_index()
    calendar = market.index
    frames = []
    for symbol, sector in cfg["universe"].items():
        raw = bars[bars.symbol == symbol].set_index("date").sort_index()
        f = raw.reindex(calendar).copy()
        close = f.close
        r = close.pct_change(fill_method=None)
        for n in [1, 5, 10, 20, 60]:
            f[f"ret{n}"] = close.pct_change(n, fill_method=None)
        for n in [10, 20]:
            f[f"vol{n}"] = r.rolling(n).std() * np.sqrt(252)
        f["range14"] = ((f.high - f.low) / close).rolling(14).mean()
        f["gap"] = f.open / close.shift(1) - 1
        f["body"] = close / f.open - 1
        for n in [10, 20, 50]:
            f[f"ema{n}_gap"] = (
                close / close.ewm(span=n, adjust=False, min_periods=n).mean() - 1
            )
        gain = r.clip(lower=0).rolling(14).mean()
        loss = -r.clip(upper=0).rolling(14).mean()
        f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
        f["volume_ratio"] = f.volume / f.volume.rolling(20).mean()
        f["volume_change"] = f.volume.pct_change(fill_method=None).clip(-1, 5)
        f["drawdown60"] = close / close.rolling(60).max() - 1
        # Rupee turnover reconstructed from nominal OHLC and nominal shares.
        f["adv20"] = (f.raw_close * f.raw_volume).rolling(20).mean()
        f["history_sessions"] = f.close.notna().cumsum()
        mr = market.close.pct_change(fill_method=None)
        f["market_ret5"] = market.close.pct_change(5, fill_method=None)
        f["market_ret20"] = market.close.pct_change(20, fill_method=None)
        f["market_vol20"] = mr.rolling(20).std() * np.sqrt(252)
        f["market_trend"] = (
            market.close
            / market.close.ewm(span=100, adjust=False, min_periods=100).mean()
            - 1
        )
        sector_frame = (
            bars[bars.symbol == "INDEX_" + sector].set_index("date").reindex(calendar)
        )
        f["sector_ret5"] = sector_frame.close.pct_change(5, fill_method=None)
        f["sector_ret20"] = sector_frame.close.pct_change(20, fill_method=None)
        f["relative20"] = f.ret20 - f.sector_ret20
        f["beta60"] = r.rolling(60).cov(mr) / mr.rolling(60).var()
        # Entry after signal close; target excludes dividends and is explicitly price return.
        f["target"] = f.open.shift(-(cfg["horizon"] + 1)) / f.open.shift(-1) - 1
        f["label_end"] = pd.Series(calendar, index=calendar).shift(
            -(cfg["horizon"] + 1)
        )
        f["entry_date"] = pd.Series(calendar, index=calendar).shift(-1)
        f["symbol"] = symbol
        f["sector"] = sector
        f["eligible"] = (
            (f.history_sessions >= 100)
            & (f.adv20 >= cfg["min_adv_inr"])
            & (f.raw_volume > 0)
        )
        # E2E traded on SME lots before its documented mainboard migration.
        if symbol == "E2E":
            f.loc[f.index < pd.Timestamp("2022-04-12"), "eligible"] = False
        # Do not fabricate labels across a suspension/missing session.
        valid = (
            f.open.notna()
            .rolling(cfg["horizon"] + 1)
            .sum()
            .shift(-(cfg["horizon"] + 1))
            == cfg["horizon"] + 1
        )
        f.loc[~valid, "target"] = np.nan
        frames.append(f.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out[PRICE_FEATURES + CONTEXT_FEATURES] = out[
        PRICE_FEATURES + CONTEXT_FEATURES
    ].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def training_before(frame, start, purge_sessions=6):
    """Purge overlapping targets and additionally embargo six full market sessions.
    Because all training is strictly earlier, no post-validation training is used.
    """
    dates = np.sort(frame.date.unique())
    pos = np.searchsorted(dates, np.datetime64(pd.Timestamp(start)))
    if pos <= purge_sessions:
        return frame.iloc[:0].copy()
    cutoff = pd.Timestamp(dates[pos - purge_sessions])
    return frame[
        (frame.date < cutoff)
        & (frame.label_end < pd.Timestamp(start))
        & frame.target.notna()
        & frame.eligible
    ].copy()


def point_in_time_join(prices, events, value_columns):
    """Optional future extension: use public availability, never report-period end.
    Requires symbol, available_at UTC and timestamped values. Restatement-vintage
    archives must be verified externally. This function alone cannot certify them.
    """
    if "available_at" not in events:
        raise ValueError("Point-in-time available_at is required")
    result = []
    for symbol, g in prices.groupby("symbol"):
        e = events[events.symbol == symbol].copy()
        e["available_at"] = pd.to_datetime(e.available_at, utc=True)
        g = g.copy()
        g["decision_at"] = pd.to_datetime(g.date).dt.tz_localize(
            "Asia/Kolkata"
        ) + pd.Timedelta(hours=15, minutes=30)
        g["decision_at"] = g.decision_at.dt.tz_convert("UTC")
        result.append(
            pd.merge_asof(
                g.sort_values("decision_at"),
                e[["available_at"] + value_columns].sort_values("available_at"),
                left_on="decision_at",
                right_on="available_at",
                direction="backward",
                allow_exact_matches=False,
            )
        )
    return pd.concat(result, ignore_index=True)
