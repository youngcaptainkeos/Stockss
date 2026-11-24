"""Immutable source snapshots; no broker account/order access."""

import concurrent.futures
import datetime as dt
import json
import os
import time
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
from .common import ROOT, config, digest, dump


def fetch_yahoo(symbol, start, end):
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe="")}'
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "period2": int(pd.Timestamp(end, tz="UTC").timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    for attempt in range(4):
        try:
            response = requests.get(
                url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=35
            )
            response.raise_for_status()
            result = response.json()["chart"]["result"]
            if not result or not result[0].get("timestamp"):
                raise ValueError("No historical candles")
            return response.content, response.url
        except (requests.RequestException, ValueError, KeyError, TypeError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def parse_yahoo(payload, name):
    r = json.loads(payload)["chart"]["result"][0]
    dates = (
        pd.to_datetime(r["timestamp"], unit="s", utc=True)
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
        .normalize()
    )
    frame = pd.DataFrame(r["indicators"]["quote"][0], index=dates)
    frame.index.name = "date"
    frame = frame[["open", "high", "low", "close", "volume"]].copy()
    frame["split"] = 1.0
    frame["dividend"] = 0.0
    for event in r.get("events", {}).get("splits", {}).values():
        d = (
            pd.Timestamp(event["date"], unit="s", tz="UTC")
            .tz_convert("Asia/Kolkata")
            .tz_localize(None)
            .normalize()
        )
        if d in frame.index:
            frame.loc[d, "split"] = event["numerator"] / event["denominator"]
    # Yahoo OHLC and volume are split-normalized, not dividend-adjusted.
    # Reverse subsequent split factors to reconstruct contemporaneous rupee prices.
    future = frame["split"][::-1].cumprod()[::-1] / frame["split"]
    for col in ["open", "high", "low", "close"]:
        frame["raw_" + col] = frame[col] * future
    frame["raw_volume"] = frame["volume"] / future
    for event in r.get("events", {}).get("dividends", {}).values():
        d = (
            pd.Timestamp(event["date"], unit="s", tz="UTC")
            .tz_convert("Asia/Kolkata")
            .tz_localize(None)
            .normalize()
        )
        if d in frame.index:
            frame.loc[d, "dividend"] = event["amount"] * future.loc[d]
    frame["symbol"] = name
    frame["source"] = "Yahoo Finance chart endpoint"
    return frame.reset_index(), r.get("meta", {})


def validate(frame):
    bad = frame[["open", "high", "low", "close"]].isna().any(axis=1)
    bad |= (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    bad |= frame["high"] + 0.02 < frame[["open", "close", "low"]].max(axis=1)
    bad |= frame["low"] - 0.02 > frame[["open", "close", "high"]].min(axis=1)
    duplicate = frame.duplicated("date")
    if duplicate.any():
        raise ValueError("Duplicate sessions in source")
    cleaned = frame.loc[~bad].copy()
    audit = {
        "source_rows": len(frame),
        "invalid_rows_removed": int(bad.sum()),
        "first": str(cleaned.date.min().date()),
        "last": str(cleaned.date.max().date()),
        "zero_volume_sessions": int((cleaned.volume.fillna(0) == 0).sum()),
        "split_events": int((cleaned.split != 1).sum()),
    }
    return cleaned, audit


def ingest():
    cfg = config()
    rawdir = ROOT / "data/raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    instruments = {k: k + ".NS" for k in cfg["universe"]}
    instruments.update({"INDEX_" + k: v for k, v in cfg["indices"].items()})

    def one(item):
        name, ticker = item
        path = rawdir / (name + ".json")
        if path.exists():
            payload = path.read_bytes()
            url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(
                ticker, safe=""
            )
        else:
            payload, url = fetch_yahoo(
                ticker, cfg["data_start"], cfg["snapshot_end_exclusive"]
            )
            path.write_bytes(payload)
        frame, meta = parse_yahoo(payload, name)
        frame, audit = validate(frame)
        frame.to_csv(ROOT / "data/processed" / (name + ".csv"), index=False)
        return name, {
            **audit,
            "ticker": ticker,
            "url": url,
            "sha256": digest(path),
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName"),
            "retrieved_utc": dt.datetime.fromtimestamp(
                path.stat().st_mtime, dt.timezone.utc
            ).isoformat(),
        }

    (ROOT / "data/processed").mkdir(exist_ok=True, parents=True)
    manifest = {}
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, item): item[0] for item in instruments.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                k, v = future.result()
                manifest[k] = v
                print(
                    f'{k}: {v["source_rows"]} rows; {v["first"]} to {v["last"]}',
                    flush=True,
                )
            except Exception as exc:
                errors[name] = type(exc).__name__
                print(f"{name}: source unavailable ({type(exc).__name__})", flush=True)
    dump(
        ROOT / "data/manifest.json",
        {
            "sources": manifest,
            "errors": errors,
            "config_sha256": digest(ROOT / "experiment.json"),
            "groww_access_token_present": bool(os.getenv("GROWW_ACCESS_TOKEN")),
            "provider": "Yahoo Finance; daily OHLCV and corporate-action events",
            "adjustment": "Split-only features; reconstructed raw prices. No adjusted-close dividend reinvestment.",
            "reserved": "2026 prices excluded from all experiments; subsequent split metadata only reconstructs historical nominal units.",
        },
    )
    if errors:
        raise RuntimeError(
            "Some required source series unavailable. Inspect data/manifest.json; no synthetic fallback."
        )


def load_bars(end=None):
    cfg = config()
    frames = []
    for symbol in list(cfg["universe"]) + ["INDEX_" + k for k in cfg["indices"]]:
        f = pd.read_csv(ROOT / "data/processed" / f"{symbol}.csv", parse_dates=["date"])
        if end:
            f = f[f.date <= pd.Timestamp(end)]
        frames.append(f)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


def groww_candles(symbol, start, end, interval="1day"):
    """Explicit opt-in READ-ONLY fetch; caller must supply token in environment.
    Not mixed automatically into Yahoo snapshots: corporate-action normalization
    and data-quality validation are required before using another provider.
    """
    token = os.environ.get("GROWW_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GROWW_ACCESS_TOKEN is absent; no account access attempted")
    response = requests.get(
        "https://api.groww.in/v1/historical/candles",
        params={
            "exchange": "NSE",
            "segment": "CASH",
            "groww_symbol": "NSE-" + symbol,
            "start_time": start,
            "end_time": end,
            "candle_interval": interval,
        },
        headers={
            "Authorization": "Bearer " + token,
            "X-API-VERSION": "1.0",
            "Accept": "application/json",
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Groww data request failed (HTTP {response.status_code}); response redacted"
        )
    return response.json()
