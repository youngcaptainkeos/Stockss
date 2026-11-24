"""Independent reconciliation of cached outputs; never refits or reopens the test."""

import json
import numpy as np
import pandas as pd
from .common import ROOT, config, digest, dump


def verify():
    cfg = config()
    lock = json.loads((ROOT / "artifacts/selection_lock.json").read_text())
    results = json.loads((ROOT / "artifacts/final_results.json").read_text())
    checks = []
    for filename, expected in lock["processed_hashes"].items():
        assert digest(ROOT / "data/processed" / filename) == expected
    checks.append("All processed input hashes match the pre-test selection lock.")
    for name, metrics in results["strategies"].items():
        folder = ROOT / "artifacts/final" / name
        curve = pd.read_csv(folder / "equity.csv")
        assert (curve.cash >= -1e-6).all(), name
        assert np.isclose(
            curve.equity.iloc[-1] - cfg["initial_cash"], metrics["net_pnl"]
        ), name
        assert (
            curve.date.iloc[0] >= cfg["test_start"]
            and curve.date.iloc[-1] <= cfg["test_end"]
        )
        if metrics["round_trips"]:
            trades = pd.read_csv(folder / "trades.csv")
            trips = pd.read_csv(folder / "round_trips.csv")
            assert np.isclose(trades.fees.sum(), metrics["costs"])
            if not metrics["unliquidated"]:
                assert np.isclose(trips.pnl.sum(), metrics["net_pnl"], atol=0.01), name
            assert (
                trades.loc[trades.side == "BUY", "signal_date"]
                < trades.loc[trades.side == "BUY", "date"]
            ).all()
        if name in lock["audits"]:
            audit = lock["audits"][name]
            assert audit["last_core_label"] < audit["cal_start"]
            assert audit["last_cal_label"] < cfg["test_start"]
    checks += [
        "All strategies have nonnegative cash and matching evaluation periods.",
        "Trade-fee totals reconcile with portfolio fees.",
        "Closed-trade P&L reconciles to final equity for fully liquidated portfolios.",
        "Every buy signal precedes its fill; model/calibration label dates precede test.",
    ]
    trial_files = list((ROOT / "artifacts/search").glob("*.json"))
    trials = sum(len(json.loads(p.read_text())["trials"]) for p in trial_files)
    assert trials == 128
    checks.append("Exactly 128 bounded candidate trials recorded across 16 searches.")
    payload = json.loads((ROOT / "ui/replay.json").read_text())
    assert set(["NETWEB", "E2E", "TATACOMM"]).issubset(payload["candles"])
    for rows in payload["candles"].values():
        assert all(cfg["test_start"] <= row["date"] <= cfg["test_end"] for row in rows)
    checks.append("Replay contains only the declared 2025 final period.")
    dump(
        ROOT / "artifacts/verification.json",
        {"passed": True, "checks": checks, "trials": trials},
    )
    print("\n".join("PASS: " + check for check in checks))
