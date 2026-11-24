"""Nested chronological model selection. The final test requires a frozen lock."""

import datetime as dt
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from .common import ROOT, config, digest, dump
from .data import load_bars
from .features import build_features, training_before
from .models import ModelBundle
from .backtest import simulate, baseline_signals
from .metrics import (
    performance,
    classification,
    robust_score,
    regime_metrics,
    block_bootstrap,
)

BASELINES = ["cash", "buy_hold", "momentum", "ema", "mean_reversion"]
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def frame_until(end):
    return build_features(load_bars(end), config())


def eval_slice(frame, start, end):
    prior = frame.loc[frame.date < pd.Timestamp(start), "date"].max()
    return frame[(frame.date >= prior) & (frame.date <= pd.Timestamp(end))].copy()


def evaluate(frame, predictions, start, end, cfg, cost=1.0, buy_hold=False):
    args = {**cfg}
    if buy_hold:
        args.update(
            max_positions=len(cfg["universe"]),
            max_weight=cfg["max_gross"] / len(cfg["universe"]),
        )
    return simulate(
        eval_slice(frame, start, end),
        predictions,
        start,
        end,
        args,
        cost_multiplier=cost,
        buy_hold=buy_hold,
    )


def suggest(trial, family):
    p = {
        "k": trial.suggest_categorical("k", [8, 12, 17, 25]),
        "threshold": trial.suggest_categorical(
            "threshold", [0.003, 0.006, 0.010, 0.015]
        ),
    }
    if family == "ridge":
        p["alpha"] = trial.suggest_float("alpha", 0.1, 1000, log=True)
    else:
        p.update(
            num_leaves=trial.suggest_categorical("num_leaves", [7, 12, 15]),
            max_depth=trial.suggest_categorical("max_depth", [3, 4, 5]),
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            min_child_samples=trial.suggest_categorical(
                "min_child_samples", [40, 80, 120]
            ),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.85),
            subsample=trial.suggest_float("subsample", 0.7, 0.9),
        )
    return p


def tune(training, family, feature_set, tag, cfg):
    ds = sorted(training.date.unique())
    width = cfg["inner_validation_sessions"]
    windows = []
    for j in [2, 1]:
        a = pd.Timestamp(ds[-j * width])
        b = pd.Timestamp(ds[-(j - 1) * width - 1])
        windows.append((str(a.date()), str(b.date())))
    record = []

    def objective(trial):
        params = suggest(trial, family)
        scores = {a: [] for a in cfg["architectures"]}
        details = []
        for start, end in windows:
            fit = training_before(training, start, cfg["purge_sessions"])
            model = ModelBundle(family, params, feature_set, cfg).fit(fit)
            view = eval_slice(training, start, end)
            for arch in cfg["architectures"]:
                pred = model.predict(view, arch)
                base = performance(evaluate(training, pred, start, end, cfg), cfg)
                stressed = performance(
                    evaluate(training, pred, start, end, cfg, cost=2.0), cfg
                )
                value = min(robust_score(base), robust_score(stressed))
                scores[arch].append(value)
                details.append(
                    {
                        "architecture": arch,
                        "start": start,
                        "end": end,
                        "net_return": base["net_return"],
                        "sharpe": base["sharpe"],
                        "stress_net_return": stressed["net_return"],
                        "objective": value,
                    }
                )
        aggregate = {
            a: float(np.median(v) - 0.5 * np.std(v)) for a, v in scores.items()
        }
        record.append(
            {
                "trial": trial.number,
                "params": params,
                "architecture_objectives": aggregate,
                "windows": details,
            }
        )
        return float(np.mean(list(aggregate.values())))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=4),
    )
    study.optimize(
        objective,
        n_trials=cfg["trials_per_family_per_feature_set_per_outer_fit"],
        gc_after_trial=True,
    )
    selected = {
        a: max(record, key=lambda r: r["architecture_objectives"][a])["params"]
        for a in cfg["architectures"]
    }
    dump(
        ROOT / "artifacts/search" / f"{tag}_{family}_{feature_set}.json",
        {
            "tag": tag,
            "family": family,
            "features": feature_set,
            "windows": windows,
            "budget": len(record),
            "selected": selected,
            "trials": record,
            "objective": "median inner-fold min(base,2x cost) Sharpe minus 2*absolute drawdown, minus 0.5 fold-score std; >=5 trades",
            "search_design": "Shared candidate search across architectures; each architecture chooses its best candidate on inner folds only. Per-stock hyperparameters shared, weights fitted independently.",
        },
    )
    return selected


def fit_predict_selected(
    training, view, family, feature_set, selected, cfg, save_prefix=None
):
    cache = {}
    out = {}
    audits = {}
    for arch, params in selected.items():
        key = json.dumps(params, sort_keys=True)
        if key not in cache:
            cache[key] = ModelBundle(family, params, feature_set, cfg).fit(training)
        model = cache[key]
        out[arch] = model.predict(view, arch)
        audits[arch] = model.audit
        if save_prefix:
            path = ROOT / "artifacts/models" / f"{save_prefix}_{arch}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
    return out, audits


def code_hashes():
    return {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "src/nselab").glob("*.py"))
        if p.name not in ["report.py", "server.py", "cli.py"]
    }


def develop():
    cfg = config()
    if (ROOT / "artifacts/test_access.json").exists():
        raise RuntimeError(
            "Final test has already been opened. Create a new preregistered experiment and a new untouched future period; do not retune this run."
        )
    if (ROOT / "artifacts/selection_lock.json").exists():
        raise RuntimeError("Selection already frozen. Reuse it; do not overwrite it.")
    frame = frame_until(cfg["development_end"])
    results = {}
    fit_audits = {}
    for fold, (start, end) in enumerate(cfg["outer_folds"]):
        print(f"Outer fold {fold+1}: {start} to {end}", flush=True)
        training = training_before(frame, start, cfg["purge_sessions"])
        view = eval_slice(frame, start, end)
        for family in cfg["families"]:
            for fs in cfg["feature_sets"]:
                print(
                    f"  Tuning {family}/{fs}: 8 candidates, 2 inner folds", flush=True
                )
                selected = tune(training, family, fs, f"outer{fold+1}", cfg)
                predictions, audits = fit_predict_selected(
                    training, view, family, fs, selected, cfg
                )
                for arch, pred in predictions.items():
                    name = f"{family}_{arch}_{fs}"
                    r = evaluate(frame, pred, start, end, cfg)
                    m = performance(r, cfg)
                    stress = performance(
                        evaluate(frame, pred, start, end, cfg, cost=2.0), cfg
                    )
                    m["stress_return"] = stress["net_return"]
                    m["selection_score"] = min(robust_score(m), robust_score(stress))
                    m["classification"] = classification(
                        pred[
                            (pred.date >= start)
                            & (pred.label_end <= pd.Timestamp(end))
                            & pred.eligible
                        ]
                    )
                    results.setdefault(name, []).append(m)
                    fit_audits[f"{name}_fold{fold+1}"] = audits[arch]
                    pred.to_csv(
                        ROOT / "artifacts" / f"dev_{name}_fold{fold+1}.csv", index=False
                    )
        for name in BASELINES:
            pred = baseline_signals(view, name)
            m = performance(
                evaluate(frame, pred, start, end, cfg, buy_hold=name == "buy_hold"), cfg
            )
            stress = performance(
                evaluate(
                    frame, pred, start, end, cfg, cost=2.0, buy_hold=name == "buy_hold"
                ),
                cfg,
            )
            m["selection_score"] = min(robust_score(m), robust_score(stress))
            results.setdefault(name, []).append(m)
        dump(
            ROOT / "artifacts/development.json",
            {"folds": cfg["outer_folds"], "strategies": results, "audits": fit_audits},
        )
    ranking = []
    for name, folds in results.items():
        values = np.array([x["selection_score"] for x in folds])
        ranking.append(
            {
                "strategy": name,
                "score": float(np.median(values) - 0.5 * values.std()),
                "fold_returns": [m["net_return"] for m in folds],
                "fold_sharpes": [m["sharpe"] for m in folds],
            }
        )
    ranking.sort(key=lambda x: -x["score"])
    ml = [r for r in ranking if r["strategy"] not in BASELINES]
    champion = ml[0]["strategy"]
    print(f"Development-selected ML candidate: {champion}", flush=True)
    # Freeze hyperparameters and trained models without loading the final period.
    training = training_before(
        frame, pd.Timestamp(cfg["test_start"]), cfg["purge_sessions"]
    )
    locked = {}
    final_audits = {}
    for family in cfg["families"]:
        for fs in cfg["feature_sets"]:
            print(f"Final development-only tuning: {family}/{fs}", flush=True)
            selected = tune(training, family, fs, "final_fit", cfg)
            _, audits = fit_predict_selected(
                training,
                frame.tail(21),
                family,
                fs,
                selected,
                cfg,
                save_prefix=f"{family}_{fs}",
            )
            for arch, params in selected.items():
                locked[f"{family}_{arch}_{fs}"] = {
                    "family": family,
                    "features": fs,
                    "architecture": arch,
                    "params": params,
                    "model": f"artifacts/models/{family}_{fs}_{arch}.joblib",
                }
                final_audits[f"{family}_{arch}_{fs}"] = audits[arch]
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    lock = {
        "experiment_id": cfg["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": digest(ROOT / "experiment.json"),
        "code_hashes": code_hashes(),
        "data_hashes": {k: v["sha256"] for k, v in manifest["sources"].items()},
        "processed_hashes": {
            p.name: digest(p) for p in sorted((ROOT / "data/processed").glob("*.csv"))
        },
        "champion": champion,
        "ranking": ranking,
        "candidates": locked,
        "audits": final_audits,
        "test_period": [cfg["test_start"], cfg["test_end"]],
        "decision_rule": "Research-only unless selected ML has positive test net return, beats both buy/hold and NIFTY, Sharpe>0.5, survives 2x costs, >=30 trades, and bootstrap lower bound>0. Data limitations independently block deployment.",
    }
    lock["model_hashes"] = {
        v["model"]: digest(ROOT / v["model"]) for v in locked.values()
    }
    dump(ROOT / "artifacts/selection_lock.json", lock)
    print(
        "Frozen selection_lock.json. Final test has NOT been read or evaluated.",
        flush=True,
    )


def final_test():
    cfg = config()
    lock_path = ROOT / "artifacts/selection_lock.json"
    access = ROOT / "artifacts/test_access.json"
    lock = json.loads(lock_path.read_text())
    if access.exists():
        raise RuntimeError(
            "Test already opened. Use report/serve to inspect existing results; no automatic rerun."
        )
    if (
        digest(ROOT / "experiment.json") != lock["config_sha256"]
        or code_hashes() != lock["code_hashes"]
    ):
        raise RuntimeError(
            "Config or research code changed after lock; test access refused."
        )
    for p, h in lock["processed_hashes"].items():
        if digest(ROOT / "data/processed" / p) != h:
            raise RuntimeError("Data changed after lock")
    for p, h in lock["model_hashes"].items():
        if digest(ROOT / p) != h:
            raise RuntimeError("Model changed after lock")
    dump(
        access,
        {
            "opened_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection_lock_sha256": digest(lock_path),
            "status": "opened",
            "policy": "One predetermined evaluation; any error recovery must be recorded, never silently tune on results.",
        },
    )
    frame = frame_until(cfg["test_end"])
    start, end = cfg["test_start"], cfg["test_end"]
    view = eval_slice(frame, start, end)
    results = {}
    replay = {}
    for name, spec in lock["candidates"].items():
        model = joblib.load(ROOT / spec["model"])
        pred = model.predict(view, spec["architecture"])
        results[name], replay[name] = evaluate_for_report(
            frame, pred, start, end, cfg, name
        )
        print(
            f'{name}: return {results[name]["net_return"]:.2%}; Sharpe {results[name]["sharpe"]:.2f}',
            flush=True,
        )
    for name in BASELINES:
        pred = baseline_signals(view, name)
        results[name], replay[name] = evaluate_for_report(
            frame, pred, start, end, cfg, name, buy_hold=name == "buy_hold"
        )
    market = load_bars(end)
    market = market[(market.symbol == "INDEX_market") & (market.date >= start)].copy()
    # Informational price-index benchmark, not a directly investable ETF; no dividends/fees.
    market["equity"] = cfg["initial_cash"] * market.raw_close / market.raw_open.iloc[0]
    market_return = float(market.equity.iloc[-1] / cfg["initial_cash"] - 1)
    selected = results[lock["champion"]]
    selected_curve = pd.DataFrame(replay[lock["champion"]]["curve"])
    bh_curve = pd.DataFrame(replay["buy_hold"]["curve"])
    rr = (
        np.diff(np.r_[cfg["initial_cash"], selected_curve.equity])
        / np.r_[cfg["initial_cash"], selected_curve.equity.to_numpy()[:-1]]
    )
    br = (
        np.diff(np.r_[cfg["initial_cash"], bh_curve.equity])
        / np.r_[cfg["initial_cash"], bh_curve.equity.to_numpy()[:-1]]
    )
    selected["paired_uncertainty_vs_buy_hold"] = block_bootstrap(rr, cfg, br)
    criteria = {
        "positive_net_return": selected["net_return"] > 0,
        "beats_buy_hold": selected["net_return"] > results["buy_hold"]["net_return"],
        "beats_nifty_price": selected["net_return"] > market_return,
        "sharpe_above_0_5": selected["sharpe"] > 0.5,
        "positive_at_double_cost": selected["cost_stress"]["2"]["net_return"] > 0,
        "at_least_30_trades": selected["round_trips"] >= 30,
        "positive_95_percent_return_lower_bound": selected["uncertainty"][
            "net_return_95_interval"
        ][0]
        > 0,
        "all_positions_liquidated": not selected["unliquidated"],
    }
    summary = {
        "experiment_id": cfg["experiment_id"],
        "selected_model": lock["champion"],
        "selected_before_test": True,
        "test_start": start,
        "test_end": end,
        "strategies": results,
        "nifty_price_return": market_return,
        "criteria": criteria,
        "statistical_gate_passed": all(criteria.values()),
        "status": "RESEARCH ONLY",
        "reason": "No deployment approval: survivor-selected universe, vendor-adjusted history, no official corporate-action reconciliation or order-book/circuit validation.",
        "unavailable_ablations": {
            "news_sentiment": "No verified point-in-time Indian news archive; not fitted or backfilled with current news.",
            "fundamentals": "No verified publication-time and vintage-preserved historical fundamentals.",
            "ORB_VWAP": "No validated historical intraday OHLCV archive; daily candles cannot implement these.",
        },
    }
    dump(ROOT / "artifacts/final_results.json", summary)
    # Payload only contains the final historical window. UI reveals bars up to replay cursor.
    candles = {
        s: [
            {
                "date": str(r.date.date()),
                "open": r.raw_open,
                "high": r.raw_high,
                "low": r.raw_low,
                "close": r.raw_close,
                "volume": r.raw_volume,
            }
            for r in g.itertuples()
            if np.isfinite(r.raw_close)
        ]
        for s, g in view[view.date >= start].groupby("symbol")
    }
    payload = {
        "summary": summary,
        "development_ranking": lock["ranking"],
        "strategies": replay,
        "candles": candles,
        "market_curve": [
            {"date": str(r.date.date()), "equity": r.equity}
            for r in market.itertuples()
        ],
        "config": cfg,
        "provenance": json.loads((ROOT / "data/manifest.json").read_text()),
    }
    dump(ROOT / "ui/replay.json", payload)
    dump(
        access,
        {
            "opened_utc": json.loads(access.read_text())["opened_utc"],
            "selection_lock_sha256": digest(lock_path),
            "status": "completed",
            "result_sha256": digest(ROOT / "artifacts/final_results.json"),
        },
    )
    print(
        "Final evaluation completed and cached. No model re-selection on test.",
        flush=True,
    )


def evaluate_for_report(frame, pred, start, end, cfg, name, buy_hold=False):
    result = evaluate(frame, pred, start, end, cfg, buy_hold=buy_hold)
    metrics = performance(result, cfg, bootstrap=True)
    metrics["cost_stress"] = {
        str(mult): performance(
            evaluate(frame, pred, start, end, cfg, cost=mult, buy_hold=buy_hold), cfg
        )
        for mult in cfg["cost_stress_multipliers"]
    }
    metrics["regimes"] = regime_metrics(result, frame)
    if "probability" in pred:
        valid = pred[
            (pred.date >= start) & (pred.label_end <= pd.Timestamp(end)) & pred.eligible
        ]
        metrics["classification"] = classification(valid)
        metrics["per_stock_classification"] = {
            s: classification(g) for s, g in valid.groupby("symbol")
        }
    metrics["per_stock_pnl"] = {
        s: float(sum(t["pnl"] for t in result["closed_trades"] if t["symbol"] == s))
        for s in cfg["universe"]
    }
    metrics["rejected_orders"] = len(result["rejects"])
    path = ROOT / "artifacts/final" / name
    path.mkdir(parents=True, exist_ok=True)
    result["curve"].to_csv(path / "equity.csv", index=False)
    pd.DataFrame(result["trades"]).to_csv(path / "trades.csv", index=False)
    pd.DataFrame(result["closed_trades"]).to_csv(path / "round_trips.csv", index=False)
    pred.to_csv(path / "predictions.csv", index=False)
    dump(path / "metrics.json", metrics)
    dump(
        path / "execution_audit.json",
        {"rejects": result["rejects"], "open_positions": result["open_positions"]},
    )
    # Avoid emitting NaN into JSON; missing local models remain explicit null.
    records = json.loads(
        pred.assign(date=pred.date.dt.strftime("%Y-%m-%d"))
        .drop(columns=["label_end", "entry_date", "target"], errors="ignore")
        .to_json(orient="records")
    )
    return metrics, {
        "curve": result["curve"].to_dict("records"),
        "trades": result["trades"],
        "signals": records,
    }
