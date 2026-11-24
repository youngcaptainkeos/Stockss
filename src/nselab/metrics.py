import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)
from .common import config


def performance(result, cfg=None, bootstrap=False):
    cfg = cfg or config()
    c = result["curve"]
    e = c.equity.to_numpy(float)
    r = np.diff(np.r_[cfg["initial_cash"], e]) / np.r_[cfg["initial_cash"], e[:-1]]
    sd = np.std(r, ddof=1)
    rf = (1 + cfg["risk_free_annual"]) ** (1 / 252) - 1
    sharpe = float(np.sqrt(252) * (np.mean(r) - rf) / sd) if sd > 1e-12 else 0.0
    pnls = np.array([x["pnl"] for x in result["closed_trades"]])
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    out = {
        "net_return": float(e[-1] / cfg["initial_cash"] - 1),
        "net_pnl": float(e[-1] - cfg["initial_cash"]),
        "cagr": float((e[-1] / cfg["initial_cash"]) ** (252 / len(e)) - 1),
        "sharpe": sharpe,
        "sharpe_zero_rf": float(np.sqrt(252) * np.mean(r) / sd) if sd > 1e-12 else 0.0,
        "annual_volatility": float(sd * np.sqrt(252)),
        "max_drawdown": float(
            min(
                0,
                np.min(
                    e / np.maximum.accumulate(np.r_[cfg["initial_cash"], e])[1:] - 1
                ),
            )
        ),
        "costs": float(result["costs"]),
        "slippage": float(result["slippage"]),
        "gross_pnl_before_explicit_costs": float(
            e[-1] - cfg["initial_cash"] + result["costs"] + result["slippage"]
        ),
        "round_trips": len(pnls),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else None,
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "average_trade_pnl": float(pnls.mean()) if len(pnls) else None,
        "turnover_multiple": float(result["turnover"] / cfg["initial_cash"]),
        "mean_exposure": float(c.exposure.mean()),
        "halted": bool(c.halted.any()),
        "unliquidated": result["unliquidated"],
        "sessions": len(c),
        "start": c.date.iloc[0],
        "end": c.date.iloc[-1],
    }
    if bootstrap:
        out["uncertainty"] = block_bootstrap(r, cfg)
    return out


def block_bootstrap(returns, cfg=None, benchmark=None):
    cfg = cfg or config()
    r = np.asarray(returns)
    n = len(r)
    block = cfg["bootstrap_block_sessions"]
    rng = np.random.default_rng(cfg["seed"])
    net = []
    sharpes = []
    excess = []
    rf = (1 + cfg["risk_free_annual"]) ** (1 / 252) - 1
    for _ in range(cfg["bootstrap_samples"]):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(a + np.arange(block)) % n for a in starts])[:n]
        x = r[idx]
        net.append(np.prod(1 + x) - 1)
        sd = x.std(ddof=1)
        sharpes.append(np.sqrt(252) * (x.mean() - rf) / sd if sd > 1e-12 else 0.0)
        if benchmark is not None:
            excess.append(np.prod(1 + x) - np.prod(1 + np.asarray(benchmark)[idx]))
    out = {
        "method": "circular moving-block bootstrap; conditional on selected strategy, not selection-adjusted",
        "samples": cfg["bootstrap_samples"],
        "block_sessions": block,
        "net_return_95_interval": np.quantile(net, [0.025, 0.975]).tolist(),
        "sharpe_95_interval": np.quantile(sharpes, [0.025, 0.975]).tolist(),
    }
    if excess:
        out["excess_return_95_interval"] = np.quantile(excess, [0.025, 0.975]).tolist()
    return out


def classification(predictions):
    q = predictions.dropna(subset=["target", "probability", "score"])
    if len(q) < 2:
        return {"n": len(q)}
    y = (q.target > 0).astype(int)
    p = q.probability.clip(1e-6, 1 - 1e-6)
    hard = (p >= 0.5).astype(int)
    bins = []
    for low in np.linspace(0, 0.9, 10):
        mask = (p >= low) & (p < (low + 0.1 + 1e-10))
        if mask.any():
            bins.append(
                {
                    "lower": float(low),
                    "n": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "observed_positive": float(y[mask].mean()),
                }
            )
    return {
        "n": len(q),
        "positive_prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "precision": float(precision_score(y, hard, zero_division=0)),
        "recall": float(recall_score(y, hard, zero_division=0)),
        "f1": float(f1_score(y, hard, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "return_mae": float(mean_absolute_error(q.target, q.score)),
        "rank_ic": (
            float(q.target.corr(q.score, method="spearman"))
            if q.score.nunique() > 1
            else None
        ),
        "expected_calibration_error": float(
            sum(
                x["n"] * abs(x["mean_probability"] - x["observed_positive"])
                for x in bins
            )
            / len(q)
        ),
        "reliability_bins": bins,
        "note": "Direction of five-session next-open price return; overlapping observations, no IID significance claim.",
    }


def robust_score(metrics):
    if metrics["round_trips"] < 5:
        return -5.0
    return float(np.clip(metrics["sharpe"], -5, 5) - 2 * abs(metrics["max_drawdown"]))


def regime_metrics(result, features):
    c = result["curve"].copy()
    c["date"] = pd.to_datetime(c.date)
    r = c.equity.pct_change().fillna(c.equity.iloc[0] / config()["initial_cash"] - 1)
    context = features.drop_duplicates("date").set_index("date")
    trend = c.date.map(context.market_trend.shift(1))
    vol = c.date.map(context.market_vol20.shift(1))
    # Predeclared fixed thresholds; not chosen from test distribution.
    labels = {
        "trend_positive": trend > 0,
        "trend_nonpositive": trend <= 0,
        "high_volatility": vol > 0.20,
        "lower_volatility": vol <= 0.20,
    }
    out = {}
    for name, m in labels.items():
        x = r[m.to_numpy()]
        out[name] = {
            "sessions": len(x),
            "compounded_selected_day_return": float((1 + x).prod() - 1),
            "mean_daily_return": float(x.mean()) if len(x) else None,
        }
    return out
