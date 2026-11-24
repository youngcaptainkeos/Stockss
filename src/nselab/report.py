"""Render readable reports from cached results; this module never trains models."""

import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, config


def generate_report():
    cfg = config()
    results = json.loads((ROOT / "artifacts/final_results.json").read_text())
    lock = json.loads((ROOT / "artifacts/selection_lock.json").read_text())
    chosen_name = results["selected_model"]
    chosen = results["strategies"][chosen_name]
    bh = results["strategies"]["buy_hold"]
    fmt = lambda v: f"{v:.2%}"
    rupees = lambda v: f"₹{v:,.0f}"
    rows = []
    for name, m in results["strategies"].items():
        rows.append(
            {
                "strategy": name,
                "net_return": m["net_return"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "costs_inr": m["costs"],
                "slippage_inr": m["slippage"],
                "round_trips": m["round_trips"],
                "mean_exposure": m["mean_exposure"],
                "brier": m.get("classification", {}).get("brier"),
            }
        )
    table = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    table.to_csv(ROOT / "reports/comparison.csv", index=False)
    interval = chosen["uncertainty"]["net_return_95_interval"]
    paired = chosen["paired_uncertainty_vs_buy_hold"]["excess_return_95_interval"]
    trials = sum(
        len(json.loads(p.read_text())["trials"])
        for p in (ROOT / "artifacts/search").glob("*.json")
    )
    c = chosen["classification"]
    lines = [
        "# NSE Research Lab — final historical evaluation",
        "",
        f"**Status: RESEARCH ONLY.** Development selected **{chosen_name}** before the final test. "
        f'Its 2025 net return was **{fmt(chosen["net_return"])}**, or **{rupees(chosen["net_pnl"])}** on ₹10 lakh. '
        f'The statistical gate {"passed" if results["statistical_gate_passed"] else "failed"}. '
        "No live deployment is justified by this experiment.",
        "",
        "## Evidence at a glance",
        "",
        f'- Final period: {chosen["start"]} to {chosen["end"]}, {chosen["sessions"]} market sessions; evaluated once after a frozen selection lock.',
        f'- Broad capped buy/hold: **{fmt(bh["net_return"])}**; NIFTY 50 price index: **{fmt(results["nifty_price_return"])}**. '
        "The NIFTY comparator has full exposure, no dividends and no trading costs.",
        f'- Selected strategy Sharpe: **{chosen["sharpe"]:.2f}** (6% annual hurdle); max drawdown: **{fmt(chosen["max_drawdown"])}**; '
        f'mean exposure: **{fmt(chosen["mean_exposure"])}**.',
        f'- {chosen["round_trips"]} closed trades; win rate {fmt(chosen["win_rate"] or 0)}; '
        f'explicit fees **{rupees(chosen["costs"])}**, adverse slippage **{rupees(chosen["slippage"])}**.',
        f"- Base / 2× / 3× total-cost scenario returns: "
        + " / ".join(
            fmt(chosen["cost_stress"][str(x)]["net_return"]) for x in [1, 2, 3]
        )
        + ".",
        f"- 95% moving-block bootstrap interval for net return: **{fmt(interval[0])} to {fmt(interval[1])}**. "
        f"Paired excess return versus buy/hold: **{fmt(paired[0])} to {fmt(paired[1])}**. "
        "These conditional intervals do not correct for model-search selection.",
        "",
        "## What was actually trained",
        "",
        f"{trials} candidate trials across 16 bounded searches, two inner folds each, three outer validation windows and a final development-only fit. "
        "Twelve ML candidates compare ridge versus LightGBM, pooled versus independent per-stock versus history-weighted hybrid, "
        "and price/volume versus additional Indian market/sector features. The complete configuration and objective were specified before test access in PROTOCOL.md.",
        "",
        "The final fit has 20 local stock models. E2E had only 118 eligible core observations against the fixed 126-observation minimum; its per-stock model abstained and the hybrid used the pooled model. NETWEB had 176 eligible core observations. "
        "Per-stock modelling is implemented: each eligible symbol has separately fitted coefficients/trees and selected features. "
        "Hyperparameters are shared within each architecture/family/feature group; they are not independently optimized for every stock. "
        "Sparse local histories abstain; the hybrid falls back to pooled predictions. There is no global-optimum or future-profit guarantee.",
        "",
        "## Final comparisons",
        "",
        "| Strategy | Net return | Sharpe | Max drawdown | Fees | Slippage | Closed trades | Mean exposure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results["strategies"].items():
        lines.append(
            f'| {name}{" **selected**" if name==chosen_name else ""} | {fmt(m["net_return"])} | {m["sharpe"]:.2f} | {fmt(m["max_drawdown"])} | {rupees(m["costs"])} | {rupees(m["slippage"])} | {m["round_trips"]} | {fmt(m["mean_exposure"])} |'
        )
    lines += [
        "",
        "Rows are predeclared comparisons, not permission to select a different strategy after seeing the final period.",
        "",
        "## Feature ablation and architecture stability",
        "",
        "| Family / architecture | Price-only return | Context return | Context minus price |",
        "|---|---:|---:|---:|",
    ]
    for family in cfg["families"]:
        for arch in cfg["architectures"]:
            a = results["strategies"][f"{family}_{arch}_price"]["net_return"]
            b = results["strategies"][f"{family}_{arch}_context"]["net_return"]
            lines.append(f"| {family} / {arch} | {fmt(a)} | {fmt(b)} | {fmt(b-a)} |")
    lines += [
        "",
        "The two feature groups may select different inner-fold parameters; this is a pipeline-level ablation, not an isolated causal estimate for a single feature.",
        "",
        "| Development candidate | Outer-fold returns | Robust selection score |",
        "|---|---|---:|",
    ]
    for entry in lock["ranking"]:
        lines.append(
            f'| {entry["strategy"]} | {", ".join(fmt(x) for x in entry["fold_returns"])} | {entry["score"]:.3f} |'
        )
    lines += [
        "",
        "## Calibration and prediction quality",
        "",
        f'Direction accuracy {fmt(c["accuracy"])}; balanced accuracy {fmt(c["balanced_accuracy"])}; '
        f'precision {fmt(c["precision"])}; recall {fmt(c["recall"])}; F1 {c["f1"]:.3f}; '
        f'ROC AUC {c["auc"]:.3f}; Brier {c["brier"]:.3f}; log loss {c["log_loss"]:.3f}; '
        f'expected calibration error {c["expected_calibration_error"]:.3f}; return MAE {fmt(c["return_mae"])}. '
        f'{c["n"]} labelled eligible stock-day observations. Five-session labels overlap; no independent-observation significance claim.',
        "",
        "Reliability bins and per-stock classification metrics are in the selected strategy metrics JSON. Profitability selection did not optimize accuracy.",
        "",
        "## NETWEB, E2E and TATACOMM",
        "",
        "| Symbol | Selected-strategy closed-trade P&L | Eligible classified observations |",
        "|---|---:|---:|",
    ]
    for symbol in ["NETWEB", "E2E", "TATACOMM"]:
        n = chosen.get("per_stock_classification", {}).get(symbol, {}).get("n", 0)
        lines.append(f'| {symbol} | {rupees(chosen["per_stock_pnl"][symbol])} | {n} |')
    lines += [
        "",
        "## Regime stability",
        "",
        "| Regime | Sessions | Compounded selected-day return |",
        "|---|---:|---:|",
    ]
    for name, r in chosen["regimes"].items():
        lines.append(
            f'| {name} | {r["sessions"]} | {fmt(r["compounded_selected_day_return"])} |'
        )
    lines += [
        "",
        "Regime returns combine only the indicated daily observations; they are not standalone tradable strategies. Trend uses the previous available market state, and the high-volatility threshold is fixed at 20%.",
        "",
        "## Gate results",
        "",
    ]
    lines += [
        f'- {"PASS" if passed else "FAIL"}: {name.replace("_"," ")}'
        for name, passed in results["criteria"].items()
    ]
    lines += [
        "",
        "## Provenance and material limitations",
        "",
        "Actual Yahoo Finance NSE daily OHLCV/index series are cached with source URLs, retrieval timestamps and SHA-256 hashes in data/manifest.json. "
        "No Groww credentials were present in the environment; no account was accessed. The optional Groww adapter is read-only and untested against an authenticated account. "
        "The earlier /mnt/data/groww_orb_bot.py file was absent.",
        "",
        "The 21-stock universe was chosen from current survivors, not point-in-time historical NSE membership. NETWEB begins in July 2023. "
        "E2E pre-mainboard history is excluded from eligibility. Vendor event completeness, non-cash demerger/rights entitlements, auction depth, circuit queues, "
        "partial fills and exit liquidity are not independently verified. These block deployment regardless of profitability. "
        "The current delivery tariff is applied to all dates; returns exclude investor tax, data subscriptions and exact historical bill rounding.",
        "",
        "The engine is deliberately conservative: full five-session exits, T+2-only cash reuse, 90% gross entry cap, 10% stock entry cap, "
        "30% proxy-sector entry cap and a 20% portfolio drawdown halt. This settlement rule materially reduces exposure. "
        "Buy/hold has different exposure and no trading halt. No intraday stop-loss outcomes are invented.",
        "",
        "News sentiment and fundamentals were **not tested** because a trustworthy publication-time, vintage-preserved Indian archive was unavailable. "
        "No current values were attached to old dates. ORB/VWAP were **not tested** because an intraday archive was unavailable. "
        "The price-only versus Indian market/sector ablation is complete.",
        "",
        "Model search is bounded, not exhaustive. Bootstrap intervals do not remove selection bias or establish future returns. "
        "The final 2025 period is now consumed. Any changes require a fresh preregistered future holdout; 2026 has not been used for returns/features here.",
        "",
        "## Reproduce and inspect",
        "",
        "See README.md for commands. PROTOCOL.md contains methodology, fee sources, parameter ranges, blockers and chronology. "
        "experiment.json controls the experiment. artifacts/search stores every trial. artifacts/selection_lock.json records the selection and hashes. "
        "artifacts/test_access.json records the final-test access. artifacts/final stores predictions, fills, round trips, equity and detailed metrics. "
        "The local dashboard reads the same cached artifacts and never places orders.",
        "",
    ]
    markdown = "\n".join(lines)
    (ROOT / "reports/RESULTS.md").write_text(markdown)
    # A standalone readable HTML report with the same narrative and tabular evidence.
    body = []
    in_table = False
    header = False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not in_table:
                body.append('<div class="scroll"><table>')
                in_table = True
                header = True
            tag = "th" if header else "td"
            header = False
            body.append(
                "<tr>"
                + "".join(
                    f"<{tag}>{html.escape(c.strip())}</{tag}>"
                    for c in line.strip("|").split("|")
                )
                + "</tr>"
            )
            continue
        if in_table:
            body.append("</table></div>")
            in_table = False
        if line.startswith("# "):
            body.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line.startswith("## "):
            body.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line:
            body.append("<p>" + html.escape(line).replace("**", "") + "</p>")
    if in_table:
        body.append("</table></div>")
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>NSE Lab evaluation</title><style>body{font:16px/1.65 system-ui;background:#0b111a;color:#dce5f2;max-width:1200px;margin:auto;padding:40px}h1,h2{color:#38d6b2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #304055;text-align:left}.scroll{overflow:auto}p{max-width:1000px}a{color:#38d6b2}</style>'
        + "".join(body)
        + "</html>"
    )
    (ROOT / "reports/RESULTS.html").write_text(document)
    (ROOT / "ui/report.html").write_text(document)
    print("Saved reports/RESULTS.md, RESULTS.html and comparison.csv")
