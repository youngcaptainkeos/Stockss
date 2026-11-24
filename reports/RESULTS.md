# NSE Research Lab — final historical evaluation

**Status: RESEARCH ONLY.** Development selected **lightgbm_hybrid_context** before the final test. Its 2025 net return was **-4.49%**, or **₹-44,919** on ₹10 lakh. The statistical gate failed. No live deployment is justified by this experiment.

## Evidence at a glance

- Final period: 2025-01-01 to 2025-12-31, 249 market sessions; evaluated once after a frozen selection lock.
- Broad capped buy/hold: **4.40%**; NIFTY 50 price index: **10.54%**. The NIFTY comparator has full exposure, no dividends and no trading costs.
- Selected strategy Sharpe: **-0.82** (6% annual hurdle); max drawdown: **-18.39%**; mean exposure: **72.90%**.
- 132 closed trades; win rate 43.18%; explicit fees **₹29,414**, adverse slippage **₹23,484**.
- Base / 2× / 3× total-cost scenario returns: -4.49% / -9.84% / -15.19%.
- 95% moving-block bootstrap interval for net return: **-28.54% to 29.13%**. Paired excess return versus buy/hold: **-17.58% to 3.05%**. These conditional intervals do not correct for model-search selection.

## What was actually trained

128 candidate trials across 16 bounded searches, two inner folds each, three outer validation windows and a final development-only fit. Twelve ML candidates compare ridge versus LightGBM, pooled versus independent per-stock versus history-weighted hybrid, and price/volume versus additional Indian market/sector features. The complete configuration and objective were specified before test access in PROTOCOL.md.

The final fit has 20 local stock models. E2E had only 118 eligible core observations against the fixed 126-observation minimum; its per-stock model abstained and the hybrid used the pooled model. NETWEB had 176 eligible core observations. Per-stock modelling is implemented: each eligible symbol has separately fitted coefficients/trees and selected features. Hyperparameters are shared within each architecture/family/feature group; they are not independently optimized for every stock. Sparse local histories abstain; the hybrid falls back to pooled predictions. There is no global-optimum or future-profit guarantee.

## Final comparisons

| Strategy | Net return | Sharpe | Max drawdown | Fees | Slippage | Closed trades | Mean exposure |
|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_pooled_price | -3.30% | -1.33 | -10.10% | ₹13,148 | ₹13,104 | 60 | 18.83% |
| ridge_per_stock_price | -0.06% | -0.48 | -12.65% | ₹31,530 | ₹22,690 | 136 | 73.84% |
| ridge_hybrid_price | -2.74% | -1.27 | -10.11% | ₹16,759 | ₹15,710 | 71 | 23.74% |
| ridge_pooled_context | -0.84% | -0.93 | -10.13% | ₹13,131 | ₹12,935 | 60 | 18.58% |
| ridge_per_stock_context | 7.28% | 0.21 | -8.10% | ₹30,121 | ₹22,721 | 112 | 43.02% |
| ridge_hybrid_context | 1.59% | -0.52 | -10.26% | ₹23,762 | ₹20,654 | 95 | 33.00% |
| lightgbm_pooled_price | 0.98% | -2.13 | -1.50% | ₹3,509 | ₹4,905 | 19 | 3.51% |
| lightgbm_per_stock_price | -0.13% | -0.47 | -15.56% | ₹30,893 | ₹22,198 | 132 | 74.56% |
| lightgbm_hybrid_price | -1.18% | -2.95 | -2.33% | ₹5,126 | ₹6,325 | 26 | 4.83% |
| lightgbm_pooled_context | -7.32% | -1.22 | -10.53% | ₹28,761 | ₹22,508 | 130 | 70.49% |
| lightgbm_per_stock_context | 0.68% | -0.42 | -14.73% | ₹30,868 | ₹22,191 | 134 | 73.09% |
| lightgbm_hybrid_context **selected** | -4.49% | -0.82 | -18.39% | ₹29,414 | ₹23,484 | 132 | 72.90% |
| cash | 0.00% | 0.00 | 0.00% | ₹0 | ₹0 | 0 | 0.00% |
| buy_hold | 4.40% | -0.08 | -13.23% | ₹3,441 | ₹2,054 | 21 | 85.15% |
| momentum | -9.59% | -1.80 | -13.09% | ₹28,058 | ₹20,710 | 123 | 62.41% |
| ema | -2.81% | -1.07 | -9.03% | ₹30,266 | ₹21,636 | 123 | 56.23% |
| mean_reversion | -5.86% | -2.09 | -8.97% | ₹18,965 | ₹14,562 | 73 | 19.64% |

Rows are predeclared comparisons, not permission to select a different strategy after seeing the final period.

## Feature ablation and architecture stability

| Family / architecture | Price-only return | Context return | Context minus price |
|---|---:|---:|---:|
| ridge / pooled | -3.30% | -0.84% | 2.45% |
| ridge / per_stock | -0.06% | 7.28% | 7.34% |
| ridge / hybrid | -2.74% | 1.59% | 4.33% |
| lightgbm / pooled | 0.98% | -7.32% | -8.29% |
| lightgbm / per_stock | -0.13% | 0.68% | 0.81% |
| lightgbm / hybrid | -1.18% | -4.49% | -3.31% |

The two feature groups may select different inner-fold parameters; this is a pipeline-level ablation, not an isolated causal estimate for a single feature.

| Development candidate | Outer-fold returns | Robust selection score |
|---|---|---:|
| buy_hold | 20.57%, 19.71%, 5.50% | 1.481 |
| lightgbm_hybrid_context | 6.54%, 8.55%, 0.24% | -1.120 |
| momentum | 9.89%, 6.61%, -3.70% | -1.204 |
| lightgbm_pooled_context | 6.75%, 4.27%, -4.73% | -1.342 |
| ema | 7.01%, 9.98%, -7.50% | -1.360 |
| lightgbm_per_stock_context | 5.51%, 2.95%, -7.32% | -1.624 |
| lightgbm_pooled_price | 6.81%, 2.70%, 1.92% | -1.633 |
| ridge_hybrid_price | 5.82%, 3.37%, -3.29% | -1.668 |
| lightgbm_hybrid_price | 9.70%, 4.45%, -6.61% | -1.705 |
| ridge_per_stock_price | 4.94%, 1.19%, -6.32% | -2.539 |
| ridge_pooled_price | 1.43%, 5.52%, 3.14% | -2.594 |
| ridge_pooled_context | 0.59%, 9.50%, 2.96% | -2.616 |
| ridge_per_stock_context | 6.73%, -0.22%, -6.63% | -2.982 |
| ridge_hybrid_context | 0.41%, 4.63%, -1.98% | -3.701 |
| mean_reversion | 2.49%, -2.18%, -3.18% | -4.938 |
| cash | 0.00%, 0.00%, 0.00% | -5.000 |
| lightgbm_per_stock_price | -0.64%, 3.76%, -13.35% | -5.955 |

## Calibration and prediction quality

Direction accuracy 49.31%; balanced accuracy 50.00%; precision 0.00%; recall 0.00%; F1 0.000; ROC AUC 0.482; Brier 0.259; log loss 0.711; expected calibration error 0.091; return MAE 2.94%. 5082 labelled eligible stock-day observations. Five-session labels overlap; no independent-observation significance claim.

Reliability bins and per-stock classification metrics are in the selected strategy metrics JSON. Profitability selection did not optimize accuracy.

## NETWEB, E2E and TATACOMM

| Symbol | Selected-strategy closed-trade P&L | Eligible classified observations |
|---|---:|---:|
| NETWEB | ₹5,968 | 242 |
| E2E | ₹-37,179 | 242 |
| TATACOMM | ₹6,394 | 242 |

## Regime stability

| Regime | Sessions | Compounded selected-day return |
|---|---:|---:|
| trend_positive | 171 | 0.06% |
| trend_nonpositive | 78 | -4.55% |
| high_volatility | 15 | 4.08% |
| lower_volatility | 234 | -8.23% |

Regime returns combine only the indicated daily observations; they are not standalone tradable strategies. Trend uses the previous available market state, and the high-volatility threshold is fixed at 20%.

## Gate results

- FAIL: positive net return
- FAIL: beats buy hold
- FAIL: beats nifty price
- FAIL: sharpe above 0 5
- FAIL: positive at double cost
- PASS: at least 30 trades
- FAIL: positive 95 percent return lower bound
- PASS: all positions liquidated

## Provenance and material limitations

Actual Yahoo Finance NSE daily OHLCV/index series are cached with source URLs, retrieval timestamps and SHA-256 hashes in data/manifest.json. No Groww credentials were present in the environment; no account was accessed. The optional Groww adapter is read-only and untested against an authenticated account. The earlier /mnt/data/groww_orb_bot.py file was absent.

The 21-stock universe was chosen from current survivors, not point-in-time historical NSE membership. NETWEB begins in July 2023. E2E pre-mainboard history is excluded from eligibility. Vendor event completeness, non-cash demerger/rights entitlements, auction depth, circuit queues, partial fills and exit liquidity are not independently verified. These block deployment regardless of profitability. The current delivery tariff is applied to all dates; returns exclude investor tax, data subscriptions and exact historical bill rounding.

The engine is deliberately conservative: full five-session exits, T+2-only cash reuse, 90% gross entry cap, 10% stock entry cap, 30% proxy-sector entry cap and a 20% portfolio drawdown halt. This settlement rule materially reduces exposure. Buy/hold has different exposure and no trading halt. No intraday stop-loss outcomes are invented.

News sentiment and fundamentals were **not tested** because a trustworthy publication-time, vintage-preserved Indian archive was unavailable. No current values were attached to old dates. ORB/VWAP were **not tested** because an intraday archive was unavailable. The price-only versus Indian market/sector ablation is complete.

Model search is bounded, not exhaustive. Bootstrap intervals do not remove selection bias or establish future returns. The final 2025 period is now consumed. Any changes require a fresh preregistered future holdout; 2026 has not been used for returns/features here.

## Reproduce and inspect

See README.md for commands. PROTOCOL.md contains methodology, fee sources, parameter ranges, blockers and chronology. experiment.json controls the experiment. artifacts/search stores every trial. artifacts/selection_lock.json records the selection and hashes. artifacts/test_access.json records the final-test access. artifacts/final stores predictions, fills, round trips, equity and detailed metrics. The local dashboard reads the same cached artifacts and never places orders.
