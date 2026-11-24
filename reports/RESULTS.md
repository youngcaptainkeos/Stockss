# NSE Research Lab — final historical evaluation

**Status: RESEARCH ONLY.** Development selected **lightgbm_hybrid_context** before the final test. Its 2025 net return was **-15.58%**, or **₹-155,810** on ₹10 lakh. The statistical gate failed. No live deployment is justified by this experiment.

## Evidence at a glance

- Final period: 2025-01-01 to 2025-12-31, 249 market sessions; evaluated once after a frozen selection lock.
- Broad capped buy/hold: **4.40%**; NIFTY 50 price index: **10.54%**. The NIFTY comparator has full exposure, no dividends and no trading costs.
- Selected strategy Sharpe: **-1.91** (6% annual hurdle); max drawdown: **-17.80%**; mean exposure: **49.57%**.
- 318 closed trades; win rate 35.85%; explicit fees **₹70,794**, adverse slippage **₹72,579**.
- Base / 2× / 3× total-cost scenario returns: -15.58% / -20.41% / -22.84%.
- 95% moving-block bootstrap interval for net return: **-32.46% to 5.53%**. Paired excess return versus buy/hold: **-33.06% to -7.58%**. These conditional intervals do not correct for model-search selection.

## What was actually trained

128 candidate trials across 16 bounded searches, two inner folds each, three outer validation windows and a final development-only fit. Twelve ML candidates compare ridge versus LightGBM, pooled versus independent per-stock versus history-weighted hybrid, and price/volume versus additional Indian market/sector features. The complete configuration and objective were specified before test access in PROTOCOL.md.

The final fit has 20 local stock models. E2E had only 118 eligible core observations against the fixed 126-observation minimum; its per-stock model abstained and the hybrid used the pooled model. NETWEB had 176 eligible core observations. Per-stock modelling is implemented: each eligible symbol has separately fitted coefficients/trees and selected features. Hyperparameters are shared within each architecture/family/feature group; they are not independently optimized for every stock. Sparse local histories abstain; the hybrid falls back to pooled predictions. There is no global-optimum or future-profit guarantee.

## Final comparisons

| Strategy | Net return | Sharpe | Max drawdown | Fees | Slippage | Closed trades | Mean exposure |
|---|---:|---:|---:|---:|---:|---:|---:|
| ridge_pooled_price | -7.98% | -1.36 | -12.66% | ₹35,726 | ₹52,772 | 136 | 25.40% |
| ridge_per_stock_price | -6.19% | -1.11 | -10.62% | ₹73,211 | ₹65,263 | 299 | 49.75% |
| ridge_hybrid_price | -8.81% | -1.35 | -15.87% | ₹43,147 | ₹57,582 | 162 | 31.67% |
| ridge_pooled_context | -4.02% | -0.91 | -10.94% | ₹34,878 | ₹52,839 | 130 | 24.45% |
| ridge_per_stock_context | 3.96% | -0.16 | -9.04% | ₹60,950 | ₹56,227 | 224 | 41.20% |
| ridge_hybrid_context | -9.93% | -1.49 | -15.71% | ₹50,134 | ₹58,540 | 193 | 36.36% |
| lightgbm_pooled_price | 0.36% | -1.09 | -4.27% | ₹10,815 | ₹20,085 | 37 | 7.37% |
| lightgbm_per_stock_price | -6.39% | -1.14 | -11.71% | ₹72,706 | ₹65,750 | 299 | 49.67% |
| lightgbm_hybrid_price | -2.63% | -1.53 | -4.77% | ₹13,494 | ₹23,878 | 47 | 9.34% |
| lightgbm_pooled_context | -9.46% | -1.38 | -12.88% | ₹72,933 | ₹71,162 | 321 | 49.68% |
| lightgbm_per_stock_context | -5.64% | -1.04 | -11.39% | ₹73,347 | ₹66,137 | 309 | 49.61% |
| lightgbm_hybrid_context **selected** | -15.58% | -1.91 | -17.80% | ₹70,794 | ₹72,579 | 318 | 49.57% |
| cash | 0.00% | 0.00 | 0.00% | ₹0 | ₹0 | 0 | 0.00% |
| buy_hold | 4.40% | -0.08 | -13.23% | ₹3,441 | ₹2,054 | 21 | 85.15% |
| momentum | -15.49% | -2.43 | -17.42% | ₹64,384 | ₹59,208 | 286 | 46.24% |
| ema | -10.27% | -2.07 | -13.67% | ₹60,612 | ₹49,168 | 260 | 42.60% |
| mean_reversion | -8.23% | -2.48 | -9.62% | ₹26,751 | ₹22,158 | 100 | 18.88% |

Rows are predeclared comparisons, not permission to select a different strategy after seeing the final period.

## Feature ablation and architecture stability

| Family / architecture | Price-only return | Context return | Context minus price |
|---|---:|---:|---:|
| ridge / pooled | -7.98% | -4.02% | 3.95% |
| ridge / per_stock | -6.19% | 3.96% | 10.15% |
| ridge / hybrid | -8.81% | -9.93% | -1.12% |
| lightgbm / pooled | 0.36% | -9.46% | -9.82% |
| lightgbm / per_stock | -6.39% | -5.64% | 0.74% |
| lightgbm / hybrid | -2.63% | -15.58% | -12.96% |

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
| NETWEB | ₹-17,440 | 242 |
| E2E | ₹-79,789 | 242 |
| TATACOMM | ₹-1,985 | 242 |

## Regime stability

| Regime | Sessions | Compounded selected-day return |
|---|---:|---:|
| trend_positive | 171 | -13.13% |
| trend_nonpositive | 78 | -2.82% |
| high_volatility | 15 | 4.82% |
| lower_volatility | 234 | -19.46% |

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
