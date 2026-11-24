# NSE Research Lab — frozen experiment protocol

Recorded before opening 2025 outcomes. This is historical research, not an investment recommendation or a live trading system.

## Question and design

Can a bounded model search produce a stable long-only Indian equity strategy after execution costs? Compare 12 candidates: two regression families (regularized ridge and LightGBM), three architectures (pooled, independent per-stock, and history-weighted hybrid), and two feature groups (price/volume and price/volume plus Indian market/sector context). Evaluate all on identical dates and execution rules. A candidate is selected using development results alone. All 12 predeclared test comparisons are descriptive; do not choose another winner from the final test.

The fixed universe contains 21 currently surviving stocks across nine broad sector/proxy groups, including NETWEB, E2E and TATACOMM. It is a user-directed research basket, **not a historical point-in-time NSE constituent universe**. Survivorship/selection bias limits every conclusion. NETWEB has no data before 27 July 2023; there is no backfill. E2E is ineligible before its 12 April 2022 mainboard migration because SME lot rules are not simulated.

## Calendar and leakage controls

- Source snapshot: daily observations from 1 January 2020 to 23 November 2026. All 2026 prices are reserved and excluded from this experiment. Subsequent split metadata is used only to reverse vendor normalization into contemporaneous prices.
- Development: through 31 December 2024. Outer evaluations: July–December 2023, January–June 2024, July–December 2024. Each fits on expanding earlier history.
- Inner tuning: two successive 63-session windows immediately preceding each outer fit. Every fit uses only earlier data. Purge six market sessions plus require label end strictly before the validation start. This conservative pre-boundary gap also supplies the embargo; no later observations enter earlier fits.
- Within each training fold, retain its latest 63 sessions for Platt probability calibration. Fit the return models on earlier, additionally purged data. Calibration never uses outer or final labels. Return models are not refitted on calibration observations.
- LightGBM early stopping uses a separate chronological last-20% split within the return-model core, with a six-session gap. Feature selection for that split is earlier-core-only; then refit for the chosen number of trees using the entire permitted core. Calibration and evaluation remain disjoint.
- Features use data available at signal-date close; no backward filling. Training-only median imputation, scaling, Spearman target association ranking, and removal of features with absolute pairwise correlation above 0.95. Feature-count budget is tuned within the inner folds.
- Final test: 1 January–31 December 2025. Freeze candidate parameters, fitted models, development-selected candidate, source/data/code hashes in `artifacts/selection_lock.json` before opening it. A persistent test-access ledger refuses automatic repeat evaluation or development after test access. 2026 remains a future independent validation option.

## Models and bounded tuning

Target: five-session **price return** from the open of session t+1 to the open of session t+6, following a signal at close t. Dividend entitlements are included separately in portfolio P&L; target excludes them. Classification is secondary: probability that this same price return exceeds zero.

Price/volume features include past returns, volatility, ranges, gaps, candle body, EMA distances, RSI, relative volume, and drawdown. Context adds NIFTY price returns/volatility/trend, sector-index returns, relative strength and rolling beta. Technology is an imperfect proxy for hardware/cloud/telecom; BHARTIARTL uses infrastructure. No US inputs.

Per-stock models share the selected hyperparameters but learn separate coefficients/trees and feature selections. They require 126 eligible core observations; absent local models abstain. Pooled/hybrid can use shorter histories after the common 100-session warmup. Hybrid local weight is `min(0.75, n/(n+504))`, with pooled fallback. This is a fixed statistical shrinkage heuristic, not an optimized financial claim. Model availability differs by architecture and is recorded; evaluation dates remain identical.

For each model-family/feature-group pair and fit window, run exactly eight sequential Optuna TPE candidates, seed 42, first four random. Candidates are shared among the three architectures; each architecture selects its own best inner score. Local stocks are not given separate independent searches. Four fit windows (three outer and one final development-only) × four pairs × eight = **128 candidate trials**, each with two inner folds. This is a deliberately bounded first experiment, not exhaustive or globally optimal convergence.

Search space: 8/12/17/25 retained features; entry return threshold 0.3%/0.6%/1.0%/1.5%; ridge alpha 0.1–1000 log scale; LightGBM leaves 7/15/31, learning rate 0.015–0.10 log scale, minimum child observations 30/60/120, L2 0.1–20 log scale, maximum 240 trees and 20-round early stopping.

Objective per inner fold: the worse of base-cost and double-cost `(Sharpe − 2 × absolute drawdown)`, with fewer than five completed trades assigned −5. Across inner folds: median minus half the standard deviation. The sampler optimizes the average architecture score; each architecture chooses the best candidate in that same explored set. Across outer windows, use the same median/stability penalty to select one ML candidate. Sharpe uses a fixed 6% annual hurdle; also report zero-hurdle Sharpe. Cash earns zero.

## Execution, costs and benchmarks

Start with ₹10 lakh, no borrowing/shorting/leverage. Signals rank eligible stocks after the close; next session open is the reference fill, with adverse slippage. Rebalance every five market sessions: fully close each cycle, then fund new purchases from **settled cash only**. Use conservative fixed T+2 throughout, even when actual Indian settlement became T+1. Because newly sold capital cannot fund same-open purchases, exposure will often be materially below the 90% cap. This deliberate conservative implementation is part of the strategy, not an estimate of optimal capital use.

Maximum 10 positions, 10% entry allocation per stock, 30% per proxy sector, 90% gross entry exposure. Caps apply when adding positions; overnight moves can exceed them. A 20% close-to-peak portfolio drawdown permanently halts new entries and attempts liquidation at the following open. No fictional intraday stop fills. Minimum trailing 20-session average daily turnover ₹2 crore; order notional at most 0.1% of that known trailing turnover. Integer entry shares; split entitlements update quantities. Exits are attempted at the next prescribed open; absent/zero-volume or full-session flat bars defer fills. Daily data cannot verify opening-auction depth, partial fills, circuit queues, or auction penalties; full-day locked-bar screening is an execution feasibility approximation, not a signal feature.

Adverse slippage: 10 bps each side, 30 bps for NETWEB/E2E, multiplied in stress tests. Exact open availability and exit participation are unverified with daily data. Stress multiplies **all charges and slippage** by 2 and 3, and re-simulates cash/integer orders; trade paths may therefore differ. These tests are scenarios, not forecasts.

Cost model uses the current standard Groww-like delivery schedule uniformly across history: brokerage lower of ₹20/0.1%, minimum ₹5 subject to 2.5% cap; STT 0.1% both sides; buy stamp 0.015%; NSE exchange 0.00307%; SEBI 0.0001%; IPFT 0.0001%; sell DP ₹20 (₹3.5 below ₹100 debit), plus 18% GST on brokerage, exchange, SEBI, IPFT and DP. Actual historical tariff changes, invoice rounding, account-specific plans, investor capital-gains/income taxes, API/data subscriptions and financing/opportunity costs are excluded. This is net of modelled trading costs, before investor tax.

Splits: source OHLC is split-normalized; reverse later split factors to reconstruct historical nominal prices and volume; apply split ratios to held quantities. Dividend entitlement is recognized on the ex-date as a **non-spendable receivable**, not reinvested; no payment-date claim. Non-cash demergers/rights/merger entitlements and event completeness have not been officially reconciled. This is a deployment blocker even if statistical criteria pass.

Fixed baselines: cash, broad equal-entry-weight buy/hold (90% total target, all eligible stocks, same sector caps), 60-session positive momentum, EMA20/EMA50 trend rule, and RSI<30 five-session mean reversion. Trading baselines use the same five-session engine, risk rules and cost assumptions. Buy/hold does not use the drawdown trading halt and holds to the last test open. NIFTY 50 price-index return is an additional informational full-exposure benchmark, with no dividends/costs and not directly investable. These exposure differences are reported rather than hidden.

## Reporting and decision gate

Report net/gross P&L, explicit charges, slippage, turnover, completed trades, hit rate/profit factor, CAGR, volatility, Sharpe, drawdown, exposures, rejections and residual positions. Report direction accuracy/balanced accuracy, precision/recall/F1, AUC, Brier/log loss, reliability bins, calibration error, return MAE and rank correlation on eligible labelled observations. Samples overlap and are not independent trials.

Use 1,000 circular moving-block bootstrap replicates, 20-session blocks, fixed seed, for return/Sharpe intervals and paired return differences versus buy/hold. Intervals are conditional on the selected strategy and observed year; they **do not correct for model-search multiplicity** or future regime changes. Report fixed regime bins (positive/nonpositive prior market trend; prior realized market volatility above/below 20%). No regime threshold is fitted on the test.

Statistical gate: selected candidate has positive net return; exceeds buy/hold and NIFTY price return; Sharpe above 0.5; positive return under double costs; at least 30 completed trades; positive 95% bootstrap lower return bound; no residual positions. Independently, current data/execution limitations prevent live deployment. Display research-only regardless of profitability. No live account or order actions are implemented.

Sentiment/fundamentals ablation is explicitly unavailable unless a reliable, licensed, timestamped Indian publication-vintage archive is obtained. Do not fetch today's news/fundamentals and attach them to earlier dates. A tested availability-time join exists, but no unverified sentiment model is trained. ORB/VWAP are unavailable without a validated intraday archive.

## Primary source references

- [Groww pricing](https://groww.in/pricing) and [delivery brokerage/DP details](https://groww.in/help/stocks/sx-pricing/what-is-the-charge-for-buy-order-delivery), checked 24 November 2026. The help article and pricing page differ on exchange charge; the higher current pricing-page 0.00307% is used.
- [NSE statutory fees and taxes](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies).
- [Groww historical-data API](https://groww.in/trade-api/docs/curl/backtesting). No token was present; adapter is GET-only and not exercised against an account.
- [E2E mainboard migration notice](https://nsearchives.nseindia.com/corporate/E2E_08042022183417_NSEIntimation08042022.pdf).
- Yahoo Finance actual chart endpoints and SHA-256 source snapshots are enumerated in `data/manifest.json`. Unofficial endpoint availability/licensing may change; no data is redistributed publicly.
- [Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html), [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
