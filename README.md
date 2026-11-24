# NSE Research Lab

A local, research-only Indian equity trading laboratory. It downloads real historical NSE prices, compares bounded and chronologically tuned models, simulates delivery trades after costs, and replays the held-out year in a dark web terminal. **There is no live trading or order-placement code.**

Start with **[the results report](reports/RESULTS.md)** and **[the experiment protocol](PROTOCOL.md)**. The report is generated only after a frozen, one-time final evaluation. All 2026 prices remain outside this experiment.

## Open the dashboard

From this directory:

```bash
./run.sh serve
```

Open **http://127.0.0.1:8765**. To change the port:

```bash
./run.sh serve --port 8766
```

The interface has historical candles and volume, buy/sell markers, instrument and strategy selection, play/pause, speed controls, a date slider, date seeking, portfolio P&L and exposure, benchmarks, and an execution ledger. The Evaluation page shows the fixed final comparison, costs, classification metrics and uncertainty. All charts and fonts work without external CDN dependencies. Only the loopback interface is bound.

## Commands

| Command | What it does |
|---|---|
| `./run.sh ingest` | Fetch/cache the actual daily source snapshots; existing snapshots are reused. |
| `./run.sh develop` | Run nested chronological tuning through 2024 and freeze all model selections. |
| `./run.sh final-test` | Open 2025 once, evaluate predeclared models, record access and export replay data. |
| `./run.sh report` | Regenerate readable reports from cached results; does not train anything. |
| `./run.sh verify` | Reconcile P&L, fees, cash, chronology, source hashes and search counts. |
| `./run.sh serve` | Serve the local read-only dashboard. |
| `.venv/bin/python -m pytest -q` | Run focused leakage, timing, execution, corporate-action and fee tests. |

For a new checkout/environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

The original completed experiment refuses development or repeat final-test access once the test is consumed. This prevents an accidental silent retune. Recomputing an identical frozen experiment in a separate copy is a **reproduction**, not another untouched test. New strategy ideas require a separate experiment ID, separate artifact directory/checkout and a genuinely unused final period. Never delete the access ledger to present a consumed test as untouched.

## Code map

```text
experiment.json           Single experiment configuration and search/risk budgets
PROTOCOL.md               Preregistered decisions, assumptions, sources and limits
src/nselab/
  cli.py                  Small command-line entry point
  common.py               Paths, JSON persistence and checksums
  data.py                 Cached Yahoo ingestion; optional GET-only Groww adapter
  features.py             Close-time features, next-open labels, purging and PIT joins
  costs.py                India delivery fees with component-level accounting
  models.py               Train-only selection, ridge/trees, per-stock/hybrid/calibration
  backtest.py             Event-driven cash/settlement/position and execution accounting
  metrics.py              Return, risk, classification, regimes and block bootstrap
  experiment.py           Nested tuning, selection lock and one-time final evaluation
  verification.py         Independent cached-artifact reconciliation
  report.py               Human-readable report generation from existing results
  server.py               Loopback-only static web server
ui/
  index.html              Semantic dashboard layout
  styles.css              Dark responsive trading-terminal theme
  app.js                  Replay state, controls and canvas charts
  replay.json             Generated historical research data, never live orders
  report.html             Generated standalone results report
 tests/                   Synthetic fixtures used only to test accounting/leakage
 data/raw/                Actual vendor responses (ignored by Git)
 data/processed/          Validated nominal and split-normalized daily data (ignored)
 data/manifest.json       Source URLs, retrieval timestamps and hashes
 artifacts/search/        Every candidate and inner-fold objective
 artifacts/models/        Local trusted fitted model files (ignored by Git)
 artifacts/selection_lock.json  Frozen choices, chronology and input/model/code hashes
 artifacts/test_access.json     Final test access record
 artifacts/final/          Per-strategy equity, predictions, fills, P&L and detailed metrics
 reports/                 Readable final results and comparison CSV
 work/                    Logs and scratch files (ignored by Git)
```

Source code is formatted with Black. The UI uses plain JavaScript and CSS so there is no front-end build toolchain to maintain. The Python package exposes small functions rather than a single trading script. Tests use synthetic accounting fixtures; **no synthetic data is used in any reported result**. Model files are local `joblib` artifacts: only load models you trust.

## Change or extend the research

- Universe, periods, initial capital, risk limits, feature-group/family lists, seed and trial budget live in `experiment.json`. The current experiment is sealed; change these only in a new experiment copy.
- Add an indicator to `features.py` using data available at the close, add it to the appropriate feature list and extend the future-perturbation test. Do not use centered windows, backwards fills, full-sample normalization or future corporate fundamentals.
- Add a technical rule in `baseline_signals` and preregister it before opening a new final period. Backtests must continue to use the common engine.
- Add a model through `ReturnModel`; fit preprocessing and feature selection only on the permitted training segment. Keep calibration and evaluation separate.
- `ModelBundle` fits separate models per eligible stock. Hyperparameters are shared within the architecture; the stock models themselves are independent. Hybrid predictions shrink local models toward the pooled prediction based on available history.
- For a new provider, normalize and validate its corporate actions, session timezone, missing sessions and nominal prices before using it. Do not silently mix source series.
- `point_in_time_join` accepts event **availability timestamps**, not fiscal period dates. It is tested but cannot certify a news/fundamental archive's historical vintages. Those ablations were not run here.

## Source and fee limits

No Groww credential was present in the environment and the earlier `/mnt/data/groww_orb_bot.py` was absent. Actual Yahoo Finance daily NSE OHLCV and Indian index data were fetched instead. The optional Groww adapter accepts `GROWW_ACCESS_TOKEN` only through the process environment and uses a single historical-data GET endpoint; it has no account/order functions. It has not been authenticated or validated here. Do not put tokens into commands, reports, Git, or chat. `.env` files are ignored and are not automatically loaded.

The fee model follows [Groww's standard pricing](https://groww.in/pricing) and [NSE statutory levies](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies), verified 24 November 2026. It applies that schedule consistently to historical periods, not historically exact invoices. Net results are before investor tax and subscription costs.

The fixed universe is survivor-selected. Corporate-action completeness and executable opening-auction liquidity are not certified. E2E has sparse/zero-volume history; NETWEB is newly listed relative to the training period. Sector classifications are proxies. Daily bars cannot establish circuit queue fills or intraday ORB/VWAP outcomes. Full details and the conservative settlement/cash treatment are in the protocol.

## Reproducibility record

The original snapshot, immutable hashes, seed, complete trial logs, selected parameters, fitted models, split dates, accounting ledgers and pinned runtime dependencies are retained locally. `requirements.lock` is generated from the versions used for this run. Source snapshots are not published or sent elsewhere. The optional source endpoint may change or restrict access; cached data supports offline reruns and inspection.

In this run, the final fit contains 20 local stock models. E2E had 118 eligible core observations, below the fixed 126-row minimum, so the per-stock architecture abstains on E2E and the hybrid uses its pooled prediction. NETWEB had 176 eligible core observations and a local model.

The final test is not a promise of future performance. The selected model stays selected even if another candidate looks better in 2025; do not optimize retrospectively against the final comparison table.
