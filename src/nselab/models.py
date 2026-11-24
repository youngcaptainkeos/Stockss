"""Train-only feature selection, bounded regression models, chronological calibration."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from .features import PRICE_FEATURES, CONTEXT_FEATURES, training_before
from .common import config


class ReturnModel:
    def __init__(self, family, params, feature_set):
        self.family = family
        self.params = params
        self.feature_set = feature_set

    def fit(self, data):
        base = PRICE_FEATURES + (
            CONTEXT_FEATURES if self.feature_set == "context" else []
        )
        ranks = (
            data[base]
            .corrwith(data.target, method="spearman")
            .abs()
            .fillna(0)
            .sort_values(ascending=False)
        )
        selected = []
        corr = data[base].corr().abs()
        for name in ranks.index:
            if data[name].notna().sum() < 20 or data[name].std() < 1e-10:
                continue
            if any(corr.loc[name, other] > 0.95 for other in selected):
                continue
            selected.append(name)
            if len(selected) >= self.params["k"]:
                break
        self.features = selected
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        # Feature ranking is re-fit on core data before an early-stopping split below.
        if self.family == "lightgbm":
            ds = sorted(data.date.unique())
            boundary = pd.Timestamp(ds[max(1, int(len(ds) * 0.8))])
            core = training_before(data, boundary, 6)
            valid = data[data.date >= boundary]
            if len(core) >= 80 and len(valid) >= 20:
                # Recompute selection on the earlier core to isolate early stopping.
                cr = (
                    core[base]
                    .corrwith(core.target, method="spearman")
                    .abs()
                    .fillna(0)
                    .sort_values(ascending=False)
                )
                cc = core[base].corr().abs()
                self.features = []
                for name in cr.index:
                    if core[name].notna().sum() < 20 or core[name].std() < 1e-10:
                        continue
                    if any(cc.loc[name, other] > 0.95 for other in self.features):
                        continue
                    self.features.append(name)
                    if len(self.features) >= self.params["k"]:
                        break
                im = SimpleImputer(strategy="median")
                x = im.fit_transform(core[self.features])
                xv = im.transform(valid[self.features])
                early = self._tree(240)
                early.fit(
                    x,
                    core.target,
                    eval_set=[(xv, valid.target)],
                    callbacks=[early_stopping(20, verbose=False), log_evaluation(0)],
                )
                self.iterations = max(1, early.best_iteration_)
            else:
                self.iterations = 60
        x = self.imputer.fit_transform(data[self.features])
        x = self.scaler.fit_transform(x)
        self.model = (
            Ridge(alpha=self.params["alpha"])
            if self.family == "ridge"
            else self._tree(self.iterations)
        )
        self.model.fit(x, data.target)
        self.last_label_date = str(data.label_end.max().date())
        self.rows = len(data)
        return self

    def _tree(self, n):
        p = self.params
        return LGBMRegressor(
            n_estimators=n,
            num_leaves=p["num_leaves"],
            learning_rate=p["learning_rate"],
            min_child_samples=p["min_child_samples"],
            reg_lambda=p["reg_lambda"],
            max_depth=-1,
            n_jobs=1,
            random_state=42,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )

    def predict(self, data):
        return self.model.predict(
            self.scaler.transform(self.imputer.transform(data[self.features]))
        )


class ModelBundle:
    def __init__(self, family, params, feature_set, cfg=None):
        self.family = family
        self.params = params
        self.feature_set = feature_set
        self.cfg = cfg or config()

    def fit(self, training):
        ds = sorted(training.date.unique())
        cal_start = pd.Timestamp(ds[-self.cfg["calibration_sessions"]])
        core = training_before(training, cal_start, self.cfg["purge_sessions"])
        cal = training[training.date >= cal_start]
        self.pooled = ReturnModel(self.family, self.params, self.feature_set).fit(core)
        self.locals = {}
        self.counts = {}
        for symbol, g in core.groupby("symbol"):
            self.counts[symbol] = len(g)
            if len(g) >= self.cfg["min_model_rows_per_stock"]:
                self.locals[symbol] = ReturnModel(
                    self.family, self.params, self.feature_set
                ).fit(g)
        raw = self.raw_predict(cal)
        self.calibrators = {}
        self.priors = {}
        y = (cal.target > 0).astype(int).to_numpy()
        for arch, scores in raw.items():
            mask = np.isfinite(scores)
            self.priors[arch] = float(y[mask].mean()) if mask.any() else 0.5
            if mask.sum() >= 30 and len(np.unique(y[mask])) == 2:
                # Fixed regularized Platt model; scale returns to percentage points.
                m = LogisticRegression(C=1.0, max_iter=500, random_state=42)
                m.fit((scores[mask] * 100).reshape(-1, 1), y[mask])
                self.calibrators[arch] = m
        self.audit = {
            "core_start": str(core.date.min().date()),
            "core_end": str(core.date.max().date()),
            "last_core_label": str(core.label_end.max().date()),
            "cal_start": str(cal.date.min().date()),
            "cal_end": str(cal.date.max().date()),
            "last_cal_label": str(cal.label_end.max().date()),
            "local_models": len(self.locals),
            "pooled_features": self.pooled.features,
            "local_features": {s: m.features for s, m in self.locals.items()},
            "tree_iterations": getattr(self.pooled, "iterations", None),
            "note": "All calibrator labels precede evaluation; underlying models retain the earlier core fit.",
        }
        return self

    def raw_predict(self, data):
        pooled = self.pooled.predict(data)
        local = np.full(len(data), np.nan)
        for symbol, model in self.locals.items():
            mask = (data.symbol == symbol).to_numpy()
            if mask.any():
                local[mask] = model.predict(data.loc[mask])
        weights = np.array(
            [
                min(0.75, self.counts.get(s, 0) / (self.counts.get(s, 0) + 504))
                for s in data.symbol
            ]
        )
        hybrid = pooled.copy()
        mask = np.isfinite(local)
        hybrid[mask] = (1 - weights[mask]) * pooled[mask] + weights[mask] * local[mask]
        return {"pooled": pooled, "per_stock": local, "hybrid": hybrid}

    def predict(self, data, arch):
        score = self.raw_predict(data)[arch]
        p = np.full(len(data), np.nan)
        mask = np.isfinite(score)
        if arch in self.calibrators:
            p[mask] = self.calibrators[arch].predict_proba(
                (score[mask] * 100).reshape(-1, 1)
            )[:, 1]
        else:
            p[mask] = self.priors[arch]
        out = data[
            ["date", "symbol", "target", "label_end", "entry_date", "eligible"]
        ].copy()
        out["score"] = score
        out["probability"] = p
        out["threshold"] = self.params["threshold"]
        return out
