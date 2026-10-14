"""Rate-of-penetration modeling.

Models ROP as a function of the controllable drilling parameters - weight on
bit, rotary speed and flow rate - optionally with formation/context features
(depth, inclination, bit size) that the driller cannot change.

Two families are provided:

* ``power_law`` - a Bingham-type multiplicative model
  ``ROP = K * WOB^a * RPM^b * Q^c`` fitted by least squares in log space. The
  exponents are directly interpretable (``a`` close to 1 means ROP still
  responds linearly to weight, i.e. no founder yet).
* Machine-learning regressors (polynomial ridge, random forest, gradient
  boosting) that capture founder points and parameter interactions.

Drilling data is strongly autocorrelated with depth, so random K-fold splits
leak information between neighbouring feet and overstate skill. Models are
evaluated with contiguous depth blocks instead (:func:`blocked_cv`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CONTROL_FEATURES = ["wob", "rpm", "flow"]
CONTEXT_FEATURES = ["depth", "inc", "bit_size", "rop_trail", "mse_trail"]


def mechanical_specific_energy(wob_klbf, rpm, torque_ftlbf, rop_fthr, bit_size_in):
    """Teale (1965) mechanical specific energy, psi.

    MSE = 4 W / (pi d^2) + 480 N T / (d^2 ROP)

    with W in lbf, N in rev/min, T in ft-lbf, d in inches and ROP in ft/hr.
    MSE approaching the rock's confined compressive strength indicates
    efficient drilling; a rising MSE at constant lithology flags dysfunction
    (bit balling, whirl, stick-slip, founder).
    """
    wob = np.asarray(wob_klbf, dtype=float) * 1000.0
    d2 = np.asarray(bit_size_in, dtype=float) ** 2
    rop = np.maximum(np.asarray(rop_fthr, dtype=float), 1e-3)
    return 4.0 * wob / (np.pi * d2) + 480.0 * np.asarray(rpm) * np.asarray(torque_ftlbf) / (d2 * rop)


def add_drilling_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Add MSE (kpsi) and trailing drillability features.

    ``rop_trail`` / ``mse_trail`` are the median ROP and MSE over the previous
    ``window`` depth bins, shifted by one bin so a row never sees its own
    target. They act as a real-time proxy for rock strength: a formation
    change (e.g. FORGE's basin-fill to granite contact) shows up in them
    within a few tens of feet, which surface parameters alone cannot capture.
    """
    df = df.sort_values("depth").copy()
    if {"torque", "bit_size"} <= set(df.columns):
        df["mse"] = mechanical_specific_energy(
            df["wob"], df["rpm"], df["torque"], df["rop"], df["bit_size"]) / 1000.0
        df["mse_trail"] = df["mse"].shift(1).rolling(window, min_periods=5).median()
    df["rop_trail"] = df["rop"].shift(1).rolling(window, min_periods=5).median()
    return df


class PowerLawROP(BaseEstimator, RegressorMixin):
    """ROP = K * prod(x_i ^ a_i), fitted by OLS on log-transformed data."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.lr_ = LinearRegression().fit(np.log(np.clip(X, 1e-6, None)), np.log(y))
        self.exponents_ = self.lr_.coef_
        self.K_ = float(np.exp(self.lr_.intercept_))
        # Smearing estimate (Duan 1983) corrects the bias of back-transforming
        # a log-space mean.
        resid = np.log(y) - self.lr_.predict(np.log(np.clip(X, 1e-6, None)))
        self.smearing_ = float(np.mean(np.exp(resid)))
        return self

    def predict(self, X):
        X = np.log(np.clip(np.asarray(X, dtype=float), 1e-6, None))
        return np.exp(self.lr_.predict(X)) * self.smearing_


MODEL_NAMES = {
    "power_law": "Power law (Bingham-type)",
    "poly_ridge": "Polynomial ridge (deg 2)",
    "random_forest": "Random forest",
    "gbr": "Gradient boosting",
}


def make_model(kind: str, random_state: int = 0):
    if kind == "power_law":
        return PowerLawROP()
    if kind == "poly_ridge":
        return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0))
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=random_state
        )
    if kind == "gbr":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=random_state,
        )
    raise ValueError(f"unknown model kind {kind!r}; choose from {list(MODEL_NAMES)}")


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-6)) * 100),
    }


def depth_blocks(depth: np.ndarray, n_blocks: int) -> np.ndarray:
    """Assign each sample to one of ``n_blocks`` contiguous depth intervals."""
    order = np.argsort(np.argsort(depth))
    return (order * n_blocks // len(depth)).astype(int)


def blocked_cv(model, X: pd.DataFrame, y: pd.Series, depth, n_blocks: int = 5) -> pd.DataFrame:
    """Leave-one-depth-block-out cross-validation."""
    blocks = depth_blocks(np.asarray(depth), n_blocks)
    rows = []
    for b in range(n_blocks):
        test = blocks == b
        m = clone(model).fit(X[~test], y[~test])
        d = np.asarray(depth)[test]
        rows.append({"block": b, "depth_from": d.min(), "depth_to": d.max(),
                     "n": int(test.sum()), **metrics(y[test], m.predict(X[test]))})
    return pd.DataFrame(rows)


@dataclass
class ROPModel:
    """A fitted ROP model with its feature list and hold-out diagnostics."""

    kind: str
    features: list[str]
    estimator: object = None
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    cv: pd.DataFrame | None = None
    test_frame: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return MODEL_NAMES[self.kind]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(df[self.features])

    def importance(self) -> pd.Series | None:
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=self.features).sort_values()
        if isinstance(est, PowerLawROP):
            return pd.Series(est.exponents_, index=self.features)
        return None


def clean_for_model(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df.dropna(subset=features + ["rop"]).query("rop > 0")


def fit_rop_model(
    df: pd.DataFrame,
    kind: str = "gbr",
    features: list[str] | None = None,
    test_fraction: float = 0.2,
    n_blocks: int = 5,
    run_cv: bool = True,
) -> ROPModel:
    """Fit an ROP model, holding out the deepest ``test_fraction`` of the data.

    Holding out the bottom of the hole mimics the real use case: a model
    trained on what has been drilled so far, predicting what is ahead.
    """
    features = list(features or CONTROL_FEATURES)
    data = clean_for_model(df, features).sort_values("depth")
    split = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:split], data.iloc[split:]

    est = make_model(kind).fit(train[features], train["rop"])
    result = ROPModel(kind=kind, features=features, estimator=est)
    result.train_metrics = metrics(train["rop"], est.predict(train[features]))
    if len(test):
        pred = est.predict(test[features])
        result.test_metrics = metrics(test["rop"], pred)
        result.test_frame = test.assign(rop_pred=pred)
    if run_cv:
        result.cv = blocked_cv(make_model(kind), data[features], data["rop"],
                               data["depth"], n_blocks)
    return result


def compare_models(df: pd.DataFrame, features: list[str] | None = None,
                   kinds: list[str] | None = None, n_blocks: int = 5) -> pd.DataFrame:
    """Blocked-CV comparison table across model kinds."""
    features = list(features or CONTROL_FEATURES)
    data = clean_for_model(df, features).sort_values("depth")
    rows = []
    for kind in kinds or list(MODEL_NAMES):
        cv = blocked_cv(make_model(kind), data[features], data["rop"], data["depth"], n_blocks)
        rows.append({"model": MODEL_NAMES[kind], **cv[["r2", "rmse", "mae", "mape"]].mean().to_dict()})
    return pd.DataFrame(rows).sort_values("rmse")
