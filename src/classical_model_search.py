"""Assumption-aware Optuna search for the classical forecasting models.

The objective is rolling-origin MAE.  A candidate is pruned when the training
transformation is not stationary, or when fitted residuals retain material
serial correlation.  Volatility models additionally require no remaining ARCH
effect in standardized residuals.  These checks deliberately happen inside
each training fold, so a configuration cannot pass by relying on future data.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.classical_models import (
    APARCH,
    ARFIMA,
    ARIMA,
    GARCH,
    SARFIMA,
    SARIMA,
    ExponentialSmoothingModel,
    MovingAverage,
    fractional_difference,
)


@dataclass(frozen=True)
class AssumptionReport:
    """Result of the diagnostics used to accept a search candidate."""

    passed: bool
    checks: Dict[str, float | bool | str]


def _finite_series(values: Sequence[float]) -> pd.Series:
    series = pd.Series(values, dtype="float64").dropna().reset_index(drop=True)
    if len(series) < 24 or not np.isfinite(series.to_numpy()).all():
        raise ValueError("Search requires at least 24 finite observations")
    return series


def _adf_pvalue(values: pd.Series) -> float:
    from statsmodels.tsa.stattools import adfuller

    # ADF can use only a small number of lags on short rolling folds.
    max_lag = min(10, max(0, len(values) // 5 - 1))
    return float(adfuller(values, maxlag=max_lag, autolag="AIC")[1])


def _ljung_box_pvalue(residuals: np.ndarray) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 12:
        return 0.0
    lag = min(10, len(residuals) // 5)
    return float(acorr_ljungbox(residuals, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])


def _arch_lm_pvalue(residuals: np.ndarray) -> float:
    from statsmodels.stats.diagnostic import het_arch

    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 20:
        return 0.0
    lags = min(10, len(residuals) // 5)
    return float(het_arch(residuals, nlags=lags)[1])


def check_assumptions(model: Any, values: Sequence[float], alpha: float = 0.05) -> AssumptionReport:
    """Fit ``model`` and test the assumptions relevant to its family.

    The ADF test is applied to returns for volatility models, to the configured
    differenced/fractionally-differenced series for ARIMA-family models, and is
    not imposed on level/trend smoothing models.  A non-significant Ljung-Box
    test is required of all fitted residuals.  GARCH/APARCH also require a
    non-significant ARCH-LM test on standardized residuals.
    """
    series = _finite_series(values)
    checks: Dict[str, float | bool | str] = {}
    is_volatility = isinstance(model, (GARCH, APARCH))
    is_fractional = isinstance(model, (ARFIMA, SARFIMA))
    is_arima = isinstance(model, (ARIMA, SARIMA))

    if is_volatility:
        if (series <= 0).any():
            return AssumptionReport(False, {"positive_prices": False})
        transformed = 100 * np.log(series / series.shift(1)).dropna()
    elif is_fractional:
        transformed = fractional_difference(series, model.d)
    elif is_arima:
        d = model.order[1]
        transformed = series.diff(d).dropna() if d else series
        seasonal_d = model.seasonal_order[1]
        seasonal_period = model.seasonal_order[3]
        if seasonal_d and seasonal_period:
            transformed = transformed.diff(seasonal_period * seasonal_d).dropna()
    else:
        transformed = pd.Series(dtype="float64")

    if is_volatility or is_fractional or is_arima:
        if len(transformed) < 20:
            return AssumptionReport(False, {"enough_transformed_data": False})
        try:
            checks["stationarity_adf_pvalue"] = _adf_pvalue(transformed)
        except (ValueError, np.linalg.LinAlgError):
            return AssumptionReport(False, {"stationarity_test": "failed"})
        if checks["stationarity_adf_pvalue"] >= alpha:
            return AssumptionReport(False, checks)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(series)
    except (ValueError, np.linalg.LinAlgError, OverflowError):
        return AssumptionReport(False, {**checks, "fit": "failed"})

    if isinstance(model, MovingAverage):
        # This baseline has no likelihood fit object; residuals around its
        # training level are the relevant whiteness diagnostic.
        residuals = (series - model._level).to_numpy(dtype="float64")
    else:
        fitted = model._fit
        residuals = np.asarray(
            getattr(fitted, "std_resid", getattr(fitted, "resid", [])), dtype="float64"
        )
    checks["residual_ljung_box_pvalue"] = _ljung_box_pvalue(residuals)
    if checks["residual_ljung_box_pvalue"] < alpha:
        return AssumptionReport(False, checks)

    if is_volatility:
        checks["standardized_residual_arch_lm_pvalue"] = _arch_lm_pvalue(residuals)
        if checks["standardized_residual_arch_lm_pvalue"] < alpha:
            return AssumptionReport(False, checks)
    return AssumptionReport(True, checks)


def build_model(model_name: str, params: Dict[str, Any]) -> Any:
    """Instantiate one supported classical model from Optuna parameters."""
    factories: Dict[str, Callable[..., Any]] = {
        "moving_average": MovingAverage,
        "exponential_smoothing": ExponentialSmoothingModel,
        "arima": ARIMA,
        "sarima": SARIMA,
        "arfima": ARFIMA,
        "sarfima": SARFIMA,
        "garch": GARCH,
        "aparch": APARCH,
    }
    try:
        return factories[model_name](**params)
    except KeyError as error:
        raise ValueError(f"Unknown classical model: {model_name}") from error


def suggest_parameters(trial: Any, model_name: str, seasonal_period: int = 0) -> Dict[str, Any]:
    """Define deliberately compact, statistically valid search spaces."""
    if model_name == "moving_average":
        return {"window": trial.suggest_int("window", 2, 60)}
    if model_name == "exponential_smoothing":
        trend = trial.suggest_categorical("trend", [None, "add"])
        seasonal = trial.suggest_categorical("seasonal", [None, "add"])
        if seasonal and seasonal_period < 2:
            raise ValueError("seasonal_period must be at least 2 for seasonal smoothing")
        return {"trend": trend, "seasonal": seasonal, "seasonal_periods": seasonal_period if seasonal else None}

    if model_name in {"arima", "sarima"}:
        order = (
            trial.suggest_int("p", 0, 3),
            trial.suggest_int("d", 0, 2),
            trial.suggest_int("q", 0, 3),
        )
        if model_name == "arima":
            return {"order": order}
        if seasonal_period < 2:
            raise ValueError("SARIMA search requires a seasonal_period of at least 2")
        return {
            "order": order,
            "seasonal_order": (
                trial.suggest_int("P", 0, 2),
                trial.suggest_int("D", 0, 1),
                trial.suggest_int("Q", 0, 2),
                seasonal_period,
            ),
        }
    if model_name in {"arfima", "sarfima"}:
        params: Dict[str, Any] = {
            "d": trial.suggest_float("fractional_d", 0.05, 0.95, step=0.05),
            # Fractional differencing already handles integration; do not
            # introduce a second integer differencing operation.
            "order": (trial.suggest_int("p", 0, 3), 0, trial.suggest_int("q", 0, 3)),
        }
        if model_name == "sarfima":
            if seasonal_period < 2:
                raise ValueError("SARFIMA search requires a seasonal_period of at least 2")
            params["seasonal_order"] = (
                trial.suggest_int("P", 0, 2), 0, trial.suggest_int("Q", 0, 2), seasonal_period
            )
        return params
    if model_name == "garch":
        return {"p": trial.suggest_int("p", 1, 3), "q": trial.suggest_int("q", 1, 3),
                "distribution": trial.suggest_categorical("distribution", ["t", "ged", "normal"])}
    if model_name == "aparch":
        p = trial.suggest_int("p", 1, 3)
        return {"p": p, "o": trial.suggest_int("o", 1, 3), "q": trial.suggest_int("q", 1, 3),
                "distribution": trial.suggest_categorical("distribution", ["t", "ged", "normal"])}
    raise ValueError(f"Unknown classical model: {model_name}")


def rolling_origins(n_observations: int, horizon: int, folds: int) -> Iterable[int]:
    if horizon < 1 or folds < 1:
        raise ValueError("horizon and folds must be positive")
    first_end = n_observations - folds * horizon
    if first_end < max(24, horizon * 3):
        raise ValueError("Not enough observations for the requested rolling validation")
    return range(first_end, n_observations - horizon + 1, horizon)


def search_classical_model(
    values: Sequence[float],
    model_name: str,
    *,
    horizon: int = 1,
    folds: int = 3,
    trials: int = 50,
    seasonal_period: int = 0,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
) -> Any:
    """Run an assumption-aware, expanding-window Optuna search.

    The returned study stores each accepted trial's diagnostics in user
    attributes and minimizes mean absolute error across chronological folds.
    """
    import optuna

    series = _finite_series(values)
    origins = list(rolling_origins(len(series), horizon, folds))

    def objective(trial: Any) -> float:
        params = suggest_parameters(trial, model_name, seasonal_period)
        # Optuna requires a fixed distribution for each parameter name.  Keep
        # the o range fixed, then prune combinations invalid for APARCH.
        if model_name == "aparch" and params["o"] > params["p"]:
            trial.set_user_attr("assumption_failure", {"aparch_order": "o exceeds p"})
            raise optuna.TrialPruned("invalid APARCH order")
        errors = []
        reports = []
        for origin in origins:
            train = series.iloc[:origin]
            report = check_assumptions(build_model(model_name, params), train)
            reports.append(report.checks)
            if not report.passed:
                trial.set_user_attr("assumption_failure", report.checks)
                raise optuna.TrialPruned("model assumptions failed")
            model = build_model(model_name, params).fit(train)
            prediction = model.forecast(horizon).mean.to_numpy(dtype="float64")
            actual = series.iloc[origin : origin + horizon].to_numpy(dtype="float64")
            if len(prediction) != len(actual) or not np.isfinite(prediction).all():
                raise optuna.TrialPruned("invalid forecast")
            errors.extend(np.abs(prediction - actual))
        trial.set_user_attr("assumption_reports", reports)
        return float(np.mean(errors))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=trials)
    return study


def _main() -> None:
    parser = argparse.ArgumentParser(description="Search classical forecasting model parameters")
    parser.add_argument("csv", type=Path, help="CSV containing the historical series")
    parser.add_argument("--column", default="Close")
    parser.add_argument("--model", choices=["moving_average", "exponential_smoothing", "arima", "sarima", "arfima", "sarfima", "garch", "aparch"], required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seasonal-period", type=int, default=0)
    parser.add_argument("--storage", help="Optuna storage URL, e.g. sqlite:///classical.db")
    parser.add_argument("--study-name")
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    if args.column not in data:
        raise ValueError(f"Column {args.column!r} was not found in {args.csv}")
    study = search_classical_model(
        data[args.column], args.model, horizon=args.horizon, folds=args.folds,
        trials=args.trials, seasonal_period=args.seasonal_period,
        storage=args.storage, study_name=args.study_name,
    )
    print({"best_value_mae": study.best_value, "best_params": study.best_params})


if __name__ == "__main__":
    _main()
