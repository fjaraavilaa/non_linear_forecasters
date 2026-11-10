"""Classical forecasting models with a shared fit/forecast interface."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Decision-agent-ready forecast without prescribing an action."""

    mean: pd.Series
    variance: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_value: Optional[float] = None
    target: str = "Close"

    def decision_signals(self, z_score: float = 1.96) -> pd.DataFrame:
        """Return standardized signals for a separate decision policy.

        The output describes expected movement and uncertainty. It deliberately
        does not emit buy/sell actions or position sizes.
        """
        if self.current_value is None:
            raise ValueError("current_value is required to calculate decision signals")
        if z_score <= 0:
            raise ValueError("z_score must be positive")
        mean = self.mean.astype("float64")
        variance = self.variance.astype("float64").clip(lower=0)
        standard_deviation = np.sqrt(variance)
        expected_change = mean - self.current_value
        expected_return = expected_change / self.current_value
        signal_strength = expected_change.abs() / standard_deviation.replace(0, np.nan)
        signal_strength = signal_strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame(
            {
                "forecast_mean": mean,
                "forecast_variance": variance,
                "forecast_std": standard_deviation,
                "lower_bound": mean - z_score * standard_deviation,
                "upper_bound": mean + z_score * standard_deviation,
                "current_value": self.current_value,
                "expected_change": expected_change,
                "expected_return": expected_return,
                "direction": np.select(
                    [expected_change > 0, expected_change < 0],
                    ["up", "down"],
                    default="flat",
                ),
                "signal_strength": signal_strength,
            },
            index=mean.index,
        ).assign(
            target=self.target,
            model=self.metadata.get("model", "unknown"),
            policy="signals_only",
        )


def _as_series(values: Sequence[float], name: str = "Close") -> pd.Series:
    series = pd.Series(values, dtype="float64", name=name).dropna()
    if series.empty:
        raise ValueError("At least one finite training value is required")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Training values must be finite")
    return series


class MovingAverage:
    """Rolling moving-average baseline."""

    def __init__(self, window: int = 24):
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._level: Optional[float] = None
        self._current_value: Optional[float] = None

    def fit(self, values: Sequence[float]) -> "MovingAverage":
        series = _as_series(values)
        self._level = float(series.tail(self.window).mean())
        self._current_value = float(series.iloc[-1])
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if self._level is None:
            raise RuntimeError("Call fit before forecast")
        index = pd.RangeIndex(horizon)
        return ForecastResult(
            mean=pd.Series(self._level, index=index, name="mean"),
            variance=pd.Series(0.0, index=index, name="variance"),
            metadata={"model": "moving_average", "window": self.window},
            current_value=self._current_value,
        )


class ExponentialSmoothingModel:
    """Holt-Winters exponential smoothing model."""

    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ):
        self.options = {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
        }
        self._fit = None
        self._current_value: Optional[float] = None

    def fit(self, values: Sequence[float]) -> "ExponentialSmoothingModel":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series = _as_series(values)
        self._fit = ExponentialSmoothing(series, **self.options).fit()
        self._current_value = float(series.iloc[-1])
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if self._fit is None:
            raise RuntimeError("Call fit before forecast")
        mean = pd.Series(self._fit.forecast(horizon), name="mean")
        return ForecastResult(
            mean=mean,
            variance=pd.Series(0.0, index=mean.index, name="variance"),
            metadata={"model": "exponential_smoothing", **self.options},
            current_value=self._current_value,
        )


class _SARIMAXModel:
    def __init__(
        self,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        model_name: str = "arima",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_name = model_name
        self._fit = None
        self._current_value: Optional[float] = None

    def fit(self, values: Sequence[float]) -> "_SARIMAXModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = _as_series(values)
        self._fit = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="n",
            # A search should not be allowed to select an explosive ARMA
            # process merely because it happened to score well on one split.
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
        self._current_value = float(series.iloc[-1])
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if self._fit is None:
            raise RuntimeError("Call fit before forecast")
        prediction = self._fit.get_forecast(horizon)
        mean = pd.Series(prediction.predicted_mean, name="mean")
        variance = pd.Series(prediction.var_pred_mean, index=mean.index, name="variance")
        return ForecastResult(
            mean=mean,
            variance=variance,
            metadata={
                "model": self.model_name,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
            },
            current_value=self._current_value,
        )


class ARIMA(_SARIMAXModel):
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__(order, model_name="arima")


class SARIMA(_SARIMAXModel):
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24),
    ):
        super().__init__(order, seasonal_order, model_name="sarima")


def _fractional_weights(length: int, d: float, threshold: float) -> np.ndarray:
    if length < 1:
        return np.array([], dtype="float64")
    if d < 0:
        raise ValueError("d must be non-negative")
    weights = [1.0]
    for _ in range(1, length):
        weight = -weights[-1] * (d - len(weights) + 1) / len(weights)
        if abs(weight) < threshold:
            break
        weights.append(weight)
    return np.asarray(weights, dtype="float64")


def fractional_difference(
    values: Sequence[float], d: float, threshold: float = 1e-4
) -> pd.Series:
    """Apply fractional differencing and drop the warm-up observations."""
    if d < 0:
        raise ValueError("d must be non-negative")
    series = _as_series(values)
    weights = _fractional_weights(len(series), d, threshold)
    output = np.full(len(series), np.nan)
    values_array = series.to_numpy()
    window = len(weights)
    for index in range(window - 1, len(series)):
        output[index] = np.dot(
            weights, values_array[index - window + 1 : index + 1][::-1]
        )
    return pd.Series(output, name=series.name).dropna().reset_index(drop=True)


def inverse_fractional_difference(
    values: Sequence[float],
    history: Sequence[float],
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Recover original-scale values from fractional-difference values.

    ``history`` must contain the original-scale observations immediately before
    the first value in ``values``. Each recovered value is appended to that
    history before reconstructing the next one.
    """
    differenced = _as_series(values, name="fractional_difference")
    original_history = _as_series(history, name="history")
    weights = _fractional_weights(
        len(original_history) + len(differenced), d, threshold
    )
    reconstructed = list(original_history.to_numpy())

    for differenced_value in differenced.to_numpy():
        lag_count = min(len(weights) - 1, len(reconstructed))
        lag_values = reconstructed[-lag_count:][::-1] if lag_count else []
        lag_weights = weights[1 : lag_count + 1]
        reconstructed.append(
            float(differenced_value)
            - float(np.dot(lag_weights, lag_values))
        )

    return pd.Series(
        reconstructed[-len(differenced) :],
        index=differenced.index,
        name="value",
    )


class _FractionalARIMA:
    def __init__(
        self,
        d: float = 0.4,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        model_name: str = "arfima",
    ):
        self.d = d
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_name = model_name
        self._history: Optional[pd.Series] = None
        self._fit = None

    def fit(self, values: Sequence[float]) -> "_FractionalARIMA":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._history = _as_series(values)
        differenced = fractional_difference(self._history, self.d)
        self._fit = SARIMAX(
            differenced,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if self._fit is None or self._history is None:
            raise RuntimeError("Call fit before forecast")
        prediction = self._fit.get_forecast(horizon)
        differenced_forecast = np.asarray(prediction.predicted_mean)
        mean = inverse_fractional_difference(
            differenced_forecast, self._history, self.d
        )
        mean.name = "mean"
        variance = pd.Series(np.asarray(prediction.var_pred_mean), index=mean.index, name="variance")
        return ForecastResult(
            mean=mean,
            variance=variance,
            metadata={
                "model": self.model_name,
                "fractional_d": self.d,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
            },
            current_value=float(self._history.iloc[-1]),
        )


class ARFIMA(_FractionalARIMA):
    pass


class SARFIMA(_FractionalARIMA):
    def __init__(
        self,
        d: float = 0.4,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24),
    ):
        super().__init__(d, order, seasonal_order, model_name="sarfima")


class VolatilityModel:
    """GARCH or APARCH return model with forecasts converted to Close prices."""

    def __init__(
        self,
        volatility: str = "GARCH",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        distribution: str = "t",
    ):
        if volatility not in {"GARCH", "APARCH"}:
            raise ValueError("volatility must be 'GARCH' or 'APARCH'")
        if p < 1 or q < 1 or o < 0:
            raise ValueError("p and q must be positive and o must be non-negative")
        if volatility == "APARCH" and o > p:
            raise ValueError("APARCH asymmetry order o cannot exceed p")
        if distribution not in {"normal", "t", "skewt", "ged"}:
            raise ValueError("Unsupported innovation distribution")
        self.volatility = volatility
        self.p, self.o, self.q = p, o, q
        self.distribution = distribution
        self._fit = None
        self._last_close: Optional[float] = None

    def fit(self, values: Sequence[float]) -> "VolatilityModel":
        from arch import arch_model

        series = _as_series(values)
        if (series <= 0).any():
            raise ValueError("Volatility models require strictly positive Close prices")
        returns = 100 * np.log(series / series.shift(1)).dropna()
        self._last_close = float(series.iloc[-1])
        self._fit = arch_model(
            returns,
            mean="Constant",
            vol=self.volatility,
            p=self.p,
            o=self.o if self.volatility == "APARCH" else 0,
            q=self.q,
            dist=self.distribution,
        ).fit(disp="off")
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if self._fit is None or self._last_close is None:
            raise RuntimeError("Call fit before forecast")
        method = "simulation" if self.volatility == "APARCH" and horizon > 1 else "analytic"
        forecast = self._fit.forecast(
            horizon=horizon,
            method=method,
            simulations=1000 if method == "simulation" else 1000,
            reindex=False,
        )
        mean_returns = np.asarray(forecast.mean.iloc[-1]) / 100
        variances = np.asarray(forecast.variance.iloc[-1]) / 10000
        mean = self._last_close * np.exp(np.cumsum(mean_returns))
        return ForecastResult(
            mean=pd.Series(mean, name="mean"),
            variance=pd.Series(variances * mean**2, name="variance"),
            metadata={
                "model": self.volatility.lower(),
                "p": self.p,
                "o": self.o,
                "q": self.q,
                "distribution": self.distribution,
                "target": "Close",
            },
            current_value=self._last_close,
            target="Close",
        )


class GARCH(VolatilityModel):
    def __init__(self, p: int = 1, q: int = 1, distribution: str = "t"):
        super().__init__("GARCH", p=p, q=q, distribution=distribution)


class APARCH(VolatilityModel):
    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        distribution: str = "t",
    ):
        super().__init__("APARCH", p=p, o=o, q=q, distribution=distribution)


MODEL_REGISTRY = {
    "arima": ARIMA,
    "sarima": SARIMA,
    "arfima": ARFIMA,
    "sarfima": SARFIMA,
    "garch": GARCH,
    "aparch": APARCH,
    "moving_average": MovingAverage,
    "exponential_smoothing": ExponentialSmoothingModel,
}
