import logging
import sys
import pathlib as pl

import numpy as np

sys.path.append(pl.Path.cwd().parent.as_posix())

from src.binance_data_collector import BinanceDataCollector
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
    inverse_fractional_difference,
)

logging.basicConfig(level=logging.INFO)


def load_binance_close_prices():
    """Load a sufficiently long daily BTC-USD history for model validation."""
    data = BinanceDataCollector().fetch_crypto_data(
        symbols=["BTC-USD"],
        interval="1d",
        days_back=365,
    )
    close_prices = data.loc[data["Symbol"] == "BTC-USD", "Close"].dropna()

    if len(close_prices) < 300:
        logging.warning("Binance returned only %d daily closes", len(close_prices))
    if not np.isfinite(close_prices.to_numpy()).all():
        logging.warning("Binance returned non-finite close prices")
    if not (close_prices > 0).all():
        logging.warning("Binance returned non-positive close prices")
    return close_prices.to_numpy()


_close_prices = None
forecast_horizon = 3


def get_close_prices():
    global _close_prices
    if _close_prices is None:
        _close_prices = load_binance_close_prices()
    return _close_prices


def validate_forecast_shape(model):
    result = model.fit(get_close_prices()).forecast(forecast_horizon)

    if result.mean is None:
        logging.warning("Forecast mean is None")
    if result.variance is None:
        logging.warning("Forecast variance is None")
    if len(result.mean) != forecast_horizon:
        logging.warning("Mean length is %d; expected %d", len(result.mean), forecast_horizon)
    if len(result.variance) != forecast_horizon:
        logging.warning(
            "Variance length is %d; expected %d",
            len(result.variance),
            forecast_horizon,
        )
    if not np.isfinite(result.mean.to_numpy()).all():
        logging.warning("Forecast mean contains non-finite values")
    if not np.isfinite(result.variance.to_numpy()).all():
        logging.warning("Forecast variance contains non-finite values")

    logging.info(
        "%s forecast shape passed: mean=%s variance=%s",
        result.metadata["model"],
        result.mean.shape,
        result.variance.shape,
    )
    return result


def validate_decision_signals(model):
    result = model.fit(get_close_prices()).forecast(forecast_horizon)
    signals = result.decision_signals()
    required_columns = ["forecast_mean", "forecast_variance", "forecast_std", "lower_bound", 
                        "upper_bound", "current_value", "expected_change", "expected_return", 
                        "direction", "signal_strength", "target", "model", "policy"]
    missing_columns = [column for column in required_columns if column not in signals]
    if missing_columns:
        logging.warning("Decision signals are missing columns: %s", missing_columns)
    if len(signals) != forecast_horizon:
        logging.warning("Signal length is %d; expected %d", len(signals), forecast_horizon)
    if set(signals["policy"]) != {"signals_only"}:
        logging.warning("Models must not emit actions or position sizes")
    if not set(signals["direction"]).issubset({"up", "down", "flat"}):
        logging.warning("Direction uses non-standard labels")

    logging.info("Decision signals passed for %s", result.metadata["model"])
    return signals


def test_moving_average():
    return validate_forecast_shape(MovingAverage(window=12))


def test_exponential_smoothing():
    return validate_forecast_shape(ExponentialSmoothingModel())


def test_arima():
    return validate_forecast_shape(ARIMA(order=(1, 1, 0)))


def test_sarima():
    return validate_forecast_shape(
        SARIMA(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
    )


def test_arfima():
    return validate_forecast_shape(ARFIMA(d=0.4, order=(1, 0, 0)))


def test_sarfima():
    return validate_forecast_shape(
        SARFIMA(
            d=0.4,
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
        )
    )


def test_garch():
    return validate_forecast_shape(GARCH(p=1, q=1))


def test_aparch():
    return validate_forecast_shape(APARCH(p=1, o=1, q=1))


def test_decision_signal_interface():
    for model in (
        MovingAverage(window=12),
        ARIMA(order=(1, 1, 0)),
        GARCH(p=1, q=1),
    ):
        validate_decision_signals(model)


def test_fractional_difference_round_trip():
    values = np.linspace(100.0, 130.0, 100)
    fractional_values = fractional_difference(values, d=0.4)
    warmup = len(values) - len(fractional_values)

    reconstructed = inverse_fractional_difference(
        fractional_values,
        history=values[:warmup],
        d=0.4,
    )

    np.testing.assert_allclose(
        reconstructed.to_numpy(),
        values[warmup:],
        rtol=0.0,
        atol=1e-10,
    )


test_moving_average()
test_exponential_smoothing()
test_arima()
test_sarima()
test_arfima()
test_sarfima()
test_garch()
test_aparch()
test_decision_signal_interface()
test_fractional_difference_round_trip()
