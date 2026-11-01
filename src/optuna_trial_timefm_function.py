"""
Hyperparameter search for TimesFM using Optuna.
"""
import argparse
import json

import numpy as np
import optuna
import pandas as pd

from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.timefm_functions import forecast, forecast_with_covariates, set_timefm_model


with open("validation/healthy_dates.json", "r") as file:
    config_dates = json.load(file)

parser = argparse.ArgumentParser(
    description="Arguments for the TimesFM hyperparameter search"
)
parser.add_argument(
    "--symbols", nargs="+", default=config_dates["symbols"]
)
parser.add_argument("--interval", type=str, default="1h")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument(
    "--covariates",
    nargs="*",
    default=[],
    help="Historical feature columns to use as TimesFM numerical covariates",
)

SYMBOLS = parser.parse_args().symbols
INTERVAL = parser.parse_args().interval
DEVICE = parser.parse_args().device
COVARIATES = parser.parse_args().covariates

DATES_START_STUDY = config_dates["train"]["start"]
DATES_END_STUDY = config_dates["test"]["end"]
VALIDATION_END = pd.Timestamp(config_dates["validation"]["end"])
TIMESFM_CONTEXT_LIMIT = 16384


def objective(trial):
    version = trial.suggest_categorical("version", ["2.5", "3.0"])
    context_length = trial.suggest_categorical(
        "context_length", [512, 1024, 2048, 4096, 8192, 16384]
    )
    horizon = trial.suggest_categorical(
        "horizon", [1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    if context_length + horizon > TIMESFM_CONTEXT_LIMIT:
        raise optuna.TrialPruned(
            f"context_length={context_length} and horizon={horizon} "
            f"exceed the TimesFM limit of {TIMESFM_CONTEXT_LIMIT}"
        )

    collector = BinanceDataCollector()
    crypto_raw_data = collector.fetch_crypto_data(
        symbols=SYMBOLS,
        interval=INTERVAL,
        start_date=DATES_START_STUDY,
        end_date=DATES_END_STUDY,
    )
    crypto_raw_data["Date"] = pd.to_datetime(crypto_raw_data["Date"])
    crypto_raw_data = crypto_raw_data.sort_values(["Symbol", "Date"])
    if COVARIATES:
        crypto_raw_data = DataPreprocessor().prepare_features(
            crypto_raw_data,
            target_col="Log_Returns",
            drop_bad_values=False,
        )
        missing_covariates = [
            column for column in COVARIATES
            if column not in crypto_raw_data.columns
        ]
        if missing_covariates:
            raise ValueError(f"Missing covariate columns: {missing_covariates}")

    model = set_timefm_model(version, context_length, horizon)
    errors = []

    for symbol in SYMBOLS:
        symbol_data = crypto_raw_data[crypto_raw_data["Symbol"] == symbol]
        if COVARIATES:
            symbol_data = symbol_data.dropna(
                subset=["Close", *COVARIATES]
            )
        history = symbol_data.loc[
            symbol_data["Date"] <= VALIDATION_END, "Close"
        ].to_numpy(dtype=np.float32)
        actual = symbol_data.loc[
            symbol_data["Date"] > VALIDATION_END, "Close"
        ].to_numpy(dtype=np.float64)[:horizon]

        if len(history) < context_length or len(actual) < horizon:
            raise ValueError(
                f"Insufficient data for {symbol} with context={context_length} "
                f"and horizon={horizon}"
            )

        input_history = history[-context_length:]
        if COVARIATES:
            context_data = symbol_data.loc[
                symbol_data["Date"] <= VALIDATION_END
            ].tail(context_length)
            dynamic_covariates = {
                column: np.concatenate([
                    context_data[column].to_numpy(dtype=np.float32),
                    np.repeat(
                        context_data[column].iloc[-1], horizon
                    ).astype(np.float32),
                ])
                for column in COVARIATES
            }
            prediction = forecast_with_covariates(
                model,
                [input_history],
                dynamic_covariates,
            )
        else:
            prediction = forecast(model, input_history, horizon)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        elif hasattr(prediction, "forecast"):
            prediction = prediction.forecast
        elif hasattr(prediction, "point_forecast"):
            prediction = prediction.point_forecast

        prediction = np.atleast_1d(
            np.asarray(prediction, dtype=np.float64).squeeze()
        )
        if prediction.size < horizon:
            raise ValueError(
                f"Model returned {prediction.size} predictions for "
                f"horizon={horizon}"
            )
        errors.extend(np.abs(prediction[:horizon] - actual))

    return float(np.mean(errors))


storage_name = "sqlite:///mydb_timefm_5min.db"

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="timefm_forecast_search",
        storage=storage_name,
    )
    study.optimize(objective, n_trials=40)
