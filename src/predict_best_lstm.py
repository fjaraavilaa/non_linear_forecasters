import argparse
import json
import logging

import pandas as pd
import torch

from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.help_functions.predictor import (
    create_df_from_predictions,
    predict_with_torch_model,
)
from src.lstm_model import LSTMGeneralized_Output_SameSize


logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict with the LSTM trained using the best hyperparameters"
    )
    parser.add_argument(
        "--model",
        default="trained_best_lstm.pt",
        help="Path to the trained LSTM checkpoint",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC-USD"],
        help="List of symbols to fetch data for",
    )
    parser.add_argument(
        "--input_study_features",
        nargs="+",
        default=None,
        help="Features used during training; defaults to checkpoint metadata",
    )
    parser.add_argument(
        "--model_target",
        default="Log_Returns",
        help="Target column used to build the prediction split",
    )
    parser.add_argument(
        "--output_prefix",
        default="best_lstm_predictions",
        help="Prefix for the train, validation, and test Parquet files",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    model_config = checkpoint["model_config"]
    input_features = args.input_study_features or checkpoint.get(
        "input_features", ["Returns"]
    )

    with open("validation/healthy_dates.json", "r") as file:
        config_dates = json.load(file)

    collector = BinanceDataCollector()
    crypto_raw_data = collector.fetch_crypto_data(
        symbols=args.symbols,
        interval="5m",
        start_date=config_dates["train"]["start"],
        end_date=config_dates["test"]["end"],
    )

    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_features(
        crypto_raw_data,
        target_col=args.model_target,
        drop_bad_values=False,
    )
    processed_data = preprocessor.scale_features(
        processed_data.set_index(["Date", "Symbol"])[input_features].dropna(),
        scaler="minmax",
    )

    lstm_data = preprocessor.prepare_lstm_data(
        processed_data.reset_index(),
        symbols=args.symbols,
        sequence_length=model_config["lstm_shape_lag"],
        target_col=args.model_target,
        prediction_horizon=1,
        date_splits={
            "train": config_dates["train"],
            "validation": config_dates["validation"],
            "test": config_dates["test"],
        },
        input_study_features=input_features,
    )

    train_inputs = torch.tensor(lstm_data["X_train"], dtype=torch.float32)
    model = LSTMGeneralized_Output_SameSize(
        input_size=train_inputs.shape[2],
        hidden_size=model_config["lstm_hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        output_size=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    split_output_names = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }
    for split, output_name in split_output_names.items():
        split_inputs = torch.tensor(
            lstm_data[f"X_{split}"], dtype=torch.float32
        )
        mean, variance = predict_with_torch_model(model, split_inputs)
        split_predictions = create_df_from_predictions(
            mean.cpu().numpy().reshape(-1),
            variance.cpu().numpy().reshape(-1),
            pd.to_datetime(lstm_data[f"dates_{split}"]),
        )
        split_predictions.index.name = "Date"
        split_predictions["Actual"] = lstm_data[f"y_{split}"].reshape(-1)
        split_predictions["Split"] = split
        split_predictions = split_predictions.rename(
            columns={"mean": "Prediction", "variance": "Variance"}
        )
        split_predictions = split_predictions[
            ["Split", "Actual", "Prediction", "Variance"]
        ]
        output_path = f"{args.output_prefix}_{output_name}.parquet"
        split_predictions.to_parquet(output_path)
        logging.info(
            "Predicted %d %s observations and saved them to %s",
            len(split_predictions),
            output_name,
            output_path,
        )
        print(split_predictions.head())


if __name__ == "__main__":
    main()
