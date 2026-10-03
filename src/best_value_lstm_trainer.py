import argparse
import json
import logging

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.lstm_model import LSTMGeneralized_Output_SameSize
from src.training_functions import train_lstm_model


logging.basicConfig(level=logging.INFO)


def main():
	parser = argparse.ArgumentParser(
		description="Train an LSTM with the best saved hyperparameters"
	)
	parser.add_argument("--symbols", nargs="+", default=["BTC-USD"], help="List of symbols to fetch data for",)
	parser.add_argument("--input_study_features", nargs="+", default=["Returns"], help="Features to use as LSTM inputs",)
	parser.add_argument("--model_output",default="trained_best_lstm.pt",help="Path where the trained model checkpoint is saved",)
	parser.add_argument("--model_target", default = "Log_Returns", help = "Target column for the LSTM model")
	parser.add_argument("--metrics_output", default="best_lstm_metrics.json", help="Path where training metrics are saved",)
	args = parser.parse_args()

	with open("validation/healthy_dates.json", "r") as file:
		config_dates = json.load(file)

	with open("lstm_only_best_params.json", "r") as file:
		best_params = json.load(file)

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
		processed_data.set_index(["Date", "Symbol"])[args.input_study_features].dropna(),
		scaler="minmax",
	)

	lstm_data = preprocessor.prepare_lstm_data(
		processed_data.reset_index(),
		symbols=args.symbols,
		sequence_length=best_params["lstm_shape_lag"],
		target_col=args.model_target,
		prediction_horizon=1,
		date_splits={
			"train": config_dates["train"],
			"validation": config_dates["validation"],
			"test": config_dates["test"],
		},
		input_study_features=args.input_study_features,
	)

	logging.info("LSTM training data shape: %s", lstm_data["X_train"].shape)

	train_data = torch.utils.data.TensorDataset(
		torch.tensor(lstm_data["X_train"], dtype=torch.float32),
		torch.tensor(lstm_data["y_train"], dtype=torch.float32),
	)
	validation_data = torch.utils.data.TensorDataset(
		torch.tensor(lstm_data["X_val"], dtype=torch.float32),
		torch.tensor(lstm_data["y_val"], dtype=torch.float32),
	)

	model = LSTMGeneralized_Output_SameSize(
		input_size=lstm_data["X_train"].shape[2],
		hidden_size=best_params["lstm_hidden_size"],
		num_layers=best_params["num_layers"],
		dropout=best_params["dropout"],
		output_size=1,
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	model, metrics, best_metrics = train_lstm_model(
		model=model,
		optimizing_criterion=torch.nn.L1Loss(),
		train_data=train_data,
		val_data=validation_data,
		optimizer=optimizer,
		epochs=1000,
		patience=10,
		min_delta=1e-5,
		batch_size=best_params["batch_size"],
		optimizer_metric="mae",
		give_model_metrics=True,
	)

	model.eval()
	with torch.no_grad():
		test_predictions = model(
			torch.tensor(lstm_data["X_test"], dtype=torch.float32)
		).cpu().numpy().reshape(-1)

	test_actuals = lstm_data["y_test"].reshape(-1)
	test_predictions_df = pd.DataFrame(
		{
			"Date": pd.to_datetime(lstm_data["dates_test"]),
			"Actual": test_actuals,
			"Prediction": test_predictions,
		}
	)
	test_mse = mean_squared_error(
		test_predictions_df["Actual"], test_predictions_df["Prediction"]
	)
	test_mae = mean_absolute_error(
		test_predictions_df["Actual"], test_predictions_df["Prediction"]
	)
	test_predictions_df.to_csv("best_lstm_test_predictions.csv", index=False)
	metrics["sklearn_test_mse"] = test_mse
	metrics["sklearn_test_mae"] = test_mae
	logging.info("Sklearn test MSE: %.6f, MAE: %.6f", test_mse, test_mae)

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"model_config": best_params,
			"input_features": args.input_study_features,
			"best_metrics": best_metrics,
		},
		args.model_output,
	)

	with open(args.metrics_output, "w") as file:
		json.dump(metrics, file)

	logging.info("Model saved to %s", args.model_output)
	logging.info("Best metrics: %s", best_metrics)


if __name__ == "__main__":
	main()
