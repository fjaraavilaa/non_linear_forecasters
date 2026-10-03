import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.help_functions.predictor import compute_huber_delta


def calculate_huber_delta(
    predictions: pd.DataFrame,
    k: float = 1.345,
    metric: str = "mae",
) -> tuple[float, dict[str, float]]:
    """Calculate Huber delta using the project's existing helper function."""
    required_columns = {"Actual", "Prediction"}
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction file is missing required columns: {sorted(missing_columns)}"
        )

    values = predictions[["Actual", "Prediction"]].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("Prediction file contains no valid Actual/Prediction pairs")

    residuals = values["Actual"] - values["Prediction"]
    delta = compute_huber_delta(
        values,
        prediction_col="Prediction",
        actuals_col="Actual",
        k=k,
        metric=metric,
    )
    summary = {
        "metric": metric,
        "k": float(k),
        "sample_count": int(len(residuals)),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "huber_delta": float(delta),
    }
    return float(delta), summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Huber deltas from LSTM train, validation, and test predictions"
    )
    parser.add_argument(
        "--input_prefix",
        default="best_lstm_predictions",
        help="Prefix used by the train, validation, and test Parquet files",
    )
    parser.add_argument(
        "--output",
        default="lstm_huber_delta.json",
        help="JSON file for the calculated delta and residual summary",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "mse", "mad"],
        default="mae",
        help="Residual scale used by the existing project convention",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=1.345,
        help="Huber tuning constant",
    )
    args = parser.parse_args()

    summaries = {}
    for split in ("train", "validation", "test"):
        predictions_path = Path(f"{args.input_prefix}_{split}.parquet")
        predictions = pd.read_parquet(predictions_path)
        _, summary = calculate_huber_delta(
            predictions,
            k=args.k,
            metric=args.metric,
        )
        summaries[split] = summary

    with open(args.output, "w") as file:
        json.dump(summaries, file, indent=2)

    for split, summary in summaries.items():
        print(f"{split} Huber delta: {summary['huber_delta']:.10g}")
    print(f"Saved summary to: {args.output}")


if __name__ == "__main__":
    main()
