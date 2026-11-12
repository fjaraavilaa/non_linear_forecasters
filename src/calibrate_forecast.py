"""Fit and apply a validation-only multiplicative Gaussian variance calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_evaluation import forecast_metrics


def fit_variance_scale(predictions):
    forecast_metrics(predictions)
    if not isinstance(predictions.index, pd.DatetimeIndex) or not predictions.index.is_unique:
        raise ValueError("Calibration requires unique datetime target timestamps")
    residual = predictions['actual'].to_numpy() - predictions['mean'].to_numpy()
    scale = float(np.mean(residual ** 2 / predictions['variance'].to_numpy()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration variance multiplier must be finite and positive")
    return {
        'method': 'gaussian_variance_scale',
        'variance_multiplier': scale,
        'fit_start': predictions.index.min().isoformat(),
        'fit_end': predictions.index.max().isoformat(),
        'fit_count': len(predictions),
    }


def apply_variance_scale(predictions, calibration):
    if calibration['method'] != 'gaussian_variance_scale':
        raise ValueError("Unknown variance calibration method")
    if not isinstance(predictions.index, pd.DatetimeIndex) or not predictions.index.is_unique:
        raise ValueError("Calibration requires unique datetime target timestamps")
    if predictions.empty or predictions.index.min() <= pd.Timestamp(calibration['fit_end']):
        raise ValueError("Calibration can only be applied after its fitting period")
    scale = calibration['variance_multiplier']
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration variance multiplier must be finite and positive")
    forecast_metrics(predictions)
    result = predictions.copy()
    result['variance'] *= scale
    forecast_metrics(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('validation', type=Path)
    fit.add_argument('--end', required=True, help='Last target timestamp permitted for fitting')
    fit.add_argument('--output', type=Path, required=True)
    apply = commands.add_parser('apply')
    apply.add_argument('predictions', type=Path)
    apply.add_argument('--calibration', type=Path, required=True)
    apply.add_argument('--output', type=Path, required=True)
    apply.add_argument('--after', help='Only evaluate target timestamps after this time')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.command == 'fit':
        predictions = pd.read_parquet(args.validation)
        if not isinstance(predictions.index, pd.DatetimeIndex):
            raise ValueError("Validation predictions need a datetime index")
        calibration = fit_variance_scale(predictions.loc[predictions.index <= pd.Timestamp(args.end)])
        calibration['source_sha256'] = hashlib.sha256(args.validation.read_bytes()).hexdigest()
        with args.output.open('x') as file:
            json.dump(calibration, file, indent=2)
    else:
        calibration = json.loads(args.calibration.read_text())
        predictions = pd.read_parquet(args.predictions)
        if args.after:
            predictions = predictions.loc[predictions.index > pd.Timestamp(args.after)]
        result = apply_variance_scale(predictions, calibration)
        result.to_parquet(args.output)


if __name__ == '__main__':
    main()
