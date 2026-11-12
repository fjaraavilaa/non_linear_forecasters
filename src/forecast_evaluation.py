"""Metrics for forecasts of observed values in the target's native units."""

import numpy as np
from scipy.stats import norm


def forecast_metrics(predictions):
    required = {'mean', 'variance', 'actual'}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    values = predictions[['mean', 'variance', 'actual']].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Predictions must be nonempty and finite")
    mean, variance, actual = values.T
    if (variance <= 0).any():
        raise ValueError("Gaussian predictive variances must be positive")
    residual = actual - mean
    std = np.sqrt(variance)
    result = {
        'count': len(values),
        'mae': float(np.mean(np.abs(residual))),
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'gaussian_nll': float(np.mean(0.5 * (np.log(2 * np.pi * variance) + residual ** 2 / variance))),
    }
    for level in (80, 95):
        half_width = norm.ppf((1 + level / 100) / 2) * std
        result[f'coverage_{level}'] = float(np.mean(np.abs(residual) <= half_width))
        result[f'width_{level}'] = float(np.mean(2 * half_width))
    result['coverage_error'] = (
        abs(result['coverage_80'] - 0.80) + abs(result['coverage_95'] - 0.95)
    ) / 2
    return result


def compare_forecasts(baseline, candidate):
    """Compare native-unit forecasts at common, unambiguous target timestamps."""
    if not baseline.index.is_unique or not candidate.index.is_unique:
        raise ValueError("Prediction timestamps must be unique for comparison")
    common = baseline.index.intersection(candidate.index).sort_values()
    if common.empty:
        raise ValueError("Forecasts have no common target timestamps")
    left = baseline.loc[common]
    right = candidate.loc[common]
    if 'symbol' in left.columns and 'symbol' in right.columns:
        if not (left['symbol'].to_numpy() == right['symbol'].to_numpy()).all():
            raise ValueError("Symbols differ at common target timestamps")
    if not np.allclose(left['actual'], right['actual'], rtol=1e-5, atol=1e-4):
        raise ValueError("Actual values differ at common target timestamps")
    return {
        'common_count': len(common),
        'baseline_only_count': len(baseline) - len(common),
        'candidate_only_count': len(candidate) - len(common),
        'baseline': forecast_metrics(left),
        'candidate': forecast_metrics(right),
    }
