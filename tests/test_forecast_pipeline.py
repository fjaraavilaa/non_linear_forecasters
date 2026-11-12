import numpy as np
import pandas as pd
import pytest
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

from src.data_preprocessor import DataPreprocessor
from src.calibrate_forecast import apply_variance_scale, fit_variance_scale
from src.deep_kernel import LstmExtractorSameSizeLayers
from src.forecast_evaluation import compare_forecasts, forecast_metrics
from src.gpytorch_model import DeepKernelSVGP, LatentSVGP
from src.help_functions.predictor import predict_with_trained_model, predictions_in_target_units
from src.training_functions import train_with_gp


def test_training_only_scaler_target_and_calendar_boundaries():
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    raw = pd.DataFrame({
        'Date': dates, 'Symbol': 'BTC', 'Close': np.arange(72, dtype=float),
        'returns': np.arange(72, dtype=float) / 100,
    })
    splits = {
        'train': {'start': '2024-01-01', 'end': '2024-01-01'},
        'validation': {'start': '2024-01-02', 'end': '2024-01-02'},
        'test': {'start': '2024-01-03', 'end': '2024-01-03'},
    }
    preprocessor = DataPreprocessor()
    scaled = preprocessor.scale_for_date_splits(raw, ['returns'], 'Close', splits['train'])
    assert list(preprocessor.scaler_used.feature_names_in_) == ['returns', 'Close']
    assert preprocessor.scaler_used.data_max_[1] == 23
    assert scaled['Close'].iloc[-1] > 1
    arrays = preprocessor.prepare_lstm_data(
        scaled.reset_index(), ['BTC'], sequence_length=2, target_col='Close',
        prediction_horizon=1, date_splits=splits, input_study_features=['returns'])
    assert [len(arrays[f'y_{s}']) for s in ('train', 'val', 'test')] == [22, 24, 24]
    assert pd.Timestamp(arrays['dates_val'][0]) == dates[24]
    assert pd.Timestamp(arrays['dates_val'][-1]) == dates[47]
    np.testing.assert_allclose(arrays['X_val'][0, :, 0], scaled['returns'].iloc[22:24])
    assert arrays['y_val'][0] == scaled['Close'].iloc[24]
    assert set(arrays['symbols_val']) == {'BTC'}


def test_sequences_do_not_cross_missing_hours():
    dates = pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00',
                            '2024-01-01 03:00', '2024-01-01 04:00',
                            '2024-01-01 05:00'])
    data = pd.DataFrame({'Date': dates, 'Symbol': 'BTC', 'Close': np.arange(5.)})
    x, y, target_dates = DataPreprocessor().create_sequences(
        data, 'BTC', sequence_length=2, target_col='Close',
        input_study_features=['Close'], expected_interval='1h')
    assert len(x) == 1
    assert y[0] == 4
    assert pd.Timestamp(target_dates[0]) == dates[-1]
    with pytest.raises(ValueError, match='Duplicate timestamps'):
        DataPreprocessor().create_sequences(
            pd.concat([data, data.iloc[-1:]]), 'BTC', sequence_length=2,
            target_col='Close', input_study_features=['Close'], expected_interval='1h')


def test_prediction_inverse_variance_and_invalid_variance():
    scaler = MinMaxScaler().fit(pd.DataFrame({'Close': [100., 300.]}))
    data = pd.DataFrame({'mean': [0.5], 'actual': [0.75], 'variance': [0.04]})
    result = predictions_in_target_units(data, scaler, 'Close')
    assert result.loc[0, 'mean'] == pytest.approx(200)
    assert result.loc[0, 'actual'] == pytest.approx(250)
    assert result.loc[0, 'variance'] == pytest.approx(1600)
    data.loc[0, 'variance'] = -1
    with pytest.raises(ValueError):
        predictions_in_target_units(data, scaler, 'Close')


def test_forecast_metrics():
    data = pd.DataFrame({'mean': [0., 0.], 'variance': [1., 1.], 'actual': [0., 2.]})
    metrics = forecast_metrics(data)
    assert metrics['mae'] == pytest.approx(1)
    assert metrics['rmse'] == pytest.approx(np.sqrt(2))
    assert metrics['coverage_80'] == pytest.approx(0.5)
    assert metrics['coverage_95'] == pytest.approx(0.5)
    assert metrics['gaussian_nll'] == pytest.approx(0.5 * np.log(2 * np.pi) + 1)
    candidate = data.copy()
    candidate.index = [1, 2]
    candidate.loc[1, 'actual'] = 2.
    comparison = compare_forecasts(data, candidate)
    assert comparison['common_count'] == 1
    assert comparison['baseline_only_count'] == comparison['candidate_only_count'] == 1
    candidate.loc[1, 'actual'] = 99
    with pytest.raises(ValueError, match='Actual values differ'):
        compare_forecasts(data, candidate)
    candidate.loc[1, 'actual'] = 2.
    data['symbol'] = 'BTC'
    candidate['symbol'] = 'ETH'
    with pytest.raises(ValueError, match='Symbols differ'):
        compare_forecasts(data, candidate)


def test_variance_calibration_is_validation_only():
    dates = pd.date_range('2024-01-01', periods=4, freq='h')
    predictions = pd.DataFrame({
        'mean': [0., 0., 0., 0.], 'variance': [4., 4., 4., 4.],
        'actual': [1., -1., 1., -1.],
    }, index=dates)
    calibration = fit_variance_scale(predictions.iloc[:2])
    assert calibration['variance_multiplier'] == pytest.approx(0.25)
    calibrated = apply_variance_scale(predictions.iloc[2:], calibration)
    np.testing.assert_allclose(calibrated['variance'], 1.)
    with pytest.raises(ValueError, match='after its fitting period'):
        apply_variance_scale(predictions, calibration)


def test_gp_train_and_reload(tmp_path):
    torch.manual_seed(1)
    x = torch.linspace(0, 1, 24).reshape(24, 1, 1).repeat(1, 3, 1)
    y = x[:, -1, 0]
    extractor = LstmExtractorSameSizeLayers(1, 3, 1, 1, 0.)
    gp = LatentSVGP(torch.randn(4, 3), RBFKernel(), use_natural_gradient=False)
    model = DeepKernelSVGP(extractor, gp)
    likelihood = GaussianLikelihood()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    path = tmp_path / 'model.pt'
    model, likelihood, metrics, best = train_with_gp(
        model, TensorDataset(x[:16], y[:16]), TensorDataset(x[16:], y[16:]),
        batch_size=8, epochs=2, likelihood=likelihood, optimizer=optimizer,
        patience=2, checkpoint_path=path)
    mean, variance = predict_with_trained_model(model, likelihood, x[16:], batch_size=4)
    assert torch.isfinite(mean).all() and (variance > 0).all()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    likelihood.load_state_dict(checkpoint['likelihood'])
    reloaded_mean, reloaded_variance = predict_with_trained_model(model, likelihood, x[16:])
    torch.testing.assert_close(mean, reloaded_mean)
    torch.testing.assert_close(variance, reloaded_variance)
    assert len(metrics['val_losses']) == 2
    assert best['epoch'] in (0, 1)


def test_cli_frozen_data_and_reload(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dates = pd.date_range('2024-01-01', periods=96, freq='h')
    close = 100 + np.arange(96) / 10
    raw = pd.DataFrame({
        'Date': dates, 'Symbol': 'BTC-USD', 'Open': close, 'High': close + 1,
        'Low': close - 1, 'Close': close, 'Volume': np.full(96, 100.),
    })
    raw_path = tmp_path / 'raw.parquet'
    raw.to_parquet(raw_path)
    date_config = tmp_path / 'dates.json'
    date_config.write_text(json.dumps({
        'train': {'start': '2024-01-01', 'end': '2024-01-02'},
        'validation': {'start': '2024-01-03', 'end': '2024-01-03'},
        'test': {'start': '2024-01-04', 'end': '2024-01-04'},
    }))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'lstm_shape_lag': 3, 'lstm_hidden_size': 3, 'batch_size': 16,
        'num_layers': 1, 'dropout': 0., 'natural_gradient_lr': 0.01,
        'starting_learning_rate': 0.01, 'max_epochs_per_cycle': 5,
        'inducing_points': 4,
    }))
    first, second = tmp_path / 'first', tmp_path / 'second'
    command = [
        sys.executable, str(root / 'main.py'), '--device', 'cpu', '--raw_data', str(raw_path),
        '--input_study_features', 'Close', '--config', str(config), '--dates-config', str(date_config),
        '--target-mode', 'delta', '--epochs', '1',
    ]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(command + ['--artifacts_dir', str(first)], cwd=root, env=env, check=True,
                   capture_output=True, text=True)
    subprocess.run(command + ['--artifacts_dir', str(second), '--model_location', str(first / 'checkpoint.pt')],
                   cwd=root, env=env, check=True, capture_output=True, text=True)
    for split in ('validation', 'test'):
        left = pd.read_parquet(first / f'predictions_{split}_native.parquet')
        right = pd.read_parquet(second / f'predictions_{split}_native.parquet')
        pd.testing.assert_frame_equal(left, right)
        expected = raw.set_index('Date').loc[left.index, 'Close'].to_numpy()
        np.testing.assert_allclose(left['actual'], expected)
        assert (left['variance'] > 0).all()
import json
import os
from pathlib import Path
import subprocess
import sys
