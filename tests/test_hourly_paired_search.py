import numpy as np
import pandas as pd
import pytest
import torch

from src.hourly_paired_search import adam_schedule, evaluate, fit, inputs_for, prepare_data


def test_cosine_schedule_has_floor_and_does_not_change_other_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    adam = torch.optim.Adam([parameter], lr=0.01)
    other = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.2)
    scheduler = adam_schedule(adam, 'cosine')
    rates = [adam.param_groups[0]['lr']]
    for _ in range(400):
        adam.step()
        scheduler.step()
        rates.append(adam.param_groups[0]['lr'])
    assert rates[0] == pytest.approx(0.01)
    assert rates[150] == pytest.approx(0.0055)
    assert rates[300] == pytest.approx(0.001)
    assert rates[400] == pytest.approx(0.001)
    assert other.param_groups[0]['lr'] == 0.2
    assert adam_schedule(adam, 'constant') is None


def test_inner_scaling_alignment_and_test_exclusion(tmp_path):
    dates = pd.date_range('2024-11-20', '2025-03-04 23:00', freq='h')
    close = 100.0 + np.arange(len(dates)) / 100
    raw = pd.DataFrame({'Date': dates, 'Symbol': 'BTC-USD', 'Open': close,
                        'High': close + 1, 'Low': close - 1, 'Close': close,
                        'Volume': np.arange(len(dates)) + 1.0})
    path = tmp_path / 'data.parquet'
    raw.to_parquet(path)
    split = {'train': {'start': '2024-11-20', 'end': '2025-03-01'},
             'validation': {'start': '2025-03-02', 'end': '2025-03-03'},
             'test': {'start': '2025-03-04', 'end': '2025-03-04'}}
    inner, scaler = prepare_data(path, 'BTC-USD', split, inner=True)
    full, _ = prepare_data(path, 'BTC-USD', split, inner=False)
    assert scaler.data_max_[list(scaler.feature_names_in_).index('Close')] == pytest.approx(
        raw.loc[raw.Date < pd.Timestamp('2024-12-01'), 'Close'].max())
    assert pd.to_datetime(inner['dates_val']).min() == pd.Timestamp('2024-12-01')
    assert pd.to_datetime(inner['dates_val']).max() == pd.Timestamp('2025-03-01 23:00')
    assert pd.to_datetime(full['dates_val']).min() == pd.Timestamp('2025-03-02')
    assert len(full['dates_test']) == 0
    baseline = inputs_for(full, 'val', ['Log_Returns', 'Close'], 24)
    augmented = inputs_for(full, 'val', ['Log_Returns', 'Close', 'Volume', 'HL_Ratio'], 96)
    assert baseline.shape[0] == augmented.shape[0]
    np.testing.assert_array_equal(baseline, augmented[:, -24:, :2])
    for split_name in ('train', 'val'):
        for key in ('X', 'y', 'dates', 'symbols'):
            name = f'{key}_{split_name}'
            full[name] = full[name][:12]
    parameters = dict(lstm_shape_lag=24, lstm_hidden_size=4, batch_size=6, num_layers=1,
                      dropout=0., natural_gradient_lr=0.005, starting_learning_rate=0.001,
                      inducing_points=4)
    model, likelihood, best, history = fit(
        full, scaler, ['Log_Returns', 'Close'], parameters, 'cosine',
        42, 'cpu', 2, tmp_path / 'inner', True)
    assert 1 <= best['epoch'] + 1 <= 2
    assert len(history['selection_metrics']) == 2
    model, likelihood, _, _ = fit(
        full, scaler, ['Log_Returns', 'Close'], parameters, 'cosine',
        42, 'cpu', 2, tmp_path / 'refit', False)
    scores = evaluate(model, likelihood, full, scaler, ['Log_Returns', 'Close'],
                      parameters, tmp_path / 'predictions.parquet')
    assert scores['observations'] == 12
    assert np.isfinite(scores['price_mae'])
    assert np.isfinite(scores['price_nll'])
