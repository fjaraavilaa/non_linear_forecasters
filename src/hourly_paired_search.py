"""Paired hourly feature search with an inner stopping period."""

import json
import logging
import math
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from src.data_preprocessor import DataPreprocessor, calendar_end
from src.help_functions.predictor import (
    create_df_from_predictions, hourly_price_predictions, predict_with_trained_model,
)
from src.hourly_gp_model import build_hourly_gp
from src.training_functions import train_with_gp

FEATURES = ['Log_Returns', 'Close', 'Volume', 'HL_Ratio']
BASELINE = FEATURES[:2]
INNER_TRAIN_END = '2024-11-30'
INNER_VAL_START = '2024-12-01'
MAX_LAG = 96
PILOT_PARAMS = [
    dict(lstm_shape_lag=24, lstm_hidden_size=8, batch_size=512, num_layers=1,
         dropout=0.1, natural_gradient_lr=0.005, starting_learning_rate=0.001, inducing_points=64),
    dict(lstm_shape_lag=58, lstm_hidden_size=10, batch_size=1516, num_layers=2,
         dropout=0.2, natural_gradient_lr=0.01, starting_learning_rate=0.003, inducing_points=128),
    dict(lstm_shape_lag=96, lstm_hidden_size=16, batch_size=1516, num_layers=2,
         dropout=0.25, natural_gradient_lr=0.03, starting_learning_rate=0.005, inducing_points=365),
]


def adam_schedule(optimizer, mode, horizon=300, floor=0.1):
    if mode == 'constant':
        return None
    if mode != 'cosine':
        raise ValueError(f'Unknown scheduler: {mode}')
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: floor + (1 - floor) *
        (1 + math.cos(math.pi * min(epoch, horizon) / horizon)) / 2)


def prepare_data(path, symbol, dates, inner):
    raw = pd.read_parquet(path)
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.loc[(raw['Symbol'] == symbol) &
                  (raw['Date'] >= pd.Timestamp(dates['train']['start'])) &
                  (raw['Date'] <= calendar_end(dates['validation']['end']))].copy()
    if raw.empty:
        raise ValueError(f'No hourly data for {symbol} in the configured dates')
    raw = raw.sort_values(['Symbol', 'Date'])
    if raw.duplicated(['Symbol', 'Date']).any():
        raise ValueError('Duplicate symbol/timestamp rows in hourly data')
    preprocessor = DataPreprocessor()
    prepared = preprocessor.prepare_features(raw, target_col='Log_Returns', drop_bad_values=False)
    prepared = preprocessor.remove_gap_returns(prepared, '1h')
    train_dates = dict(dates['train'])
    validation_dates = dict(dates['validation'])
    if inner:
        train_dates['end'] = INNER_TRAIN_END
        validation_dates = {'start': INNER_VAL_START, 'end': dates['train']['end']}
    scaled = preprocessor.scale_for_date_splits(
        prepared.set_index(['Date', 'Symbol']), FEATURES, train_dates).dropna()
    data = preprocessor.prepare_lstm_data(
        scaled.reset_index(), [symbol], sequence_length=MAX_LAG,
        target_col='Log_Returns', prediction_horizon=1,
        date_splits={'train': train_dates, 'validation': validation_dates, 'test': dates['test']},
        input_study_features=FEATURES, expected_interval='1h', warn_if_unscaled=False)
    if not len(data['y_val']):
        raise ValueError('No valid validation sequences')
    return data, preprocessor.scaler_used


def inputs_for(data, split, features, lag):
    indices = [FEATURES.index(feature) for feature in features]
    return np.asarray(data['X_' + split][:, -lag:, :][:, :, indices], dtype=np.float32)


def prices_for(mean, variance, data, scaler, features, inputs):
    predictions = create_df_from_predictions(mean, variance, data['dates_val'], data['y_val'])
    return hourly_price_predictions(predictions, inputs, features, scaler, 'Log_Returns')


def fit(data, scaler, features, params, schedule, seed, device, epochs, path, select):
    torch.manual_seed(seed)
    x_train = inputs_for(data, 'train', features, params['lstm_shape_lag'])
    x_val = inputs_for(data, 'val', features, params['lstm_shape_lag']) if select else None
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train),
        torch.from_numpy(np.asarray(data['y_train'], dtype=np.float32)))
    val = (torch.utils.data.TensorDataset(torch.from_numpy(x_val),
        torch.from_numpy(np.asarray(data['y_val'], dtype=np.float32))) if select else None)
    model, likelihood, adam, natural = build_hourly_gp(x_train, params, device, seed)

    def price_mae(mean):
        prices = prices_for(mean.numpy(), np.zeros(len(mean)), data, scaler, features, x_val)
        return float(np.abs(prices['mean'] - prices['actual']).mean())

    model, likelihood, metrics, best = train_with_gp(
        model, train, val, params['batch_size'], epochs, likelihood, adam,
        natural_gradient_optimizer=natural, learning_scheduler=adam_schedule(adam, schedule),
        min_delta=0.0, patience=100 if select else None, min_epochs=150,
        monitor='val' if select else 'train', validation_metric=price_mae if select else None,
        checkpoint_path=str(path.with_suffix('.pt')), restore_best=select,
        device=torch.device(device))
    with path.with_suffix('.json').open('x') as file:
        json.dump({'metrics': metrics, 'best': best}, file)
    return model, likelihood, best, metrics


def evaluate(model, likelihood, data, scaler, features, params, path):
    inputs = inputs_for(data, 'val', features, params['lstm_shape_lag'])
    mean, variance = predict_with_trained_model(model, likelihood, torch.from_numpy(inputs))
    mean, variance = mean.numpy(), variance.numpy()
    prices = prices_for(mean, variance, data, scaler, features, inputs)
    prices['symbol'] = data['symbols_val']
    prices.to_parquet(path)
    scale = abs(scaler.scale_[list(scaler.feature_names_in_).index('Log_Returns')])
    target_error = (mean - data['y_val']) / scale
    return_variance = variance / scale ** 2
    if not np.all(np.isfinite(return_variance) & (return_variance > 0)):
        raise FloatingPointError('Non-positive or non-finite predictive variance')
    nll = (0.5 * (np.log(2 * np.pi * return_variance) + target_error ** 2 / return_variance)
           + np.log(prices['actual'].to_numpy()))
    sections = []
    for indices in np.array_split(np.arange(len(prices)), 3):
        part = prices.iloc[indices]
        if part.empty:
            raise ValueError('Not enough predictions for three validation sections')
        sections.append({'start': str(part.index.min()), 'end': str(part.index.max()),
            'price_mae': float(np.abs(part['mean'] - part['actual']).mean()),
            'persistence_mae': float(np.abs(part['baseline'] - part['actual']).mean())})
    return {
        'price_mae': float(np.abs(prices['mean'] - prices['actual']).mean()),
        'persistence_mae': float(np.abs(prices['baseline'] - prices['actual']).mean()),
        'raw_return_mae': float(np.abs(target_error).mean()),
        'coverage_95': float(((prices['actual'] >= prices['lower_95']) &
                              (prices['actual'] <= prices['upper_95'])).mean()),
        'price_nll': float(np.mean(nll)), 'sections': sections, 'observations': len(prices),
    }


def suggest(trial):
    return {
        'lstm_shape_lag': trial.suggest_categorical('lstm_shape_lag', [24, 58, 96]),
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [8, 10, 16]),
        'batch_size': trial.suggest_categorical('batch_size', [512, 1516]),
        'num_layers': trial.suggest_int('num_layers', 1, 2),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'natural_gradient_lr': trial.suggest_float('natural_gradient_lr', 0.005, 0.05, log=True),
        'starting_learning_rate': trial.suggest_float('starting_learning_rate', 0.001, 0.005, log=True),
        'inducing_points': trial.suggest_categorical('inducing_points', [64, 128, 365]),
    }


def run_paired_search(args, dates):
    if not args.raw_data or len(args.symbols) != 1 or args.target != 'Log_Returns':
        raise ValueError('Paired search requires --raw_data, one symbol, and --target Log_Returns')
    if args.pretrained_mean_function:
        raise ValueError('Paired search does not use a pretrained mean')
    if args.epochs < 150:
        raise ValueError('Paired search requires at least 150 epochs')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    inner, inner_scaler = prepare_data(args.raw_data, args.symbols[0], dates, inner=True)
    full, full_scaler = prepare_data(args.raw_data, args.symbols[0], dates, inner=False)
    pilot = []
    for number, params in enumerate(PILOT_PARAMS):
        for schedule in ('constant', 'cosine'):
            logging.info('Scheduler pilot %s: %s', number, schedule)
            _, _, best, metrics = fit(inner, inner_scaler, FEATURES, params, schedule,
                args.seed, args.device, args.epochs, output / f'pilot_{number}_{schedule}', True)
            pilot.append({'configuration': number, 'schedule': schedule, 'params': params,
                          'best_epoch': best['epoch'] + 1,
                          'price_mae': best['best_selection_metric'],
                          'stopping_reason': metrics['stopping_reason']})
    schedule = min(('constant', 'cosine'),
                   key=lambda mode: np.mean([p['price_mae'] for p in pilot if p['schedule'] == mode]))
    with (output / 'pilot_summary.json').open('x') as file:
        json.dump({'results': pilot, 'selected_scheduler': schedule}, file, indent=2)
    logging.info('Selected scheduler: %s', schedule)
    study = optuna.create_study(study_name=args.study_name, storage=args.storage,
        direction='minimize', load_if_exists=False, sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.set_user_attr('experiment', {'dates': dates, 'inner_train_end': INNER_TRAIN_END,
        'inner_validation_start': INNER_VAL_START, 'features': FEATURES, 'symbol': args.symbols[0],
        'raw_data': str(args.raw_data.resolve()), 'epochs': args.epochs, 'seed': args.seed,
        'scheduler': schedule, 'max_lag_for_alignment': MAX_LAG})

    def objective(trial):
        params = suggest(trial)
        results = {}
        for label, features in [('baseline', BASELINE), ('augmented', FEATURES)]:
            prefix = output / f'trial_{trial.number}_{label}'
            _, _, best, metrics = fit(inner, inner_scaler, features, params, schedule,
                args.seed, args.device, args.epochs, Path(str(prefix) + '_inner'), True)
            model, likelihood, _, _ = fit(full, full_scaler, features, params, schedule,
                args.seed, args.device, best['epoch'] + 1, Path(str(prefix) + '_refit'), False)
            result = evaluate(model, likelihood, full, full_scaler, features, params,
                              Path(str(prefix) + '_predictions.parquet'))
            result.update({'best_epoch': best['epoch'] + 1,
                           'inner_price_mae': best['best_selection_metric'],
                           'stopping_reason': metrics['stopping_reason']})
            results[label] = result
            trial.set_user_attr(label, result)
        return results['augmented']['price_mae']

    study.optimize(objective, n_trials=args.trials)
    logging.info('Best augmented validation MAE: %.6f; parameters: %s',
                 study.best_value, study.best_params)
