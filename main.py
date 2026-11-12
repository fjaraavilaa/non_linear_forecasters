import logging
import torch
from gpytorch.kernels import MaternKernel, PeriodicKernel
from src.gpytorch_model import LatentSVGP, DeepKernelSVGP
from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor, calendar_bounds
from src.deep_kernel import LstmExtractorSameSizeLayers
from src.training_functions import train_with_gp
from src.help_functions.predictor import (
    predict_with_trained_model, create_df_from_predictions, predictions_in_target_units,
)
from src.lstm_model import LSTMGeneralized_Output_SameSize
from src.mean_functions import LSTMPreTrainedMeanFunction
import argparse
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import optim
import json
import joblib
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def save_pipeline_artifacts(artifacts_dir, raw_data, engineered_data, scaled_data, lstm_data, metadata):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_data.to_parquet(artifacts_dir / 'raw_data.parquet', index=False)
    engineered_data.to_parquet(artifacts_dir / 'engineered_features.parquet', index=False)
    scaled_data.reset_index().to_parquet(artifacts_dir / 'scaled_features.parquet', index=False)

    for split in ('train', 'val', 'test'):
        np.savez_compressed(
            artifacts_dir / f'lstm_{split}.npz',
            X=lstm_data[f'X_{split}'],
            y=lstm_data[f'y_{split}'],
            dates=lstm_data[f'dates_{split}'],
            symbols=lstm_data[f'symbols_{split}'],
        )

    with open(artifacts_dir / 'metadata.json', 'w') as file:
        json.dump(metadata, file, indent=2, default=str)

def main():
    parser = argparse.ArgumentParser(
        description = "hard-coded arguments for the lstm+kernel"
    )

    parser.add_argument('--symbols', nargs='+', default=['BTC-USD'], help='List of symbols to fetch data for')
    parser.add_argument('--input_study_features', nargs='+', default=None, help='List of features to use in the study')
    parser.add_argument('--device' , type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--work_with_validation_set', action=argparse.BooleanOptionalAction, default=True,
                        help='Use validation-based early stopping (default); disable only for legacy training')
    parser.add_argument('--model_location', type=str, default=None, help='Location for the model')
    parser.add_argument('--pretrained_mean_function', type=str, default=None, help='Pretrained LSTM checkpoint')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts',
                        help='Directory for production comparison artifacts')
    parser.add_argument('--raw_data', type=Path, help='Use a frozen raw-data parquet instead of downloading')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=Path, default=Path('best_trial_params_1h.json'),
                        help='JSON model hyperparameters')
    parser.add_argument('--dates-config', type=Path, default=Path('validation/healthy_dates.json'),
                        help='JSON calendar split configuration')
    parser.add_argument('--validation-only', action='store_true',
                        help='Do not predict the test split during model selection')
    parser.add_argument('--target-mode', choices=['level', 'delta'], default='level',
                        help='Model the scaled Close level or its change from the last observed Close')
    args = parser.parse_args()
    with open(args.config) as file:
        config_best_values = json.load(file)
    with open(args.dates_config) as file:
        config_dates = json.load(file)
    if args.validation_only and not args.work_with_validation_set:
        raise ValueError("--validation-only requires validation-based training")
    dates_start_study = config_dates['train']['start']
    _, test_end = calendar_bounds(config_dates['test'])
    dates_end_study = (test_end - pd.Timedelta(milliseconds=1)).isoformat()
    artifacts_dir = Path(args.artifacts_dir)
    if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        raise FileExistsError(f"Artifact directory is not empty: {artifacts_dir}; choose a new directory")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    SYMBOLS = args.symbols
    DEVICE = torch.device(args.device)
    TRAIN_W_VALIDATION = args.work_with_validation_set

    mean_function_configs = None
    mean_config = None
    if args.pretrained_mean_function is not None:
        mean_function_configs = torch.load(args.pretrained_mean_function, map_location=DEVICE)
        mean_config = mean_function_configs['model_config']
    INPUT_STUDY_FEATURES = args.input_study_features or (
        mean_function_configs['input_features']
        if mean_function_configs is not None
        else ['Log_Returns']
    )
    if args.target_mode == 'delta' and 'Close' not in INPUT_STUDY_FEATURES:
        raise ValueError("--target-mode delta requires Close among the input features")
    if args.target_mode == 'delta' and args.pretrained_mean_function:
        raise ValueError("A pretrained level mean cannot be used with --target-mode delta")

    logging.info(f"Using the following input features for the study: {INPUT_STUDY_FEATURES}")
    
    interval = '1h'
    if args.raw_data is not None:
        crypto_raw_data = pd.read_parquet(args.raw_data)
        crypto_raw_data = crypto_raw_data[crypto_raw_data['Symbol'].isin(SYMBOLS)].copy()
        if set(crypto_raw_data['Symbol'].unique()) != set(SYMBOLS):
            raise ValueError("Frozen data does not contain all requested symbols")
    else:
        collector = BinanceDataCollector()
        crypto_raw_data = collector.fetch_crypto_data(symbols = SYMBOLS,
                                                      interval = interval,
                                                      start_date = dates_start_study,
                                                      end_date = dates_end_study)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_features(crypto_raw_data, target_col = 'Close', drop_bad_values = False)
    logging.info('Features Prepared')
    engineered_data = processed_data.copy()

    processed_data = preprocessor.scale_for_date_splits(
        processed_data, INPUT_STUDY_FEATURES, 'Close', config_dates['train'])
    
    logging.info('Features Scaled')

    lstm_data = preprocessor.prepare_lstm_data(processed_data.reset_index(), symbols = SYMBOLS,
                                               sequence_length = config_best_values['lstm_shape_lag'],
                                               target_col = 'Close',
                                               prediction_horizon=1,
                                               date_splits =  {
                                                   'train': config_dates['train'],
                                                   'validation': config_dates['validation'],
                                                   'test': config_dates['test']
                                                   },
                                               input_study_features = INPUT_STUDY_FEATURES,
                                               expected_interval = interval
                                               )      

    save_pipeline_artifacts(
        args.artifacts_dir,
        crypto_raw_data,
        engineered_data,
        processed_data,
        lstm_data,
        {
            'symbols': SYMBOLS,
            'interval': interval,
            'input_features': INPUT_STUDY_FEATURES,
            'target_column': 'Close',
            'sequence_length': config_best_values['lstm_shape_lag'],
            'prediction_horizon': 1,
            'date_splits': config_dates,
            'split_shapes': {
                split: {
                    'X': list(lstm_data[f'X_{split}'].shape),
                    'y': list(lstm_data[f'y_{split}'].shape),
                }
                for split in ('train', 'val', 'test')
            },
            'scaler': type(preprocessor.scaler_used).__name__,
            'scaler_fit_split': 'train',
            'prediction_units': {'scaled': 'MinMaxScaler target', 'native': 'Close price'},
            'variance_semantics': 'predictive variance including Gaussian observation noise',
            'seed': args.seed,
            'target_mode': args.target_mode,
            'sequence_interval': interval,
            'model_config': config_best_values,
            'raw_data_sha256': hashlib.sha256(args.raw_data.read_bytes()).hexdigest() if args.raw_data else None,
            'pretrained_mean_sha256': hashlib.sha256(Path(args.pretrained_mean_function).read_bytes()).hexdigest() if args.pretrained_mean_function else None,
        },
    )
    joblib.dump(preprocessor.scaler_used, Path(args.artifacts_dir) / 'scaler.joblib')


    logging.info(f'lstm matrixes done. Check Dimensions: {str(lstm_data["X_train"].shape)}')

    logging.info('LSTM Data Prepared')

    if args.target_mode == 'delta':
        close_index = INPUT_STUDY_FEATURES.index('Close')
        for split in ('train', 'val', 'test'):
            lstm_data[f'y_{split}'] = (
                lstm_data[f'y_{split}'] - lstm_data[f'X_{split}'][:, -1, close_index]
            )

    def build_model():
        mean_function = None
        if mean_function_configs is not None:
            pretrained_lstm = LSTMGeneralized_Output_SameSize(
                input_size = lstm_data['X_train'].shape[2],
                hidden_size = mean_config['lstm_hidden_size'],
                num_layers = mean_config['num_layers'],
                dropout = mean_config['dropout'],
                output_size = 1,
            )
            pretrained_lstm.load_state_dict(mean_function_configs['model_state_dict'])
            pretrained_lstm.eval()
            mean_function = LSTMPreTrainedMeanFunction(pretrained_lstm, DEVICE)

        feature_extractor = LstmExtractorSameSizeLayers(
            input_size=lstm_data['X_train'].shape[2],
            hidden_size=config_best_values['lstm_hidden_size'],
            num_layers=config_best_values['num_layers'],
            dropout=config_best_values['dropout'],
            output_size=1,
        ).to(DEVICE)
        inducing_indices = np.random.choice(len(lstm_data['X_train']),
                                            config_best_values['inducing_points'],
                                            replace=len(lstm_data['X_train']) < config_best_values['inducing_points'])
        inducing_sequences = torch.from_numpy(lstm_data['X_train'][inducing_indices]).float().to(DEVICE)
        feature_extractor.eval()
        with torch.no_grad():
            inducing_features = feature_extractor(inducing_sequences)
        feature_extractor.train()
        base_kernel = MaternKernel() + PeriodicKernel()
        inferential_model = LatentSVGP(
            inducing_points = inducing_features,
            base_kernel = base_kernel,
            mean_function = mean_function,
            device = torch.device(DEVICE),
            learn_inducing_locations=True,
            use_mean_function_features=False
            )
        logging.info(f'Inferential Model Prepared with the following parameters: {str(config_best_values)}')
        logging.info(f'Feature Extractor Prepared with the following parameters: {str(config_best_values)}')
        logging.info(f'Mean Function Prepared with the following parameters: {str(inferential_model.mean_module)}')
        compiled_model = DeepKernelSVGP(
            feature_extractor = feature_extractor,
            inferential_model = inferential_model,
            ).to(DEVICE)
        likelihood = GaussianLikelihood().to(DEVICE)

        return compiled_model, likelihood

    trained_model, likelihood = build_model()
    scaler_signature = {
        'features': list(preprocessor.scaler_used.feature_names_in_),
        'scale': preprocessor.scaler_used.scale_.tolist(),
        'offset': preprocessor.scaler_used.min_.tolist(),
    }
    mean_signature = (hashlib.sha256(Path(args.pretrained_mean_function).read_bytes()).hexdigest()
                      if args.pretrained_mean_function else None)
    if args.model_location is not None:
        checkpoint = torch.load(args.model_location, map_location=DEVICE, weights_only=False)
        if checkpoint.get('config') != config_best_values or checkpoint.get('scaler') != scaler_signature:
            raise ValueError("Checkpoint configuration or scaler does not match this run")
        if checkpoint.get('input_features') != INPUT_STUDY_FEATURES:
            raise ValueError("Checkpoint input features do not match this run")
        if checkpoint.get('target_mode') != args.target_mode:
            raise ValueError("Checkpoint target mode does not match this run")
        if checkpoint.get('pretrained_mean_sha256') != mean_signature:
            raise ValueError("Checkpoint pretrained mean function does not match this run")
        trained_model.load_state_dict(checkpoint['model'])
        likelihood.load_state_dict(checkpoint['likelihood'])
    else:
        train_x, train_y = lstm_data['X_train'], lstm_data['y_train']
        validation_data = None
        if TRAIN_W_VALIDATION:
            validation_data = torch.utils.data.TensorDataset(
                torch.from_numpy(lstm_data['X_val']).float(),
                torch.from_numpy(lstm_data['y_val']).float(),
            )
        else:
            train_x = np.concatenate([train_x, lstm_data['X_val']])
            train_y = np.concatenate([train_y, lstm_data['y_val']])
        train_data = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
        natural_gradient_optimizer = optim.NGD(
            trained_model.inferential_process.variational_parameters(),
            num_data=len(train_data), lr=config_best_values['natural_gradient_lr'])
        optimizer = torch.optim.Adam([
            {'params': trained_model.feature_extractor.parameters()},
            {'params': trained_model.inferential_process.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=config_best_values['starting_learning_rate'])
        learning_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config_best_values['max_epochs_per_cycle'], eta_min=0.0)
        trained_model, likelihood, metrics, best_metrics = train_with_gp(
            trained_model, train_data, validation_data,
            batch_size=config_best_values['batch_size'], epochs=args.epochs,
            likelihood=likelihood, optimizer=optimizer,
            natural_gradient_optimizer=natural_gradient_optimizer,
            learning_scheduler=learning_scheduler, min_delta=1e-5,
            patience=args.patience, monitor='val' if TRAIN_W_VALIDATION else 'train',
            device=DEVICE)
        suffix = 'validation' if TRAIN_W_VALIDATION else 'train'
        with open(artifacts_dir / f'training_metrics_{suffix}_stopping.json', 'w') as file:
            json.dump(metrics, file, indent=2)
        logging.info("Best training metrics: %s", best_metrics)
        torch.save({
            'model': trained_model.state_dict(), 'likelihood': likelihood.state_dict(),
            'config': config_best_values, 'scaler': scaler_signature,
            'input_features': INPUT_STUDY_FEATURES,
            'target_mode': args.target_mode,
            'pretrained_mean_sha256': mean_signature,
        }, artifacts_dir / 'checkpoint.pt')

    splits = ('val',) if args.validation_only else (('val', 'test') if TRAIN_W_VALIDATION else ('test',))
    for split in splits:
        mean, variance = predict_with_trained_model(
            trained_model, likelihood, torch.from_numpy(lstm_data[f'X_{split}']).float())
        actual = lstm_data[f'y_{split}']
        if args.target_mode == 'delta':
            origin = lstm_data[f'X_{split}'][:, -1, close_index]
            mean = mean + torch.from_numpy(origin)
            actual = actual + origin
        predictions = create_df_from_predictions(
            mean.numpy(), variance.numpy(), lstm_data[f'dates_{split}'], actual)
        predictions['symbol'] = lstm_data[f'symbols_{split}']
        name = 'validation' if split == 'val' else 'test'
        predictions.to_parquet(artifacts_dir / f'predictions_{name}_set.parquet')
        native = predictions_in_target_units(predictions, preprocessor.scaler_used, 'Close')
        native.to_parquet(artifacts_dir / f'predictions_{name}_native.parquet')


if __name__ == "__main__":
    main()
