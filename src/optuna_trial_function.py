"""
hyperparameter search for LSTM model using Optuna, with logging and error handling
author: @fjaraavila
"""

import json
from pathlib import Path

import optuna
import logging
import torch
from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel
from src.gpytorch_model import LatentSVGP, DeepKernelSVGP
from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.deep_kernel import LstmExtractorSameSizeLayers
from src.training_functions import train_with_gp
from src.mean_functions import LSTMPreTrainedMeanFunction, LSTMMean_function
from src.lstm_model import LSTMGeneralized_Output_SameSize
import argparse
from gpytorch import optim
from gpytorch.likelihoods import GaussianLikelihood

with open('validation/healthy_dates.json', 'r') as f:
    config_dates = json.load(f)

with open('lstm_only_best_params.json', 'r') as f:
    best_params_mean = json.load(f)

parser = argparse.ArgumentParser(
    description = "hard-coded arguments for the lstm hyper-parameter search"
)
parser.add_argument('--symbols', nargs='+', default=['BTC-USD'], help='List of symbols to fetch data for')
parser.add_argument('--input_study_features', nargs='+', 
                    default=['Log_Returns'], help='List of features to use in the study')
parser.add_argument('--device' , type=str, default='cuda', 
                    help='Device to use for training (e.g., "cuda" or "cpu")')
parser.add_argument('--pretrained_mean_function', type=str, default=None, 
                    help='Path to a pretrained mean function model (optional)')
parser.add_argument('--')

DATES_START_STUDY = config_dates['train']['start']
DATES_END_STUDY = config_dates['test']['end']
SYMBOLS = parser.parse_args().symbols
INPUT_STUDY_FEATURES = parser.parse_args().input_study_features
DEVICE = parser.parse_args().device
PRETRAINED_MEAN_FUNCTION = parser.parse_args().pretrained_mean_function

def objective(trial):
    #Define hyperparameters to optimize
    #lstm_shape_lag = trial.suggest_int('lstm_shape_lag', 5, 100)
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 3, 20)
    batch_size = trial.suggest_int('batch_size', 16, 2500)
    num_layers = trial.suggest_int('num_layers', 1, 20)
    dropout_sug = trial.suggest_float('dropout', 0.0, 0.5)
    natural_gradient_lr = trial.suggest_float('natural_gradient_lr', 0.001, 0.05)
    starting_learning_rate = trial.suggest_float('starting_learning_rate', 0.00001, 0.01)
    max_epochs_per_cycle = trial.suggest_int('max_epochs_per_cycle', 10, 1000)


    #start the data collector
    collector = BinanceDataCollector()
    interval = '5m'
    crypto_raw_data = collector.fetch_crypto_data(symbols = SYMBOLS, 
                                                  interval = interval, 
                                                  start_date = DATES_START_STUDY, 
                                                  end_date = DATES_END_STUDY)
    print(f"Fetching data for symbols: {SYMBOLS} from {DATES_START_STUDY} to {DATES_END_STUDY} with frequency {interval}")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_features(crypto_raw_data, target_col='Log_Returns', drop_bad_values = False)

    processed_data = preprocessor.scale_features(processed_data.set_index(['Date', 'Symbol'])[INPUT_STUDY_FEATURES].dropna(), 
                                                 scaler = 'minmax')
    # Build training and validation splits here from the API-fetched dataframe.
    lstm_data = preprocessor.prepare_lstm_data(processed_data.reset_index(), symbols = SYMBOLS,
                                               sequence_length = best_params_mean["lstm_shape_lag"],
                                               target_col = 'Log_Returns',
                                               prediction_horizon=1,
                                               date_splits =  {
                                                   'train': config_dates['train'],
                                                   'validation': config_dates['validation'],
                                                   'test': config_dates['test']
                                                   },
                                               input_study_features = INPUT_STUDY_FEATURES
                                               )
    
    print(f'LSTM matrix is of shape {str(lstm_data["X_train"].shape)}')
    
    del crypto_raw_data
    del processed_data

    if PRETRAINED_MEAN_FUNCTION is not None:
        mean_function_configs = torch.load(PRETRAINED_MEAN_FUNCTION, map_location=DEVICE)
        lstm_configs = mean_function_configs["model_config"]
        mean_function = LSTMGeneralized_Output_SameSize(
            input_size=lstm_data['X_train'].shape[2],
            hidden_size=lstm_configs["lstm_hidden_size"],
            num_layers=lstm_configs["num_layers"],
            dropout=lstm_configs["dropout"],
            output_size=1,
            )
        mean_function.load_state_dict(mean_function_configs["model_state_dict"])
        mean_function.eval()
        mean_function = LSTMPreTrainedMeanFunction(
            pretrained_model=mean_function, 
            device=torch.device(DEVICE)
            )

    inducing_points_n = trial.suggest_int('inducing_points', 50, 1000)
    inducing_points = torch.randn(inducing_points_n, lstm_data['X_train'].shape[1],
                                  lstm_data['X_train'].shape[2]).float()

    feature_extractor = LstmExtractorSameSizeLayers(
        input_size = lstm_data['X_train'].shape[2],
        output_size = 1,
        hidden_size = lstm_hidden_size,
        num_layers = num_layers,
        dropout = dropout_sug
    ).float()
    base_kernel = MaternKernel() + PeriodicKernel()
    if PRETRAINED_MEAN_FUNCTION is not None:
        mean_function = LSTMPreTrainedMeanFunction(
            pretrained_model=mean_function, 
            device=torch.device(DEVICE)
            )
        inferential_model = LatentSVGP(
            inducing_points = feature_extractor(inducing_points.float()),
            base_kernel = base_kernel,
            mean_function = mean_function,
            device = torch.device(DEVICE),
            learn_inducing_locations=True
            )
    else:
        inferential_model = LatentSVGP(
            inducing_points = feature_extractor(inducing_points.float()),
            base_kernel = base_kernel,
            mean_function = None,
            device = torch.device(DEVICE),
            learn_inducing_locations=True
            )

    compiled_model = DeepKernelSVGP(
        feature_extractor = feature_extractor,
        inferential_model = inferential_model,
        )
    likelihood = GaussianLikelihood()

    natural_gradient_optimizer = optim.NGD(
        compiled_model.inferential_process.variational_parameters(), 
        num_data = lstm_data['X_train'].shape[0], 
        lr = natural_gradient_lr
        )

    optimizer = torch.optim.Adam([
        {'params': compiled_model.feature_extractor.parameters()},
        {'params': compiled_model.inferential_process.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=starting_learning_rate)

    learning_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epochs_per_cycle, 
            eta_min=0.0
        )
    
    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(lstm_data['X_train']).float(),
        torch.from_numpy(lstm_data['y_train']).float(),

    )
    val_data = torch.utils.data.TensorDataset(
        torch.from_numpy(lstm_data['X_val']).float(),
        torch.from_numpy(lstm_data['y_val']).float()
    )
    trained_model, likelihood, metrics, best_metrics = train_with_gp(
        compiled_model,
        train_data,
        val_data,
        batch_size=batch_size,
        epochs=10000,
        likelihood=likelihood,
        optimizer=optimizer,
        natural_gradient_optimizer=natural_gradient_optimizer,
        learning_scheduler=learning_scheduler,
        min_delta=1e-5,
        patience=50,
        monitor='val',
        save_metrics=False
    )
    return best_metrics['mae_val'] # Return the validation loss for Optuna to minimize

storage_name = "sqlite:///mydb_lstm_crypto_GP_5min_w_lag.db"

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="lstm_crypto_study_PeriodicMaternKernel_changepretrainedfunction", storage=storage_name
    )
    study.optimize(objective, n_trials=40)
