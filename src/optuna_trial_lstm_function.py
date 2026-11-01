"""
Hyper-parametrer function for the LSTM mean function. Using Optuna, logging and error handling
author: @fjaraavila
"""

import json
from pathlib import Path

import optuna
import logging
import torch
from src.lstm_model import LSTMGeneralized_Output_SameSize
from src.training_functions import train_lstm_model
from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.deep_kernel import LstmExtractorSameSizeLayers
from src.training_functions import train_with_gp
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('validation/healthy_dates.json', 'r') as f:
    config_dates = json.load(f)

parser = argparse.ArgumentParser(
    description = "hard-coded arguments for the lstm hyper-parameter search"
)
parser.add_argument('--symbols', nargs='+', default=['BTC-USD'], help='List of symbols to fetch data for')
parser.add_argument('--input_study_features', nargs='+', default=['Returns'], help='List of features to use in the study')
parser.add_argument('--device' , type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')
parser.add_argument('--used_optimizer', type=str, default='adam', 
                    help='Optimizer to use for training (e.g., "adam" or "sgd")')
parser.add_argument('--loss_function', type=str, default='mae', 
                    help='Loss function to use for training (e.g., "mae" or "mse" or "huber")')
parser.add_argument('--huber_parameter', type=float, default=None)

DATES_START_STUDY = config_dates['train']['start']
DATES_END_STUDY = config_dates['test']['end']
SYMBOLS = parser.parse_args().symbols
INPUT_STUDY_FEATURES = parser.parse_args().input_study_features
USED_OPTIMIZER = parser.parse_args().used_optimizer
LOSS_FUNCTION = parser.parse_args().loss_function
HUBER_PARAMETER = parser.parse_args().huber_parameter
DEVICE = parser.parse_args().device

def objective(trial):
    #Define hyperparameters to optimize
    lstm_shape_lag = trial.suggest_int('lstm_shape_lag', 5, 100)
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 3, 20)
    batch_size = trial.suggest_int('batch_size', 16, 2500)
    num_layers = trial.suggest_int('num_layers', 1, 20)
    dropout_sug = trial.suggest_float('dropout', 0.0, 0.8)
    starting_learning_rate = trial.suggest_float('starting_learning_rate', 0.0001, 0.1)
    #max_epochs_per_cycle = trial.suggest_int('max_epochs_per_cycle', 10, 1000)

    #start the data collector
    collector = BinanceDataCollector()
    interval = '5m'
    print(f"Fetching data for symbols: {SYMBOLS} from {DATES_START_STUDY} to {DATES_END_STUDY} with frequency {interval}")
    crypto_raw_data = collector.fetch_crypto_data(symbols = SYMBOLS, 
                                                  interval = interval, 
                                                  start_date = DATES_START_STUDY, 
                                                  end_date = DATES_END_STUDY)
    print(crypto_raw_data.head())
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_features(crypto_raw_data, target_col='Log_Returns', 
                                                   drop_bad_values = False)

    processed_data = preprocessor.scale_features(processed_data.set_index(['Date', 'Symbol'])[INPUT_STUDY_FEATURES].dropna(), 
                                                 scaler = 'minmax')
    # Build training and validation splits here from the API-fetched dataframe.
    lstm_data = preprocessor.prepare_lstm_data(processed_data.reset_index(), symbols = SYMBOLS,
                                               sequence_length = lstm_shape_lag,
                                               target_col = 'Close',
                                               prediction_horizon=1,
                                               date_splits =  {
                                                   'train': config_dates['train'],
                                                   'validation': config_dates['validation'],
                                                   'test': config_dates['test']
                                                   },
                                               input_study_features = INPUT_STUDY_FEATURES
                                               )
    
    print(f'LSTM matrix is of shape {str(lstm_data["X_train"].shape)}')

    logging.info(f'Hyperparameters for trial {trial.number}: lstm_shape_lag={lstm_shape_lag}, lstm_hidden_size={lstm_hidden_size}, batch_size={batch_size}, num_layers={num_layers}, dropout={dropout_sug}, starting_learning_rate={starting_learning_rate}')

    
    del crypto_raw_data
    del processed_data

    lstm_model = LSTMGeneralized_Output_SameSize(input_size=len(INPUT_STUDY_FEATURES),
                                                 hidden_size=lstm_hidden_size,
                                                 num_layers=num_layers,
                                                 dropout=dropout_sug,
                                                 output_size=1)

    if LOSS_FUNCTION == 'mae':
        optimizing_criterion = torch.nn.L1Loss()
    elif LOSS_FUNCTION == 'mse':
        optimizing_criterion = torch.nn.MSELoss()
    elif LOSS_FUNCTION == 'huber':
        if HUBER_PARAMETER is not None:
            optimizing_criterion = torch.nn.HuberLoss(delta=float(HUBER_PARAMETER))
        else:
            optimizing_criterion = torch.nn.HuberLoss()
    else:
        raise ValueError("Invalid loss function. Please choose from 'mae', 'mse', or 'huber'.")

    train_data = torch.utils.data.TensorDataset(torch.tensor(lstm_data['X_train'], dtype=torch.float32),
                                                torch.tensor(lstm_data['y_train'], dtype=torch.float32))
    val_data = torch.utils.data.TensorDataset(torch.tensor(lstm_data['X_val'], dtype=torch.float32),
                                              torch.tensor(lstm_data['y_val'], dtype=torch.float32))    

    used_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=starting_learning_rate)
    lstm_model, metrics, best_metrics = train_lstm_model(lstm_model,
                                                optimizing_criterion,
                                                train_data=train_data,
                                                val_data=val_data,
                                                optimizer=used_optimizer,
                                                epochs = 1000,
                                                #max_epochs_per_cycle=max_epochs_per_cycle,
                                                patience=50,
                                                min_delta=1e-5,
                                                batch_size=batch_size,
                                                optimizer_metric='mae',
                                                give_model_metrics=True
                                                )

    return best_metrics['mae_val'] # Return the validation loss for Optuna to minimize

storage_name = "sqlite:///mydb_lstm_5minprediction.db"

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="lstm_crypto_study_mean_function_minmax_morepatience", storage=storage_name
    )
    study.optimize(objective, n_trials=40)
