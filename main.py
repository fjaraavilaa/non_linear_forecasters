import logging
import torch
from gpytorch.kernels import MaternKernel, PeriodicKernel
from src.gpytorch_model import LatentSVGP, DeepKernelSVGP
from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.deep_kernel import LstmExtractorSameSizeLayers
from src.training_functions import train_with_gp
from src.help_functions.predictor import predict_with_trained_model, create_df_from_predictions
from src.lstm_model import LSTMGeneralized_Output_SameSize
from src.mean_functions import LSTMPreTrainedMeanFunction
import argparse
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import optim
import json

logging.basicConfig(level=logging.INFO)

with open('best_trial_params.json', 'r') as f:
    config_best_values = json.load(f)

with open('lstm_only_best_params.json', 'r') as f:
    config_mean_values = json.load(f)


def main():
    with open('validation/healthy_dates.json', 'r') as f:
        config_dates = json.load(f)
    
    with open('best_trial_params.json', 'r') as f:
        config_best_values = json.load(f)
    with open('lstm_only_best_params.json', 'r') as f:
        config_mean_values = json.load(f)

    parser = argparse.ArgumentParser(
        description = "hard-coded arguments for the lstm+kernel"
    )

    DATES_START_STUDY = config_dates['train']['start']
    DATES_END_STUDY = config_dates['test']['end']


    parser.add_argument('--symbols', nargs='+', default=['BTC-USD'], help='List of symbols to fetch data for')
    parser.add_argument('--input_study_features', nargs='+', default=None, help='List of features to use in the study')
    parser.add_argument('--device' , type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--work_with_validation_set', type=bool, default=False)
    parser.add_argument('--model_location', type=str, default=None, help='Location for the model')
    parser.add_argument('--pretrained_mean_function', type=str, default=None, help='Pretrained LSTM checkpoint')
    args = parser.parse_args()
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
    
    collector = BinanceDataCollector()
    interval = '5m'
    crypto_raw_data = collector.fetch_crypto_data(symbols = SYMBOLS, 
                                                  interval = interval, 
                                                  start_date = DATES_START_STUDY, 
                                                  end_date = DATES_END_STUDY)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_features(crypto_raw_data, target_col = 'Log_Returns', drop_bad_values = False)
    logging.info('Features Prepared')

    processed_data = preprocessor.scale_features(processed_data.set_index(['Date', 'Symbol'])[INPUT_STUDY_FEATURES].dropna(), 
                                                     scaler = 'minmax')  
    
    logging.info('Features Scaled')

    lstm_data = preprocessor.prepare_lstm_data(processed_data.reset_index(), symbols = SYMBOLS,
                                               sequence_length = config_mean_values['lstm_shape_lag'],
                                               target_col = 'Log_Returns',
                                               prediction_horizon=1,
                                               date_splits =  {
                                                   'train': config_dates['train'],
                                                   'validation': config_dates['validation'],
                                                   'test': config_dates['test']
                                                   },
                                               input_study_features = INPUT_STUDY_FEATURES
                                               )      

    logging.info(f'lstm matrixes done. Check Dimensions: {str(lstm_data["X_train"].shape)}')


    if args.model_location is None:
        

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
        inducing_sequences = torch.randn(config_best_values['inducing_points'], 
                                      lstm_data['X_train'].shape[1], 
                                      lstm_data['X_train'].shape[2],
                                      device=DEVICE)
        base_kernel = MaternKernel() + PeriodicKernel()
        inferential_model = LatentSVGP(
            inducing_points = feature_extractor(inducing_sequences.float()),
            base_kernel = base_kernel,
            mean_function = mean_function,
            device = DEVICE,
            learn_inducing_locations=True,
            use_mean_function_features=False,
            use_natural_gradient=True
            )
        compiled_model = DeepKernelSVGP(
            feature_extractor = feature_extractor,
            inferential_model = inferential_model,
            ).to(DEVICE)
        likelihood = GaussianLikelihood().to(DEVICE)


        natural_gradient_optimizer = optim.NGD(
            compiled_model.inferential_process.variational_parameters(), 
            num_data = lstm_data['X_train'].shape[0], 
            lr = config_best_values['natural_gradient_lr']
        )

        optimizer = torch.optim.Adam([
            {'params': compiled_model.feature_extractor.parameters()},
            {'params': compiled_model.inferential_process.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=config_best_values.get('starting_learning_rate', 0.001))

        learning_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config_best_values.get('max_epochs_per_cycle', 20), 
            eta_min=0.0
        )
        
        if TRAIN_W_VALIDATION:
            logging.info("Training with validation set...")
            train_data = torch.utils.data.TensorDataset(
                torch.from_numpy(lstm_data['X_train']).float(),
                torch.from_numpy(lstm_data['y_train']).float(),
            )   
            validation_data = torch.utils.data.TensorDataset(
                torch.from_numpy(lstm_data['X_val']).float(),
                torch.from_numpy(lstm_data['y_val']).float(),
            )
            trained_model, likelihood, metrics, best_metrics = train_with_gp(
                compiled_model,
                train_data,
                validation_data,
                batch_size=config_best_values['batch_size'],
                epochs=10000,
                likelihood=likelihood,
                natural_gradient_optimizer=natural_gradient_optimizer,
                learning_scheduler=learning_scheduler,
                optimizer=optimizer,
                min_delta=1e-5,
                patience=50,
                monitor='val',
                save_suffix = 'validation_stopping',
                save_metrics=True
            )
            with torch.no_grad():
                prediction_mean, prediction_variance = predict_with_trained_model(trained_model, 
                                                                                  likelihood, 
                                                                                  torch.from_numpy(lstm_data['X_val']).float())
            
            predictions = create_df_from_predictions(prediction_mean.numpy(), 
                                                     prediction_variance.numpy(), 
                                                     lstm_data['dates_val'])
            predictions.to_parquet('predictions_validation_set.parquet')

        else:
            train_data = torch.utils.data.TensorDataset(
                torch.cat([torch.from_numpy(lstm_data['X_train']).float(), 
                           torch.from_numpy(lstm_data['X_val']).float()]),
                           torch.cat([torch.from_numpy(lstm_data['y_train']).float(),
                                      torch.from_numpy(lstm_data['y_val']).float()]),
                                      )        
            trained_model, likelihood, metrics, best_metrics = train_with_gp(
                compiled_model,
                train_data,
                None,
                batch_size=config_best_values['batch_size'],
                epochs=10000,
                likelihood=likelihood,
                natural_gradient_optimizer=natural_gradient_optimizer,
                learning_scheduler=learning_scheduler,
                min_delta=1e-5,
                patience=50,
                monitor='train',
                save_suffix = 'train_stopping',
                save_metrics=True
            )
        with torch.no_grad():
            prediction_mean, prediction_variance = predict_with_trained_model(trained_model, 
                                                                              likelihood, 
                                                                              torch.from_numpy(lstm_data['X_test']).float())
        
        predictions = create_df_from_predictions(prediction_mean.numpy(), 
                                                 prediction_variance.numpy(), 
                                                 lstm_data['dates_test'])
        predictions.to_parquet('predictions_test_set.parquet')
    else:
        trained_model = torch.load(args.model_location, map_location=DEVICE)
        with torch.no_grad():
            prediction_mean, prediction_variance = predict_with_trained_model(trained_model['model'], 
                                                                              trained_model['likelihood'],
                                                                              torch.from_numpy(lstm_data['X_test']).float())
        predictions = create_df_from_predictions(prediction_mean.numpy(), 
                                                 prediction_variance.numpy(), 
                                                 lstm_data['dates_test'])
        predictions.to_parquet('predictions_test_set.parquet')


if __name__ == "__main__":
    main()