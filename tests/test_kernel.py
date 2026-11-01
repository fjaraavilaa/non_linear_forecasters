import logging
from src.deep_kernel import LstmExtractorSameSizeLayers, LstmExtractorDifferentSizeLayers
from src.data_preprocessor import DataPreprocessor
from tests.test_fetch import test_fetch, test_transform_to_lstm_matrix
from gpytorch.kernels import RBFKernel
import torch

df_testing = test_fetch()
sequences_lstm = test_transform_to_lstm_matrix()

logging.basicConfig(level=logging.INFO)

def test_LSTMDeepKernel_same_size_layers():
    lstm_parameters = {
        "input_size": 2,
        "hidden_size": 5,
        "num_layers": 2,
        "output_size": 5
    }
    covar_function = RBFKernel()
    extractor_features = LstmExtractorSameSizeLayers(lstm_parameters["input_size"], 
                                                         lstm_parameters["hidden_size"], 
                                                         lstm_parameters["num_layers"],
                                                         lstm_parameters["output_size"])
    
    features = extractor_features(torch.tensor(sequences_lstm['X_train'][:10], dtype=torch.float32))
    logging.info(f"Extracted features shape: {features.shape}")

    return covar_function(features)

stupid = test_LSTMDeepKernel_same_size_layers()

def test_LSTMDeepKernel_different_size_layers():
    lstm_parameters = {
        "input_size": 2,
        "hidden_sizes": [5, 10],
        "num_layers": 2,
        "output_size": 10
    }
    covar_function = RBFKernel()
    extractor_features = LstmExtractorDifferentSizeLayers(lstm_parameters["input_size"], 
                                                            lstm_parameters["hidden_sizes"], 
                                                            lstm_parameters["num_layers"],
                                                            lstm_parameters["output_size"],
                                                            batch_size=32)
    
    logging.info(f"LSTM layers: {extractor_features.lstm_layers}")

    features = extractor_features(torch.tensor(sequences_lstm['X_train'][:10], dtype=torch.float32))
    logging.info(f"Extracted features shape: {features.shape}")

    return covar_function(features)

test_lstm_diff_sizes = test_LSTMDeepKernel_different_size_layers()
test_lstm_diff_sizes.shape