import logging
import os
print(os.getcwd())
from src.gpytorch_model import DeepKernelSVGP, LatentSVGP
from src.deep_kernel import LstmExtractorSameSizeLayers, LstmExtractorDifferentSizeLayers
import torch
from tests.test_fetch import test_transform_to_lstm_matrix
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
logging.basicConfig(level=logging.INFO)


sequences_lstm = test_transform_to_lstm_matrix()
inducing_points = torch.randn(10, sequences_lstm['X_train'].shape[1], sequences_lstm['X_train'].shape[2])  # 10 inducing points, sequence length 30, input features 2
input_shape = inducing_points.shape # sequence length 30, input features
hidden_size = 2
dropout = 0.1
output_dimension = 1
number_hidden_lstm_kernel = 2

def test_model():
    feature_extractor = LstmExtractorSameSizeLayers(
        input_size = input_shape[-1],  # input features
        hidden_size = hidden_size,
        num_layers = number_hidden_lstm_kernel,
        output_size = output_dimension,
        dropout = dropout)
    base_kernel = RBFKernel()
    inferential_model = LatentSVGP(
        inducing_points=feature_extractor(inducing_points.float()),
        base_kernel=base_kernel,
        device = torch.device('cpu'),
        learn_inducing_locations=True)
    logging.info(f"Inducing points shape after feature extraction: {feature_extractor(inducing_points.float()).shape}")
    likelihood = GaussianLikelihood()
    compiled_model = DeepKernelSVGP(
        feature_extractor=feature_extractor,
        inferential_model=inferential_model
    )
    output = compiled_model(
        torch.from_numpy(sequences_lstm['X_train'][:100]).float()
        )
    return output

test_output = test_model()