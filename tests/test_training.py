from src.training_functions import train_with_gp
from src.gpytorch_model import DeepKernelSVGP, LatentSVGP
from src.deep_kernel import LstmExtractorSameSizeLayers, LstmExtractorDifferentSizeLayers
import torch
from tests.test_fetch import test_transform_to_lstm_matrix
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood


sequences_lstm = test_transform_to_lstm_matrix()
inducing_points = torch.randn(10, sequences_lstm['X_train'].shape[1], sequences_lstm['X_train'].shape[2])  # 10 inducing points, sequence length 30, input features 2
input_shape = inducing_points.shape # sequence length 30, input features
hidden_size = 2
dropout = 0.1
output_dimension = 1
number_hidden_lstm_kernel = 2

feature_extractor = LstmExtractorSameSizeLayers(
    input_size = input_shape[-1],  # input features
    hidden_size = hidden_size,
    num_layers = number_hidden_lstm_kernel,
    output_size = output_dimension,
    dropout = dropout)
base_kernel = RBFKernel()
inferential_model = LatentSVGP(
    inducing_points=feature_extractor(inducing_points.float()) ,
    base_kernel=base_kernel,
    device = torch.device('cpu'),
    learn_inducing_locations=True)
compiled_model = DeepKernelSVGP(
    feature_extractor=feature_extractor,
    inferential_model=inferential_model
)

likelihood = GaussianLikelihood()
optimizer = torch.optim.Adam([{
    'params': compiled_model.feature_extractor.parameters(),
    'params': compiled_model.inferential_process.hyperparameters(),
    'params': compiled_model.inferential_process.variational_parameters(),
    'params': likelihood.parameters(),
    'lr': 0.01
    }])


def test_training():
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(sequences_lstm['X_train'][:100]).float(), torch.from_numpy(sequences_lstm['y_train'][:100]).float())
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(sequences_lstm['X_val'][:100]).float(), torch.from_numpy(sequences_lstm['y_val'][:100]).float())
    trained_model = train_with_gp(compiled_model, train_data, val_data, 
                                  batch_size=25, epochs=5, likelihood=likelihood, 
                                  optimizer=optimizer,
                                  min_delta=1e-4, patience=5, monitor='val', save_metrics=False)
    return trained_model

test_trained_model = test_training()