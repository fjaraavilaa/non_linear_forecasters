import numpy as np
import torch
from gpytorch import optim
from gpytorch.kernels import MaternKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood

from src.deep_kernel import LstmExtractorSameSizeLayers
from src.gpytorch_model import DeepKernelSVGP, LatentSVGP


def build_hourly_gp(inputs, parameters, device, seed, mean_function=None):
    indices = np.random.default_rng(seed).choice(
        len(inputs), parameters['inducing_points'],
        replace=len(inputs) < parameters['inducing_points'])
    inducing = torch.from_numpy(inputs[indices]).float().to(device)
    extractor = LstmExtractorSameSizeLayers(
        input_size=inputs.shape[2], output_size=1,
        hidden_size=parameters['lstm_hidden_size'], num_layers=parameters['num_layers'],
        dropout=parameters['dropout']).float().to(device)
    gp = LatentSVGP(inducing_points=extractor(inducing),
                    base_kernel=MaternKernel() + PeriodicKernel(),
                    mean_function=mean_function, device=torch.device(device),
                    learn_inducing_locations=True)
    model = DeepKernelSVGP(feature_extractor=extractor, inferential_model=gp).to(device)
    likelihood = GaussianLikelihood().to(device)
    natural = optim.NGD(gp.variational_parameters(), num_data=len(inputs),
                        lr=parameters['natural_gradient_lr'])
    adam = torch.optim.Adam([
        {'params': extractor.parameters()}, {'params': gp.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=parameters['starting_learning_rate'])
    return model, likelihood, adam, natural
