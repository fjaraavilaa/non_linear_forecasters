import pandas as pd
from gpytorch import Module
from gpytorch.likelihoods import Likelihood
from scipy.stats import norm
import torch
import plotly
import numpy as np

def predict_with_trained_model(model: Module, likelihood: Likelihood, X_test: torch.Tensor,
                               batch_size: int = 256):
    """
    Predicts the output using the trained model and likelihood.

    Args:
        model: The trained model.
        likelihood: The likelihood function.
        X_test: The input data for prediction.
    """
    model.eval()
    likelihood.eval()
    if len(X_test) == 0:
        raise ValueError("Cannot predict an empty dataset")
    device = next(model.parameters()).device

    with torch.no_grad():
        mean = []
        variance = []
        for batch in torch.utils.data.DataLoader(X_test, batch_size=batch_size):
            # Get the predictive distribution
            predictive_dist = likelihood(model(batch.to(device)))

            # Get the mean and variance of the predictions
            mean.append(predictive_dist.mean.cpu())
            variance.append(predictive_dist.variance.cpu())

    mean = torch.cat(mean)
    variance = torch.cat(variance)

    return mean, variance

def predictions_in_target_units(predictions_df, scaler, target_col):
    """Invert MinMaxScaler's affine transform, including squared variance units."""
    features = list(scaler.feature_names_in_)
    if target_col not in features:
        raise ValueError(f"Scaler does not contain target {target_col!r}")
    index = features.index(target_col)
    scale = scaler.scale_[index]
    offset = scaler.min_[index]
    result = predictions_df.copy()
    for column in ('mean', 'actual'):
        if column in result:
            result[column] = (result[column] - offset) / scale
    result['variance'] = result['variance'] / scale ** 2
    values = result[['mean', 'variance']].to_numpy()
    if not np.isfinite(values).all() or (result['variance'] < 0).any():
        raise ValueError("Predictions contain non-finite means or invalid variances")
    return result

def predict_with_torch_model(model: Module, X_test: torch.Tensor):
    """
    Predicts the output using the trained PyTorch model.

    Args:
        model: The trained PyTorch model.
        X_test: The input data for prediction.
    """
    model.eval()

    with torch.no_grad():
        mean = []
        variance = []
        for batch in torch.utils.data.DataLoader(X_test):
            # Get the predictions
            predictions = model(batch)

            # Get the mean and variance of the predictions
            mean.append(predictions)
            variance.append(torch.zeros_like(predictions))  # Assuming no variance for standard PyTorch models

    mean = torch.cat(mean)
    variance = torch.cat(variance)

    return mean, variance

def create_df_from_predictions(mean, variance, indexing, actual=None):
    """
    Creates a DataFrame from the predictions.

    Args:
        mean: The mean predictions.
        variance: The variance of the predictions.
        indexing: The index for the DataFrame.
        actual: The observed target values, if available.
    """
    predictions_df = pd.DataFrame({
        'mean': mean,
        'variance': variance
    }, index=indexing)
    if actual is not None:
        predictions_df['actual'] = np.asarray(actual).reshape(-1)

    return predictions_df

def plot_predictions(predictions_df, title="Predictions", confidence=0.95):
    """
    Plots the predictions using Plotly.

    Args:
        predictions_df: DataFrame containing the mean and variance of predictions.
        title: Title of the plot.
        confidence: Confidence level for the shaded interval (default 0.95 for a 95% CI).
    """
    fig = plotly.graph_objects.Figure()

    # Add mean predictions
    fig.add_trace(plotly.graph_objects.Scatter(
        x=predictions_df.index,
        y=predictions_df['mean'],
        mode='lines',
        name='Mean Prediction'
    ))

    # Compute the confidence interval from the standard deviation
    z = norm.ppf(0.5 + confidence / 2)
    std = predictions_df['variance'] ** 0.5
    upper = predictions_df['mean'] + z * std
    lower = predictions_df['mean'] - z * std

    x_index = predictions_df.index.to_series()

    # Add confidence interval as shaded area
    fig.add_trace(plotly.graph_objects.Scatter(
        x=pd.concat([x_index, x_index[::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{int(confidence * 100)}% CI'
    ))

    fig.update_layout(title=title,
                      xaxis_title='Index',
                      yaxis_title='Predicted Value',
                      template='plotly_white')

    fig.show()

def lstm_prediction(model: torch.nn.Module, X_test: torch.Tensor, likelihood: Likelihood, indexing: pd.Index):
    """
    Predicts the output using the trained LSTM model and likelihood.

    Args:
        model: The trained LSTM model.
        X_test: The input data for prediction.
        likelihood: The likelihood function.
        indexing: The index for the DataFrame.
    """
    mean, variance = predict_with_trained_model(model, likelihood, X_test)
    predictions_df = create_df_from_predictions(mean, variance, indexing)
    return predictions_df

def compute_huber_delta(predictions_df: pd.DataFrame, 
                        prediction_col: str,
                        actuals_col: str, k: float = 1.345, 
                        metric: str = 'mae') -> float:
    """
    Computes the Huber delta for the predictions.

    Args:
        predictions_df: DataFrame containing the mean and variance of predictions.
        prediction_col: The column name for the predictions.
        actuals_col: The column name for the actual values.
        k: The parameter for the Huber loss (default is 1.345).
        metric: The metric to use for computing the residuals (default is 'mae' for Mean Absolute Error).
    """
    residuals = predictions_df[actuals_col] - predictions_df[prediction_col]
    if metric == 'mae':
        sigma = 1.345 * residuals.std()  # Common choice for delta in Huber loss
    elif metric == 'mse':
        sigma = 1.345 * (residuals ** 2).std()  # Adjusted for MSE
    elif metric == 'mad':
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = 1.4826 * mad  # Adjusted for MAD
    else:
        raise ValueError("Invalid metric. Please choose 'mae', 'mse', or 'mad'.")

    delta = k * sigma  # Common choice for delta in Huber loss
    return delta