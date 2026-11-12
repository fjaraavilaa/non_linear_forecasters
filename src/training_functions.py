import logging
from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from gpytorch.likelihoods import likelihood
from gpytorch.mlls import VariationalELBO
import json
from gpytorch import optim
import copy

from src.gpytorch_model import DeepKernelSVGP, LatentSVGP

logging.basicConfig(level=logging.INFO)

def compute_prediction_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute MSE and MAE between predictions and targets.
 
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Model predictions, same shape as y_true.
 
    Returns:
        tuple: (mse, mae)
    """
    y_true = y_true.detach().flatten().double()
    y_pred = y_pred.detach().flatten().double()
 
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()
 
    return mse, mae

def train_lstm_model(model: nn.Module, optimizing_criterion: nn.Module, 
                     train_data: TensorDataset, 
                     val_data: TensorDataset, epochs: int, optimizer: torch.optim.Optimizer,
                     patience: int = 5, min_delta: float = 1e-5, batch_size: int = 32,
                     optimizer_metric: str = 'mse', give_model_metrics: bool = False):

    """Train an LSTM model with early stopping based on validation loss.
    
    Args:
        model (nn.Module): The LSTM model to be trained.
        optimizing_criterion (nn.Module): The loss function to optimize.
        train_data_loader (DataLoader): DataLoader for the training data.
        val_data_loader (DataLoader): DataLoader for the validation data.
        epochs (int): The number of epochs to train for.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        patience (int): The number of epochs to wait before stopping early.
        min_delta (float): The minimum change in the validation loss to qualify as an improvement.
        optimizer_metric (str): The metric to use for early stopping ('mse' or 'mae').

    Returns:
        nn.Module: The trained model.
    """
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    mae_losses = []
    mse_losses = []
    mse_losses_val = []
    mae_losses_val = []

    best_val = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = optimizing_criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        


        model.eval()
        with torch.no_grad():
            mse_train_local, mae_train_local = compute_prediction_metrics(
                torch.cat([y for _, y in train_data_loader]), 
                torch.cat([model(x) for x, _ in train_data_loader])
            )
            mse_losses.append(mse_train_local)
            mae_losses.append(mae_train_local)
            val_mse, val_mae = compute_prediction_metrics(
                torch.cat([y for _, y in val_data_loader]), 
                torch.cat([model(x) for x, _ in val_data_loader])
            )

        print(torch.cat([model(x) for x, _ in val_data_loader]))
        print(torch.cat([y for _, y in val_data_loader]))
        if optimizer_metric == 'mse':
            val_loss = val_mse
        elif optimizer_metric == 'mae':
            val_loss = val_mae
        else:
            raise ValueError(f"Invalid optimizer_metric: {optimizer_metric}. Must be 'mse' or 'mae'.")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.6f}, Train MSE: {mse_train_local:.6f}, Train MAE: {mae_train_local:.6f}, Validation Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")
        mse_losses_val.append(val_mse)
        mae_losses_val.append(val_mae)
        if val_loss <= best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
            best_metrics = {
                "epoch": epoch,
                "best_val_loss": best_val,
                "mse_train": mse_train_local,
                "mae_train": mae_train_local,
                "mse_val": val_mse,
                "mae_val": val_mae
            }
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_state)
                break
    if give_model_metrics:
        metrics = {
            "mse_losses_train": mse_losses,
            "mae_losses_train": mae_losses,
            "mse_losses_val": mse_losses_val,
            "mae_losses_val": mae_losses_val
        }
        return model, metrics, best_metrics
    else:
        return model



def train_with_gp(model: DeepKernelSVGP, train_data: TensorDataset, 
                  val_data: TensorDataset, batch_size: int, epochs: int,
                  likelihood: likelihood, optimizer: torch.optim,
                  natural_gradient_optimizer: optim = None,
                  learning_scheduler: torch.optim.lr_scheduler = None,
                  min_delta: float = 0.0, patience: int = 0, monitor: str = 'val',
                  save_suffix: str = '', save_metrics: bool = False, 
                  device: torch.device = torch.device('cpu'),
                  checkpoint_path=None) -> LatentSVGP:
    """This funtions trains an SVGP model 

    Args:
        model (DeepKernelSVGP): The model to be trained with
        all the specifications already done.
        train_data (TensorDataset): Training Data already passed
        as a tensor
        val_data (TensorDataset): Validation Data already passed
        as a tensor
        batch_size (int): Batch size (can be optimized)
        epochs (int): Epochs to run
        likelihood (likelihood): Likelihood from gpflow used.
        Usually it'll be gaussian
        optimizer (torch.optim.Optimizer): optimizer from pytorch
        natural_gradient_optimizer (gpytorch.optim.Optimizer): natural gradient optimizer from pytorch
        learning_scheduler (torch.optim.lr_scheduler): learning rate scheduler from pytorch
        min_delta (float): minimum permitted difference
        patience (int): How many trials to permit

    Returns:
        LatentSVGP: Fully trained model.
    """
    if monitor not in ("val", "train"):
        raise ValueError(f"monitor must be 'val' or 'train', got {monitor!r}")
    if monitor == "val" and val_data is None:
        raise ValueError("monitor='val' requires val_data to be provided")
    if not len(train_data) or (val_data is not None and not len(val_data)):
        raise ValueError("Training and validation datasets must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train_loader = DataLoader(train_data, batch_size, shuffle = False)
    val_loader = DataLoader(val_data, batch_size, shuffle = False) if val_data is not None else None
    mll = VariationalELBO(
        likelihood.to(device), model.inferential_process.to(device), num_data = len(train_data)
    )

    model.train()
    best_val = float('inf')
    counter = 0
    
    likelihood.train()
    logging.info("Starting GP training...")
    train_losses = []
    val_losses = []
    mse_losses = []
    mae_losses = []
    mse_losses_val = []
    mae_losses_val = []
    for epoch in range(epochs):
        batch_losses = []
        model.train()
        likelihood.train()
        for x_batch, y_batch in train_loader:
            #logging.info(f"Epoch {epoch + 1}/{epochs}, Batch Loss: {mll(model(x_batch), y_batch).item()}")
            #logging.info(f"Epoch {epoch + 1}/{epochs}, Input shape: {x_batch.shape}, Target shape: {y_batch.shape}")
            if natural_gradient_optimizer is not None:
                natural_gradient_optimizer.zero_grad()
            optimizer.zero_grad()        
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if not torch.isfinite(x_batch).all() or not torch.isfinite(y_batch).all():
                raise ValueError("Non-finite values found in a training batch")
            outputs = model(x_batch)
            loss = -mll(outputs, y_batch).to(device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GP loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            if natural_gradient_optimizer is not None:
                natural_gradient_optimizer.step()
            optimizer.step()
            batch_losses.append((loss.item(), len(y_batch)))
        train_loss = sum(loss * count for loss, count in batch_losses) / len(train_data)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}")
        train_losses.append(train_loss)
        if learning_scheduler is not None:
            learning_scheduler.step()
        model.eval()
        likelihood.eval()
        if val_loader is not None:
            with torch.no_grad():
                val_loss = sum(
                    -mll(model(x_batch.to(device)), y_batch.to(device)) * len(y_batch)
                    for x_batch, y_batch in val_loader
                ) / len(val_data)
                if not torch.isfinite(val_loss):
                    raise FloatingPointError("Non-finite GP validation loss")
            val_losses.append(val_loss.item())
        
        def metrics_for_loader(loader):
            with torch.no_grad():
                pairs = [(y.to(device), model(x.to(device)).mean) for x, y in loader]
                return compute_prediction_metrics(
                    torch.cat([y for y, _ in pairs]),
                    torch.cat([prediction for _, prediction in pairs]),
                )

        mse_train_local, mae_train_local = metrics_for_loader(train_loader)
        mse_losses.append(mse_train_local)
        mae_losses.append(mae_train_local)
        if val_loader is not None:
            mse_val_local, mae_val_local = metrics_for_loader(val_loader)
            mse_losses_val.append(mse_val_local)
            mae_losses_val.append(mae_val_local)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss.item():.6f}, MSE Train: {mse_train_local:.6f}, MAE Train: {mae_train_local:.6f}, MSE Val: {mse_val_local:.6f}, MAE Val: {mae_val_local:.6f}")
        else:
            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, MSE Train: {mse_train_local:.6f}, MAE Train: {mae_train_local:.6f}")

        current_metric = val_loss if monitor == "val" else train_loss
 
        if current_metric < best_val - min_delta:
            best_val = current_metric
            best_metrics = {
                "epoch": epoch,
                "best_elbo": current_metric.item() if torch.is_tensor(current_metric) else current_metric,
                "mse_train": mse_train_local,
                "mae_train": mae_train_local,
                "mse_val": mse_val_local if val_loader is not None else None,
                "mae_val": mae_val_local if val_loader is not None else None,
                }
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "likelihood": copy.deepcopy(likelihood.state_dict()),
                }
            counter = 0
        else:
            counter += 1

        logging.info(
            f"Best epoch so far: {best_metrics['epoch'] + 1}, "
            f"Best validation MSE: {best_metrics['mse_val']}, "
            f"Best validation MAE: {best_metrics['mae_val']}"
        )
        
        if checkpoint_path is not None and counter == 0:
            torch.save(best_state, checkpoint_path)
        
        if patience > 0 and counter >= patience:
            logging.info(f"Early Stopping at Epoch {epoch} (monitor={monitor})")
            break

    
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "mse_losses_train": mse_losses,
        "mae_losses_train": mae_losses,
        "mse_losses_val": mse_losses_val,
        "mae_losses_val": mae_losses_val
    }

    model.load_state_dict(best_state["model"])
    likelihood.load_state_dict(best_state["likelihood"])

    return (model, likelihood, metrics, best_metrics)
