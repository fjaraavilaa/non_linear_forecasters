"""
LSTM Model Architecture for Cryptocurrency Volatility Prediction

This module defines the LSTM neural network model and training procedures
for predicting cryptocurrency volatility.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import pickle
from datetime import datetime
import gpytorch as gp
from gpytorch.means import Mean

class LSTMGeneralized_Output_SameSize(Mean):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                  dropout: float, output_size: int):
        super(LSTMGeneralized_Output_SameSize, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

    def provide_embedding(self, x):
       output, (h_n, c_n) = self.lstm(x)
       return h_n[-1]  # Return the last hidden state of the last LSTM
              

class LSTMPreTrainedMeanFunction(Mean):
  def __init__(self, pretrained_model: nn.Module, device: torch.device = torch.device('cpu')):
      """Transforms a pre-trained LSTM model into a GPyTorch mean function.

      Args:
          pretrained_model (nn.Module): A pre-trained LSTM model.
          device (torch.device, optional): The device to run the model on. Defaults to torch.device('cpu').
      """
      super(LSTMPreTrainedMeanFunction, self).__init__()
      self.device = device
      self.pretrained_model = pretrained_model.to(device)
      for param in self.pretrained_model.parameters():
          param.requires_grad = False
    
  def forward(self, x):
      self.pretrained_model.eval()
      with torch.no_grad():
         output = self.pretrained_model(x.to(self.device))
      return output
   

class LSTMMean_function(Mean):

    def __init__(self, input_shape: tuple[int], hidden_size: list[int], 
                 num_layers: list[int], dropout: list[float], output_dimension: int, device: torch.device = torch.device('cpu')):
        super(LSTMMean_function, self).__init__()
        self.input_shape = input_shape
        self.sizes = hidden_size.copy()
        self.sizes.insert(0, input_shape[0])
        self.step_size = input_shape[1]
        self.device = device


        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size = self.sizes[i], 
                    hidden_size = self.sizes[i + 1], 
                    num_layers = 1, batch_first = True,
                    dropout = dropout[i]) 
                    for i in range(len(hidden_size) - 1)])
        self.fc_layer = nn.Linear(self.step_size * self.input_shape[-1], output_dimension)

    def forward(self, x):
      batch_size, seq_len, _ = x.size()
      i = 0
      for layer in self.hidden_layers:
        if i == 0:
           h_i = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(self.device)
           c_i = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(self.device)
           hidden_i = (h_i, c_i)
           x, hidden_i = layer(x, hidden_i)
        else:
            x, hidden_i = layer(x, hidden_i)
        i += 1
      x = x.contiguous().view(batch_size,-1)
      return self.fc_layer(x).unsqueeze(1)
