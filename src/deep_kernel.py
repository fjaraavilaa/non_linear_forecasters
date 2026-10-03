"""
This module implements the feature extractors for the Deep Kernel SVGP model.
it also takes care of the lstm of the model

author: @fjaraavila
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel

class PretrainedEmbeddingExtractor(torch.nn.Module):
    """Use a frozen pretrained LSTM hidden state as the GP kernel feature."""

    def __init__(self, pretrained_model: torch.nn.Module):
        super().__init__()
        self.pretrained_model = pretrained_model
        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            _, (hidden_state, _) = self.pretrained_model.lstm(x)
        return hidden_state[-1]


class lstm_extractor_same_size_layers(torch.nn.Module): 
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int, output_size: int, 
                 dropout: float = 0.00):
        """Provides the feature extractor for the Deep Kernel SVGP model. 
        It consists of an LSTM with the same hidden size for all layers, followed by a 
        fully connected layer to produce the final output features.

        Args:
            input_size (int): Size of lstm input
            hidden_size (int): Size of all hidden layers
            num_layers (int): Number of layers
            output_size (int): Size of the output from the sequence
            dropout (float, optional): dropout probability in lstm. layers. Defaults to 0.00.
        """
        super(lstm_extractor_same_size_layers, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        #self.activation_fc = torch.nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the LSTM feature extractor. Returns the last hidden state of the last LSTM layer.
        """
        output, (h_n, c_n) = self.lstm(x)
        #out_features = self.fc(h_n[-1])
        #out_features = self.activation_fc(out_features)
        return h_n[-1]

class lstm_extractor_different_size_layers(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, 
                 num_layers: int, output_size: int, dropout: float = 0.00):
        super(lstm_extractor_different_size_layers, self).__init__()

        self.hidden_sizes = hidden_sizes.copy()
        self.sizes = hidden_sizes.copy()
        self.sizes.insert(0, input_size)
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=self.sizes[i], hidden_size=self.sizes[i + 1], num_layers=1, batch_first=True) for i in range(num_layers)
        ])
        self.fc = torch.nn.Linear(self.sizes[-1], output_size)

    def forward(self, x) -> torch.Tensor:
        i = 0
        for layer in self.lstm_layers:
            if i == 0:
                x, all_hidden = layer(x)
                print(f"Layer {i} output shape: {x.shape}")
                print(f"Layer {i} hidden state shape: {all_hidden[0].shape}")
            else:
                x, all_hidden = layer(x)

                h_n, c_n = all_hidden
                
                print(f"Layer {i} hidden state shape: {h_n.shape}")
            #print(h_n, c_n)
            i += 1
        print(h_n[-1].shape)
        return h_n[-1]


