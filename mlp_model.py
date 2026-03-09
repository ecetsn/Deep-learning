import torch
import torch.nn as nn
from typing import List
from parameters import ModelConfig


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) for MNIST classification.

    This model supports configurable hidden layers, activation functions, 
    dropout, and batch normalization placement.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the MLP model.

        Args:
            config (ModelConfig): Configuration object containing model hyperparameters.
        """
        super(MLP, self).__init__()
        self.config = config
        
        # Flatten layer to convert 28x28 images to 784-dimensional vectors
        self.flatten = nn.Flatten()
        
        layers = []
        input_dim = config.input_size
        self.layers = nn.ModuleList()
        
        for hidden_dim in config.hidden_sizes:
            layer_block = []
            
            # Fully connected layer
            layer_block.append(nn.Linear(input_dim, hidden_dim))
            
            # BatchNorm placement
            if config.use_batchnorm and config.bn_before_activation:
                layer_block.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            act = config.activation.lower()
            if act == "relu":
                layer_block.append(nn.ReLU())
            elif act == "gelu":
                layer_block.append(nn.GELU())
            elif act == "sigmoid":
                layer_block.append(nn.Sigmoid())
            elif act == "tanh":
                layer_block.append(nn.Tanh())
            elif act == "leaky_relu":
                layer_block.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation: {config.activation}")
            
            if config.use_batchnorm and not config.bn_before_activation:
                layer_block.append(nn.BatchNorm1d(hidden_dim))
                
            # Dropout
            if config.dropout > 0:
                layer_block.append(nn.Dropout(config.dropout))
                
            self.layers.append(nn.Sequential(*layer_block))
            input_dim = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(input_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28) or similar.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

