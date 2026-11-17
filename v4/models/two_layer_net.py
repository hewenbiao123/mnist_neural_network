""" Define a two layer neural network for MNIST """

import torch.nn as nn 
from torch import Tensor
from config import MNISTConfig

class TwoLayerNet(nn.Module):
    def __init__(self, config: MNISTConfig) -> None:
        super().__init__()
        # Define hidden layer
        #   1. Affine (input size, output size): W and B will be regarded as nn.Parameter
        #   2. Activation Function
        self.hidden_affine = nn.Linear(config.input_size, config.hidden_size)
        self.hidden_relu = nn.ReLU()
        # Define ouput layer
        self.output_affine = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, X: Tensor) -> Tensor:
        # Expand data into one dimension
        X = X.view(X.size(0), -1) # (batch, 1, 28, 28) -> (batch, 28 * 28)
        X = self.hidden_relu.forward(self.hidden_affine.forward(X))
        return self.output_affine.forward(X) # return logits



if __name__ == "__main__":
    import torch
    net = TwoLayerNet(MNISTConfig())
    print(net.forward(torch.randn(8, 1, 28, 28)).shape) # (batch, 10)