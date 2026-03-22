import torch
import torch.nn as nn

class OilPriceNN(nn.Module):
    """
    Multi-Layer Perceptron for Oil Price Prediction.
    Architecture: Input -> 64 -> 32 -> Output (1)
    """
    def __init__(self, input_size):
        super(OilPriceNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
