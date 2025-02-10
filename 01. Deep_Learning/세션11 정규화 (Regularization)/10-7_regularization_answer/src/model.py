import torch
from torch import nn


# 위 방법처럼도 구현할 수 있으나, 더 간단하게 nn.Sequential로 구현할 수 있다.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Neural Network을 구성하는 layer들을
        # initialize하는 부분
        self.fc_layers = nn.Sequential(
            nn.Linear(784, 784 // 4),
            nn.ReLU(),
            nn.Linear(784 // 4, 784 // 16),
            nn.ReLU(),
            nn.Linear(784 // 16, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Neural Network의 forward pass을 정의하는 부분
        # x은 input tensor
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x
