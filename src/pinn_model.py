import torch
import torch.nn as nn

class HeatPINN(nn.Module):
    def __init__(self, in_dim=6, out_dim=1, width=64, depth=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        for _ in range(depth - 1):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(width, width))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):

        return self.net(X)  # (N,1)
