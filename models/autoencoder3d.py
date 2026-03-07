import torch
import torch.nn as nn

class AutoEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
