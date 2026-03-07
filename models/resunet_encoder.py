import torch.nn as nn
from .resunet_blocks import ResBlock

class ResUNetEncoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1 = ResBlock(1, base)
        self.e2 = ResBlock(base, base*2)
        self.e3 = ResBlock(base*2, base*4)
        self.e4 = ResBlock(base*4, base*8)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(self.pool(x))
        x = self.e3(self.pool(x))
        x = self.e4(self.pool(x))
        return x
