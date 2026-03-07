import torch.nn as nn
from .resunet_encoder import ResUNetEncoder
from .pretrain_decoder import PretrainDecoder


class ResUNetAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ResUNetEncoder()
        self.dec = PretrainDecoder()

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
