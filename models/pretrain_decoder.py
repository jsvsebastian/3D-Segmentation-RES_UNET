import torch.nn as nn

class PretrainDecoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return self.out(x)
