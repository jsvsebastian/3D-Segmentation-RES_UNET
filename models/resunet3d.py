
import torch
import torch.nn as nn
from .resunet_blocks import ResBlock

class ResUNetSegmentation(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()

        # ENCODER — misma arquitectura que el preentrenado
        self.e1 = ResBlock(in_channels, base)
        self.e2 = ResBlock(base, base*2)
        self.e3 = ResBlock(base*2, base*4)
        self.e4 = ResBlock(base*4, base*8)
        self.pool = nn.MaxPool3d(2)

        # DECODER con skip connections — nuevo
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.d3  = ResBlock(base*8, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.d2  = ResBlock(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.d1  = ResBlock(base*2, base)

        self.out = nn.Conv3d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        d3 = self.d3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

    def load_pretrained_encoder(self, checkpoint_path):
        pretrained = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        loaded = list(pretrained_dict.keys())
        print(f" Pesos cargados: {len(loaded)} tensores")
        print(f" Bloques: {set(k.split('.')[0] for k in loaded)}")

    def get_param_groups(self, lr_encoder=1e-5, lr_decoder=1e-4):
        encoder_params = list(self.e1.parameters()) + \
                         list(self.e2.parameters()) + \
                         list(self.e3.parameters()) + \
                         list(self.e4.parameters())

        decoder_params = list(self.up3.parameters()) + \
                         list(self.d3.parameters())  + \
                         list(self.up2.parameters()) + \
                         list(self.d2.parameters())  + \
                         list(self.up1.parameters()) + \
                         list(self.d1.parameters())  + \
                         list(self.out.parameters())

        return [
            {'params': encoder_params, 'lr': lr_encoder},
            {'params': decoder_params, 'lr': lr_decoder}
        ]