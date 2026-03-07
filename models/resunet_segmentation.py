# models/resunet_segmentation.py

import torch
import torch.nn as nn
from .resunet_blocks import ResBlock


class ResUNetSegmentation(nn.Module):
    """
    ResU-Net 3D para segmentación supervisada de arterias coronarias.
    Encoder (e1-e4) carga pesos del preentrenamiento auto-supervisado.
    Decoder con skip connections se inicializa desde cero.
    """
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()

        # --- ENCODER — misma arquitectura que ResUNetEncoder ---
        self.e1   = ResBlock(in_channels, base)      # 1  → 32
        self.e2   = ResBlock(base,        base*2)    # 32 → 64
        self.e3   = ResBlock(base*2,      base*4)    # 64 → 128
        self.e4   = ResBlock(base*4,      base*8)    # 128→ 256
        self.pool = nn.MaxPool3d(2)

        # --- DECODER con skip connections ---
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.d3  = ResBlock(base*8,  base*4)   # cat(up3, e3) = 128+128 = 256 → 128

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.d2  = ResBlock(base*4,  base*2)   # cat(up2, e2) = 64+64   = 128 → 64

        self.up1 = nn.ConvTranspose3d(base*2, base,   2, 2)
        self.d1  = ResBlock(base*2,  base)     # cat(up1, e1) = 32+32   = 64  → 32

        # Capa de salida — sin activación (la aplica la loss)
        self.out = nn.Conv3d(base, out_channels, 1)

    def forward(self, x):
        # Encoder — guarda skip connections
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        # Decoder con skip connections
        d3 = self.d3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

    def load_pretrained_encoder(self, checkpoint_path):
        """
        Carga pesos del encoder preentrenado en e1-e4.
        El decoder queda con inicialización aleatoria (kaiming).
        """
        pretrained = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()

        # Solo carga keys que existen en ambos y tienen el mismo shape
        pretrained_dict = {
            k: v for k, v in pretrained.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        loaded_blocks = set(k.split('.')[0] for k in pretrained_dict.keys())
        print(f"   Encoder cargado: {len(pretrained_dict)} tensores")
        print(f"   Bloques: {sorted(loaded_blocks)}")
        print(f"   Decoder: inicialización aleatoria (kaiming)")

    def get_param_groups(self, lr_encoder=1e-5, lr_decoder=1e-4):
        """
        LR diferencial: encoder quieto, decoder activo.
        """
        encoder_params = (
            list(self.e1.parameters()) +
            list(self.e2.parameters()) +
            list(self.e3.parameters()) +
            list(self.e4.parameters())
        )
        decoder_params = (
            list(self.up3.parameters()) +
            list(self.d3.parameters())  +
            list(self.up2.parameters()) +
            list(self.d2.parameters())  +
            list(self.up1.parameters()) +
            list(self.d1.parameters())  +
            list(self.out.parameters())
        )
        return [
            {'params': encoder_params, 'lr': lr_encoder, 'name': 'encoder'},
            {'params': decoder_params, 'lr': lr_decoder, 'name': 'decoder'},
        ]