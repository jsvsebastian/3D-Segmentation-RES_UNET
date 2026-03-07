# src/dataset_segmentation.py

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class CTASegmentationDataset(Dataset):
    """
    Dataset para segmentación supervisada de arterias coronarias.
    
    Estrategia de muestreo:
    - 50% parches con sesgo positivo: centro garantizado cerca de una arteria
    - 50% parches aleatorios: contexto general y fondo
    
    Esto resuelve el desbalance extremo (0.05% vóxeles coronarios).
    """

    def __init__(
        self,
        vol_dir,
        mask_dir,
        vol_meta_dir,
        mask_meta_dir,
        file_list,
        patch_size=(128, 128, 128),
        patches_per_volume=4,
        positive_ratio=0.5,
    ):
        self.vol_dir       = vol_dir
        self.mask_dir      = mask_dir
        self.vol_meta_dir  = vol_meta_dir
        self.mask_meta_dir = mask_meta_dir
        self.patch_size    = patch_size
        self.ppv           = patches_per_volume
        self.pos_ratio     = positive_ratio

        # Filtrar solo archivos que existen
        self.files = []
        for f in file_list:
            base     = f.replace('.nii.gz', '')
            vol_path = os.path.join(vol_dir,  f"{base}.dat")
            msk_path = os.path.join(mask_dir, f"{base}.dat")
            if os.path.exists(vol_path) and os.path.exists(msk_path):
                self.files.append(base)
            else:
                print(f"⚠️  Par incompleto, ignorado: {base}")

        if len(self.files) == 0:
            raise RuntimeError("No se encontraron pares volumen/máscara.")

        print(f"✅ Dataset segmentación: {len(self.files)} volúmenes")

        # Cargar metadata
        self.vol_meta  = {}
        self.mask_meta = {}
        for base in self.files:
            with open(os.path.join(vol_meta_dir,  f"{base}.json")) as f:
                self.vol_meta[base]  = json.load(f)
            with open(os.path.join(mask_meta_dir, f"{base}.json")) as f:
                self.mask_meta[base] = json.load(f)

        # Precomputar índices de vóxeles positivos por volumen
        # (coordenadas donde la máscara == 1)
        print("Indexando vóxeles coronarios...")
        self.positive_coords = {}
        for base in self.files:
            meta = self.mask_meta[base]
            mmap = np.memmap(
                os.path.join(mask_dir, f"{base}.dat"),
                dtype='float32', mode='r',
                shape=tuple(meta['shape'])
            )
            coords = np.argwhere(mmap > 0.5)  # (N, 3) — z,y,x de cada arteria
            self.positive_coords[base] = coords
            del mmap
        print(f"✅ Indexación completa")

    def __len__(self):
        return len(self.files) * self.ppv

    def _load_volume(self, base):
        meta = self.vol_meta[base]
        return np.memmap(
            os.path.join(self.vol_dir, f"{base}.dat"),
            dtype='float32', mode='r',
            shape=tuple(meta['shape'])
        )

    def _load_mask(self, base):
        meta = self.mask_meta[base]
        return np.memmap(
            os.path.join(self.mask_dir, f"{base}.dat"),
            dtype='float32', mode='r',
            shape=tuple(meta['shape'])
        )

    def _extract_patch(self, vol, mask, z0, y0, x0):
        pz, py, px = self.patch_size
        patch_vol  = np.array(vol [z0:z0+pz, y0:y0+py, x0:x0+px])
        patch_mask = np.array(mask[z0:z0+pz, y0:y0+py, x0:x0+px])

        # Padding si el parche cae en el borde
        if patch_vol.shape != (pz, py, px):
            pad = [(0, max(0, pz - patch_vol.shape[0])),
                   (0, max(0, py - patch_vol.shape[1])),
                   (0, max(0, px - patch_vol.shape[2]))]
            patch_vol  = np.pad(patch_vol,  pad, mode='constant', constant_values=0)
            patch_mask = np.pad(patch_mask, pad, mode='constant', constant_values=0)

        return patch_vol, patch_mask

    def __getitem__(self, idx):
        base = self.files[idx // self.ppv]
        pz, py, px = self.patch_size

        vol  = self._load_volume(base)
        mask = self._load_mask(base)
        shape = vol.shape

        use_positive = (np.random.random() < self.pos_ratio and
                        len(self.positive_coords[base]) > 0)

        if use_positive:
            # Centro del parche = vóxel coronario aleatorio
            coords = self.positive_coords[base]
            center = coords[np.random.randint(len(coords))]
            cz, cy, cx = center

            # Offset aleatorio alrededor del centro (±32 vóxeles)
            z0 = int(np.clip(cz - pz//2 + np.random.randint(-32, 32),
                             0, max(0, shape[0] - pz)))
            y0 = int(np.clip(cy - py//2 + np.random.randint(-32, 32),
                             0, max(0, shape[1] - py)))
            x0 = int(np.clip(cx - px//2 + np.random.randint(-32, 32),
                             0, max(0, shape[2] - px)))
        else:
            # Parche completamente aleatorio
            z0 = np.random.randint(0, max(1, shape[0] - pz + 1))
            y0 = np.random.randint(0, max(1, shape[1] - py + 1))
            x0 = np.random.randint(0, max(1, shape[2] - px + 1))

        patch_vol, patch_mask = self._extract_patch(vol, mask, z0, y0, x0)

        return (
            torch.from_numpy(patch_vol ).float().unsqueeze(0),  # (1,128,128,128)
            torch.from_numpy(patch_mask).float().unsqueeze(0),  # (1,128,128,128)
        )