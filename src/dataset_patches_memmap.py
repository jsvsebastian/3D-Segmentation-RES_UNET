import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class CTAMemmapDataset(Dataset):
    def __init__(self, root, patch_size=(128, 128, 128), patches_per_volume=2, file_list=None):
        self.root = root
        self.vol_dir = os.path.join(root, "volumes")
        self.meta_dir = os.path.join(root, "meta")
        self.patch_size = patch_size
        self.ppv = patches_per_volume

        if file_list is not None:
            self.files = [
                f"{f}.dat" for f in file_list
                if os.path.exists(os.path.join(self.vol_dir, f"{f}.dat"))
            ]
        else:
            self.files = sorted([
                f for f in os.listdir(self.vol_dir) if f.endswith(".dat")
            ])

        if len(self.files) == 0:
            raise RuntimeError(f"No se encontraron archivos .dat en {self.vol_dir}")

        self.meta = {}
        for f in self.files:
            meta_path = os.path.join(self.meta_dir, f.replace(".dat", ".json"))
            if os.path.exists(meta_path):
                with open(meta_path, "r") as jf:
                    self.meta[f] = json.load(jf)
            else:
                raise FileNotFoundError(f"No se encontró el archivo meta: {meta_path}")

    def __len__(self):
        return len(self.files) * self.ppv

    def _load_volume(self, f):
        meta = self.meta[f]
        return np.memmap(
            os.path.join(self.vol_dir, f),
            dtype=np.dtype(meta.get("dtype", "float32")),
            mode="r",
            shape=tuple(meta["shape"])
        )

    def __getitem__(self, idx):
        vol_idx = idx // self.ppv
        f = self.files[vol_idx]
        vol = self._load_volume(f)

        shape = vol.shape
        pz, py, px = self.patch_size

        z0 = np.random.randint(0, max(1, shape[0] - pz + 1))
        y0 = np.random.randint(0, max(1, shape[1] - py + 1))
        x0 = np.random.randint(0, max(1, shape[2] - px + 1))

        patch = np.array(vol[z0:z0+pz, y0:y0+py, x0:x0+px])

        if patch.shape != (pz, py, px):
            patch = np.pad(patch, [
                (0, max(0, pz - patch.shape[0])),
                (0, max(0, py - patch.shape[1])),
                (0, max(0, px - patch.shape[2]))
            ], mode='constant', constant_values=0)

        return torch.from_numpy(patch).float().unsqueeze(0)