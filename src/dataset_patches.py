import os
import random
import torch
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset, DataLoader


PATCH_SIZE = 128
INPUT_DIR = "/media/mrsmile/IA/tesis/data/processed/normalized"


def foreground_bbox(volume, threshold=-300):
    """Bounding box del cuerpo usando HU"""
    mask = volume > threshold
    coords = np.where(mask)
    zmin, ymin, xmin = coords[0].min(), coords[1].min(), coords[2].min()
    zmax, ymax, xmax = coords[0].max(), coords[1].max(), coords[2].max()
    return zmin, zmax, ymin, ymax, xmin, xmax


def random_patch(volume, patch_size):
    """Extrae un patch aleatorio"""
    D, H, W = volume.shape
    z = random.randint(0, D - patch_size)
    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)
    return volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]


def foreground_patch(volume, patch_size):
    """Patch centrado en región con anatomía"""
    zmin, zmax, ymin, ymax, xmin, xmax = foreground_bbox(volume)
    zc = random.randint(zmin, max(zmin, zmax - patch_size))
    yc = random.randint(ymin, max(ymin, ymax - patch_size))
    xc = random.randint(xmin, max(xmin, xmax - patch_size))
    return volume[zc:zc+patch_size, yc:yc+patch_size, xc:xc+patch_size]



class CTAPatchDataset(Dataset):

    def __init__(self, input_dir, patches_per_volume=4):
        self.paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".nii.gz")
        ])
        self.patches_per_volume = patches_per_volume

    def __len__(self):
        return len(self.paths) * self.patches_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        path = self.paths[vol_idx]

        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)

      
        if random.random() < 0.7:
            patch = foreground_patch(vol, PATCH_SIZE)
        else:
            patch = random_patch(vol, PATCH_SIZE)


        patch = torch.from_numpy(patch).unsqueeze(0)

        return patch
