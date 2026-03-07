import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_patches import CTAPatchDataset
from models.resunet3d import ResUNet3D


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCHES = 10
LR = 1e-4


dataset = CTAPatchDataset(
    input_dir="/media/mrsmile/IA/tesis/data/processed/normalized",
    patches_per_volume=1
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


model = ResUNet3D(in_channels=1, out_channels=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# fake loss (solo sanity)
criterion = nn.MSELoss()


model.train()

for i, x in enumerate(loader):
    if i >= BATCHES:
        break

    x = x.to(DEVICE)

    # fake target = input (autoencoder-like)
    y = x.clone()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    print(f"[{i}] loss = {loss.item():.4f}")
