# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis (TFM VIU 09MIAR) research project on **CT Angiography (CTA) vessel segmentation/reconstruction** using deep learning. The pipeline preprocesses medical NIfTI images and trains 3D neural networks (ResUNet3D, Autoencoders) for self-supervised pretraining on vascular structures.

## Environment Setup

The project uses a virtual environment at `tesisEnv/`. Activate it before running anything:

```bash
source tesisEnv/bin/activate
```

Install dependencies (PyTorch must be installed separately with CUDA support):

```bash
pip install -r requirements.txt
```

## Running Scripts

There are no build or test commands. Scripts are run directly:

```bash
# Data preprocessing pipeline (run in order)
python src/run_filtering.py        # Anisotropic diffusion filtering
python src/run_normalization.py    # HU clipping + z-score normalization
python src/run_frangi.py           # Frangi vesselness filter

# Build memory-mapped dataset for efficient training
python src/build_memmap_dataset.py

# Sanity-check a training loop (10 batches)
python src/sanity_training.py
```

Jupyter notebooks in `notebooks/` are used for experimentation and visualization.

## Data Processing Pipeline

Data flows through these stages in order:

```
data/dataset/          → (metadata extraction)  → data/metadata/
data/dataset/          → (resampling)            → data/processed/resampled/
data/processed/resampled/   → run_filtering.py   → data/processed/filtered/
data/processed/filtered/    → run_normalization.py → data/processed/normalized/
data/processed/normalized/  → run_frangi.py      → data/processed/vesselness/
data/processed/filtered/    → build_memmap_dataset.py → data/processed/memmap/
```

Key preprocessing parameters:
- **HU clipping range**: [-100, 700]
- **Anisotropic diffusion**: 5 iterations, timestep=0.02, conductance=1.5
- **Frangi sigmas**: (0.5, 1.0, 1.5, 2.0)
- **Foreground threshold**: -300 HU

## Model Architectures

All models are in `models/` and take 3D volumetric inputs.

**ResUNet3D** (`models/resunet3d.py`) — main architecture:
- 4-level encoder with ResBlocks + MaxPool3d downsampling
- Symmetric decoder with ConvTranspose3d upsampling + skip connections
- ResBlock: Conv3d → InstanceNorm3d → Conv3d + residual skip
- Configurable `base_channels` (default=32): channels double per level (32→64→128→256→512)

**Pretraining Autoencoder** (`models/autoencoder_pretrain.py`, `models/pretrain_autoencoder.py`):
- Combines `ResUNetEncoder` (`models/resunet_encoder.py`) with `PretrainDecoder` (`models/pretrain_decoder.py`)
- Used for self-supervised pretraining before fine-tuning ResUNet3D

**Simple Autoencoder** (`models/autoencoder3d.py`):
- Lightweight baseline: 3× Conv3d (stride-2) encoder, 3× ConvTranspose3d decoder
- Channels: 1→16→32→64→32→16→1

## Dataset Classes

Two implementations in `src/` for patch-based training:

- **CTAPatchDataset** (`src/dataset_patches.py`): Loads from disk; 128×128×128 patches; 70% foreground / 30% random sampling
- **CTAMemmapDataset** (`src/dataset_patches_memmap.py`): Memory-mapped binary files; 112×112×112 patches; caches last-loaded volume; default 2 patches per volume

Training outputs go to:
- `runs/checkpoints/` — model weights
- `runs/tensorboard/` — TensorBoard logs
- `runs/pretrain/` — pretraining outputs

## Code Conventions

- CUDA is used when available (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`)
- Medical images are loaded as NIfTI (`.nii.gz`) via SimpleITK o nibabel
- Metadata is stored as pickle files in `data/metadata/`
- Memmap metadata is stored as JSON files in `data/processed/memmap/meta/`

## Behavioral Rules (Mandatory — Read Before Any Action)

These rules govern how to operate on this project. Follow them strictly on every task.

### Role
Act as a Senior Machine Learning Engineer and Vision Architecture expert.

### Active Problem Context
The encoder was trained on the full dataset without a train/val/test split, introducing **data leakage**. The goal is to correct the dataset partition and restructure the pipeline. The required split is:
- **70%** Training
- **15%** Validation
- **15%** Test/Evaluation

Both dataset classes (`imagecas` and `asoca`) must be split independently.

### Mandatory Protocol

1. **Analysis first**: Before writing any code or modifying any file, fully analyze the relevant project structure and files in the current workspace.

2. **Diagnosis report**: After analysis, deliver a written report with findings about the current pipeline and concrete recommendations for reverting the leakage error and implementing the new split.

3. **Doubts checkpoint**: Before making any physical change to files, present an explicit list of open questions about folder structure, data formats, or ambiguous requirements, along with proactive improvement suggestions. Wait for the user to resolve them.

4. **No autonomous action**: Do not run terminal commands or modify files without the user's explicit confirmation after the doubts checkpoint has been resolved.

5. **Split integrity**: Ensure the split is stratified where appropriate and that validation and test sets are fully isolated from any training step — no leakage of any kind.
