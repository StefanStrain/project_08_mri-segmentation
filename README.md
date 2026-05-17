# BraTS2021 Brain Tumour Segmentation

3D volumetric segmentation of brain tumours from multi-modal MRI, trained and evaluated on 150/1,251 BraTS2021 cases. I implemented and compared five architectures, from a 3D U-Net baseline through to a Swin Transformer and two KAN-based variants.

The segmentation covers three clinically important regions derived from four MRI modalities (FLAIR, T1, T1CE, T2), including whole tumour (WT), tumour core (TC), and enhancing tumour (ET). These regions aren't cleanly nested. They're defined by overlapping label combinations, and the class imbalance is severe, with tumour voxels making up a small fraction of each 240×240×155 volume.

All training ran on a GTX 1070 Ti (8GB VRAM), so the memory limit shaped most of the decisions including a patch-based training on 128x128x128 crops, fp16 throughout and a numpy cache to avoid re-loading NIfTI files every epoch. Every run is tracked with MLflow, logging loss curves, validation Dice per region, and hyperparameters.

The most novel part of the project is the KAN experiments. [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) replace the linear projections in the U-Net bottleneck and skip connections with learnable spline activations. I tested a partial replacement (KAN U-Net, 2.42M params) and a full replacement (KAN 3D U-Net, 22.59M params). Both matched the baseline, and the lightweight KAN U-Net matched the 62M-parameter Transformer at 26× fewer parameters.

> **Full writeup, training curves, and interactive results:**
> ### [stefanstrain.github.io/projects/mri_segmentation](https://stefanstrain.github.io/projects/mri_segmentation/)

## Results

| Model | WT | TC | ET | Mean | Params |
|---|---|---|---|---|---|
| Attention U-Net | 0.886 | 0.884 | 0.875 | 0.882 | 22.66M |
| KAN 3D U-Net | 0.879 | 0.885 | 0.869 | 0.878 | 22.59M |
| 3D U-Net | 0.876 | 0.877 | 0.869 | 0.874 | 22.58M |
| Swin UNETR | 0.882 | 0.863 | 0.862 | 0.869 | 62.19M |
| KAN U-Net | 0.878 | 0.873 | 0.856 | 0.869 | 2.42M |

Dice scores on a held-out 50-case validation split. WT/TC/ET are the standard BraTS evaluation regions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Preprocess NIfTI files to numpy cache (run once before training)
python scripts/preprocess_npy.py

# Train a model
python train_model.py --config configs/baseline.yaml

# Evaluate full-volume Dice
python src/inference.py --checkpoint checkpoints/unet3d/best_model.pt
```

## Experiment tracking

Every training run logs to MLflow automatically. To browse runs, loss curves and Dice across all experiments:

```bash
mlflow ui
```

Then open `http://localhost:5000`. 
(`scripts/plot_charts.py` script pulls from the same MLflow data to generate the interactive charts on the portfolio page)

## Structure

```
src/
  models/       # UNet3D, AttentionUNet3D, SwinUNETR, KANUNet3d, KANUNet3dFull
  dataset.py    # NpyDataset + DataLoader setup
  transforms.py # MONAI preprocessing and augmentation pipelines
  trainer.py    # training loop with MLflow logging
  inference.py  # sliding-window full-volume evaluation
  losses.py     # BCEDiceLoss, DiceLoss, FocalDiceLoss
  metrics.py    # Dice and HD95 per BraTS region
scripts/
  preprocess_npy.py     # one-time NIfTI to numpy conversion
  evaluate.py           # full-volume evaluation for any checkpoint
  plot_charts.py        # generate portfolio Plotly charts from MLflow data
  visualize_*.py        # visualisation scripts (generated with Claude)
configs/                # YAML configs for each model
notebooks/              # EDA and results visualisation
```
