# BraTS2021 Brain Tumour Segmentation

Five architectures trained and compared on BraTS2021. All on a GTX 1070 Ti (8GB VRAM) — the memory ceiling shaped most of the decisions: patch-based training, fp16, numpy cache to avoid re-loading NIfTI every epoch.

**Full writeup and results:** https://stefanstrain.github.io/projects/mri_segmentation/

| Model | WT | TC | ET | Mean | Params |
|---|---|---|---|---|---|
| Attention U-Net | 0.886 | 0.884 | 0.875 | 0.882 | 22.66M |
| KAN 3D U-Net | 0.879 | 0.885 | 0.869 | 0.878 | 22.59M |
| 3D U-Net | 0.876 | 0.877 | 0.869 | 0.874 | 22.58M |
| Swin UNETR | 0.882 | 0.863 | 0.862 | 0.869 | 62.19M |
| KAN U-Net | 0.878 | 0.873 | 0.856 | 0.869 | 2.42M |

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
