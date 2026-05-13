"""
Generate uncertainty heatmaps using Test-Time Augmentation (TTA).

Runs inference N times with random flips (x, y, z axes), averages the predictions,
then computes per-voxel standard deviation as a proxy for uncertainty.
High std = model is unsure → typically at tumour boundaries.

Outputs (in docs/figures/):
  uncertainty_mean_pred.png    — mean prediction overlay (averaged over TTA runs)
  uncertainty_heatmap.png      — uncertainty (std dev) overlaid on FLAIR slice
  uncertainty_combined.png     — 3-panel: FLAIR | mean pred | uncertainty

Usage:
  python scripts/visualize_uncertainty.py
  python scripts/visualize_uncertainty.py --model attention_unet --n-aug 8
  python scripts/visualize_uncertainty.py --case BraTS2021_00698 --slice 77
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monai.inferers import sliding_window_inference
from src.models.unet3d import UNet3D
from src.models.attention_unet3d import AttentionUNet3D
from src.models.swin_unetr import build_swin_unetr
from src.models.kan_unet3d import KANUNet3d
from scripts.visualize_segmentation import (
    get_val_cases, pick_best_case, best_axial_slice,
    make_overlay, REGION_COLORS, NPY_DIR, OUT_DIR,
)

CHECKPOINTS = {
    "attention_unet": ROOT / "checkpoints/attention_unet/best_model.pt",
    "unet3d":         ROOT / "checkpoints/unet3d/best_model.pt",
    "swin_unetr":     ROOT / "checkpoints/swin_unetr/best_model.pt",
    "kan_unet":       ROOT / "checkpoints/kan_unet3d/best_model.pt",
}

_MODEL_FACTORIES = {
    "UNet3D": UNet3D,
    "AttentionUNet3D": AttentionUNet3D,
    "SwinUNETR": build_swin_unetr,
    "KANUNet3d": KANUNet3d,
}

PATCH_SIZE = (128, 128, 128)
OVERLAP    = 0.5

# All axis-flip combinations for TTA (excluding no-flip = original)
TTA_FLIPS = [
    [],
    [2], [3], [4],
    [2, 3], [2, 4], [3, 4],
    [2, 3, 4],
]  # axes are 1-indexed for the batch dim — actual spatial axes are 2,3,4


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]["model"]
    name = cfg["name"]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    model = _MODEL_FACTORIES[name](**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def tta_inference(model, image: np.ndarray, device: torch.device,
                  n_aug: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Run TTA inference. Returns:
      mean_probs : (3, H, W, D) float32 — averaged sigmoid probabilities
      std_probs  : (3, H, W, D) float32 — std deviation (uncertainty)
    """
    x_base = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, 4, H, W, D)

    aug_list = TTA_FLIPS[:n_aug]
    all_probs = []

    for i, flip_axes in enumerate(aug_list):
        x = x_base.clone()
        for ax in flip_axes:
            x = torch.flip(x, dims=[ax])
        x = x.to(device)

        with torch.no_grad():
            logits = sliding_window_inference(
                x, roi_size=PATCH_SIZE, sw_batch_size=1,
                predictor=model, overlap=OVERLAP, mode="gaussian",
            )

        probs = torch.sigmoid(logits).cpu()

        # Undo flips
        for ax in reversed(flip_axes):
            probs = torch.flip(probs, dims=[ax])

        all_probs.append(probs.squeeze(0).numpy())
        print(f"  TTA {i+1}/{len(aug_list)} done")

    stack = np.stack(all_probs, axis=0)  # (N, 3, H, W, D)
    return stack.mean(axis=0), stack.std(axis=0)


def save_uncertainty_figures(flair_slice, mean_probs_slice, std_slice,
                              gt_mask_slice, slice_idx, model_name, out_dir):
    """
    flair_slice      : (H, W)
    mean_probs_slice : (3, H, W) — averaged probabilities
    std_slice        : (3, H, W) — per-channel std deviation
    gt_mask_slice    : (3, H, W) — ground truth binary
    """
    mean_mask = (mean_probs_slice > 0.5).astype(np.uint8)
    # Collapse std across channels → single uncertainty map
    uncertainty = std_slice.max(axis=0)  # (H, W) — worst-case uncertainty per voxel

    vmin, vmax = flair_slice.min(), flair_slice.max()
    gray = (flair_slice - vmin) / (vmax - vmin + 1e-8)

    # --- 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=140)

    axes[0].imshow(make_overlay(flair_slice, gt_mask_slice), origin="lower")
    axes[0].set_title("Ground Truth", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(make_overlay(flair_slice, mean_mask), origin="lower")
    axes[1].set_title(f"{model_name} — Mean Prediction (TTA)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(gray, cmap="gray", origin="lower")
    unc_img = axes[2].imshow(uncertainty, cmap="YlOrRd", alpha=0.7,
                              vmin=0, vmax=uncertainty.max(), origin="lower")
    axes[2].set_title("Uncertainty (TTA std dev)", fontsize=10, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(unc_img, ax=axes[2], fraction=0.046, pad=0.04, label="Std deviation")

    patches = [
        mpatches.Patch(color=REGION_COLORS[0], label="NCR"),
        mpatches.Patch(color=REGION_COLORS[1], label="ED"),
        mpatches.Patch(color=REGION_COLORS[2], label="ET"),
    ]
    axes[1].legend(handles=patches, loc="lower right", fontsize=8,
                   framealpha=0.6, edgecolor="none")

    plt.suptitle(f"Uncertainty Heatmap — {model_name}  (axial slice {slice_idx})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"uncertainty_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",  default="attention_unet",
                        choices=list(CHECKPOINTS.keys()),
                        help="Which model checkpoint to use")
    parser.add_argument("--case",   default=None)
    parser.add_argument("--slice",  type=int, default=None)
    parser.add_argument("--n-aug",  type=int, default=8,
                        help="Number of TTA augmentations (max 8)")
    args = parser.parse_args()

    ckpt_path = CHECKPOINTS[args.model]
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(ckpt_path, device)
    model_display = args.model.replace("_", " ").title()

    val_cases = get_val_cases()
    case_id   = args.case or pick_best_case(val_cases)

    image = np.load(NPY_DIR / f"{case_id}_image.npy").astype(np.float32)
    label = np.load(NPY_DIR / f"{case_id}_label.npy").astype(np.float32)

    slice_idx = args.slice if args.slice is not None else best_axial_slice(label)
    print(f"Case: {case_id}  |  Slice: {slice_idx}  |  TTA runs: {args.n_aug}")

    print("\nRunning TTA inference...")
    mean_probs, std_probs = tta_inference(model, image, device, n_aug=args.n_aug)

    flair_slice      = image[0, :, :, slice_idx]
    mean_probs_slice = mean_probs[:, :, :, slice_idx]
    std_slice        = std_probs[:, :, :, slice_idx]
    gt_mask_slice    = label[:, :, :, slice_idx].astype(np.uint8)

    save_uncertainty_figures(
        flair_slice, mean_probs_slice, std_slice, gt_mask_slice,
        slice_idx, model_display, OUT_DIR,
    )

    print(f"\nUncertainty figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
