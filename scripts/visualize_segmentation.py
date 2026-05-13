"""
Generate segmentation overlay PNGs and 3D tumour mesh for the portfolio.

Outputs (in docs/figures/):
  overlay_gt.png              — ground truth mask overlay on FLAIR
  overlay_unet3d.png          — 3D U-Net prediction overlay
  overlay_attention_unet.png  — Attention U-Net prediction overlay
  overlay_swin_unetr.png      — Swin UNETR prediction overlay
  overlay_kan_unet.png        — KAN U-Net prediction overlay
  comparison_all_models.png   — side-by-side all models on one figure
  mesh_ncr.obj                — 3D mesh of necrotic core (GT)
  mesh_ed.obj                 — 3D mesh of oedema (GT)
  mesh_et.obj                 — 3D mesh of enhancing tumour (GT)

Usage:
  python scripts/visualize_segmentation.py
  python scripts/visualize_segmentation.py --case BraTS2021_00698
  python scripts/visualize_segmentation.py --case BraTS2021_00698 --slice 77
  python scripts/visualize_segmentation.py --list-cases   # show top-10 largest tumours
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monai.inferers import sliding_window_inference
from src.models.unet3d import UNet3D
from src.models.attention_unet3d import AttentionUNet3D
from src.models.swin_unetr import build_swin_unetr
from src.models.kan_unet3d import KANUNet3d
from src.models.kan_unet3d_full import KANUNet3dFull

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "3D U-Net":        ROOT / "checkpoints/unet3d/best_model.pt",
    "Attention U-Net": ROOT / "checkpoints/attention_unet/best_model.pt",
    "Swin UNETR":      ROOT / "checkpoints/swin_unetr/best_model.pt",
    "KAN U-Net":       ROOT / "checkpoints/kan_unet3d/best_model.pt",
    "KAN 3D U-Net":    ROOT / "checkpoints/unet3d_kan/best_model.pt",
}

_MODEL_FACTORIES = {
    "UNet3D":        UNet3D,
    "AttentionUNet3D": AttentionUNet3D,
    "SwinUNETR":     build_swin_unetr,
    "KANUNet3d":     KANUNet3d,
    "KANUNet3dFull": KANUNet3dFull,
}

# BraTS colour scheme: NCR=red, ED=yellow, ET=cyan
REGION_COLORS = np.array([
    [1.00, 0.20, 0.20],   # NCR — red
    [1.00, 0.90, 0.10],   # ED  — yellow
    [0.10, 0.85, 1.00],   # ET  — cyan
], dtype=np.float32)

OVERLAY_ALPHA = 0.45
PATCH_SIZE = (128, 128, 128)
SW_OVERLAP  = 0.5

NPY_DIR    = ROOT / "cache_npy"
SPLITS     = ROOT / "data/splits.json"
OUT_DIR    = ROOT / "docs/figures"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]["model"]
    name = cfg["name"]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    model = _MODEL_FACTORIES[name](**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def tumour_voxel_count(case_id: str) -> int:
    lbl_path = NPY_DIR / f"{case_id}_label.npy"
    if not lbl_path.exists():
        return 0
    label = np.load(lbl_path)
    return int(label.max(axis=0).sum())


def get_val_cases() -> list[str]:
    splits = json.loads(SPLITS.read_text())
    val_ids = splits["val"]
    rng = random.Random(43)
    return rng.sample(val_ids, min(50, len(val_ids)))


def pick_best_case(val_cases: list[str]) -> str:
    """Return the val case with the largest total tumour volume."""
    print("Scanning val cases for tumour size...")
    counts = {c: tumour_voxel_count(c) for c in val_cases}
    best = max(counts, key=counts.get)
    print(f"  Selected {best} ({counts[best]:,} tumour voxels)")
    return best


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Run sliding-window inference. Returns (3, H, W, D) logits on CPU."""
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            x, roi_size=PATCH_SIZE, sw_batch_size=1,
            predictor=model, overlap=SW_OVERLAP, mode="gaussian",
        )
    return logits.squeeze(0).cpu().numpy()


def logits_to_binary(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid + threshold → (3, H, W, D) binary mask."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------

def best_axial_slice(label: np.ndarray) -> int:
    """Return the axial slice index (D axis) with the most tumour voxels."""
    # label: (3, H, W, D)
    per_slice = label.max(axis=0).sum(axis=(0, 1))  # (D,)
    return int(per_slice.argmax())


# ---------------------------------------------------------------------------
# Overlay generation
# ---------------------------------------------------------------------------

def make_overlay(flair_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    """
    flair_slice : (H, W) float  — normalised MRI intensity
    mask_slice  : (3, H, W) uint8 — channels are TC / WT / ET
                  (MONAI ConvertToMultiChannelBasedOnBratsClassesd convention)
    Returns     : (H, W, 3) float32 RGB image coloured as NCR / ED / ET
    """
    vmin, vmax = flair_slice.min(), flair_slice.max()
    gray = (flair_slice - vmin) / (vmax - vmin + 1e-8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # Decode TC/WT/ET → disjoint NCR/ED/ET for colouring
    tc = mask_slice[0].astype(bool)  # NCR + ET  (labels 1+4)
    wt = mask_slice[1].astype(bool)  # NCR+ED+ET (labels 1+2+4)
    et = mask_slice[2].astype(bool)  # ET only   (label 4)
    decoded = [
        tc & ~et,   # NCR = TC minus ET  → red
        wt & ~tc,   # ED  = WT minus TC  → yellow
        et,         # ET                 → cyan
    ]

    overlay = rgb.copy()
    for mask, color in zip(decoded, REGION_COLORS):
        if not mask.any():
            continue
        for ch in range(3):
            overlay[:, :, ch] = np.where(
                mask,
                OVERLAY_ALPHA * color[ch] + (1 - OVERLAY_ALPHA) * rgb[:, :, ch],
                overlay[:, :, ch],
            )
    return np.clip(overlay, 0, 1)


def save_overlay(flair_slice, mask_slice, title, path, show_legend=True):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(make_overlay(flair_slice, mask_slice), origin="lower")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")

    if show_legend:
        patches = [
            mpatches.Patch(color=REGION_COLORS[0], label="NCR"),
            mpatches.Patch(color=REGION_COLORS[1], label="ED"),
            mpatches.Patch(color=REGION_COLORS[2], label="ET"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8,
                  framealpha=0.6, edgecolor="none")

    plt.tight_layout(pad=0.3)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def save_comparison(flair_slice, gt_mask, predictions: dict, slice_idx: int, path: Path):
    """
    predictions: {model_name: binary_mask (3, H, W)}
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, 2 + n_models, figsize=(4 * (2 + n_models), 4.5), dpi=150)

    # FLAIR raw
    vmin, vmax = flair_slice.min(), flair_slice.max()
    axes[0].imshow(
        (flair_slice - vmin) / (vmax - vmin + 1e-8),
        cmap="gray", origin="lower",
    )
    axes[0].set_title("FLAIR", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(make_overlay(flair_slice, gt_mask), origin="lower")
    axes[1].set_title("Ground Truth", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Per-model predictions
    for i, (name, mask) in enumerate(predictions.items()):
        axes[2 + i].imshow(make_overlay(flair_slice, mask), origin="lower")
        axes[2 + i].set_title(name, fontsize=10, fontweight="bold")
        axes[2 + i].axis("off")

    # Shared legend
    patches = [
        mpatches.Patch(color=REGION_COLORS[0], label="NCR (Necrotic Core)"),
        mpatches.Patch(color=REGION_COLORS[1], label="ED (Oedema)"),
        mpatches.Patch(color=REGION_COLORS[2], label="ET (Enhancing Tumour)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.7, edgecolor="none", bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Axial slice {slice_idx} — all models", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# 3D mesh generation
# ---------------------------------------------------------------------------

def save_obj_mesh(vertices, faces, path: Path):
    """Write a simple Wavefront OBJ file from marching cubes output."""
    with open(path, "w") as f:
        f.write(f"# Generated by visualize_segmentation.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def generate_meshes(label: np.ndarray, out_dir: Path):
    """Generate one OBJ mesh per tumour subregion from the GT label."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("  scikit-image not found — skipping mesh generation. Install with: pip install scikit-image")
        return

    region_names = ["ncr", "ed", "et"]
    # label: (3, H, W, D)
    for c, name in enumerate(region_names):
        mask = label[c].astype(np.float32)  # (H, W, D)
        if mask.sum() < 100:
            print(f"  Skipping {name.upper()} mesh — too few voxels")
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, step_size=2)
            path = out_dir / f"mesh_{name}.obj"
            save_obj_mesh(verts, faces, path)
            print(f"  Saved {path.name}  ({len(verts):,} verts, {len(faces):,} faces)")
        except Exception as e:
            print(f"  Mesh generation failed for {name}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case",        default=None, help="BraTS2021 case ID to visualise")
    parser.add_argument("--slice",       type=int, default=None, help="Axial slice index (auto if omitted)")
    parser.add_argument("--list-cases",  action="store_true", help="Print top-10 largest tumour cases and exit")
    parser.add_argument("--no-inference",action="store_true", help="Only generate GT overlay and mesh, skip model inference")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_cases = get_val_cases()

    if args.list_cases:
        counts = sorted(
            [(c, tumour_voxel_count(c)) for c in val_cases],
            key=lambda x: -x[1]
        )[:10]
        print("\nTop-10 largest tumour cases in val set:")
        for rank, (c, n) in enumerate(counts, 1):
            print(f"  {rank:2}. {c}  —  {n:,} voxels")
        return

    case_id = args.case or pick_best_case(val_cases)

    img_path = NPY_DIR / f"{case_id}_image.npy"
    lbl_path = NPY_DIR / f"{case_id}_label.npy"
    if not img_path.exists():
        print(f"ERROR: {img_path} not found. Is the npy cache built?")
        sys.exit(1)

    print(f"\nLoading case: {case_id}")
    image = np.load(img_path).astype(np.float32)   # (4, H, W, D)
    label = np.load(lbl_path).astype(np.float32)   # (3, H, W, D)

    slice_idx = args.slice if args.slice is not None else best_axial_slice(label)
    print(f"Using axial slice: {slice_idx}")

    flair_slice = image[0, :, :, slice_idx]         # (H, W)
    gt_mask     = label[:, :, :, slice_idx].astype(np.uint8)  # (3, H, W)

    # Ground truth overlay
    print("\nGenerating ground truth overlay...")
    save_overlay(flair_slice, gt_mask, "Ground Truth", OUT_DIR / "overlay_gt.png")

    # 3D meshes from GT
    print("\nGenerating 3D tumour meshes...")
    generate_meshes(label, OUT_DIR)

    if args.no_inference:
        print("\nDone (--no-inference: skipped model predictions).")
        return

    # Per-model inference
    model_slug = {
        "3D U-Net":        "unet3d",
        "Attention U-Net": "attention_unet",
        "Swin UNETR":      "swin_unetr",
        "KAN U-Net":       "kan_unet",
        "KAN 3D U-Net":    "kan_unet3d_full",
    }

    predictions = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"\nSkipping {name} — checkpoint not found: {ckpt_path}")
            continue

        print(f"\nRunning inference: {name}")
        model = load_model(ckpt_path, device)
        logits = run_inference(model, image, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pred_mask = logits_to_binary(logits)[:, :, :, slice_idx]  # (3, H, W)
        predictions[name] = pred_mask

        slug = model_slug[name]
        save_overlay(flair_slice, pred_mask, name, OUT_DIR / f"overlay_{slug}.png")

    # Comparison figure
    if predictions:
        print("\nGenerating comparison figure...")
        save_comparison(
            flair_slice, gt_mask, predictions, slice_idx,
            OUT_DIR / "comparison_all_models.png",
        )

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
