"""
Visualise attention gate activations from the Attention U-Net.

For each decoder level, extracts the gate's sigmoid output (the spatial mask that
determines which encoder features pass through) and overlays it on the FLAIR slice.

Outputs (in docs/figures/):
  attention_gate_{1-4}.png   — per-level gate heatmap overlaid on FLAIR
  attention_gates_all.png    — all 4 levels side-by-side

Usage:
  python scripts/visualize_attention.py
  python scripts/visualize_attention.py --case BraTS2021_00698 --slice 77
"""
import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.attention_unet3d import AttentionUNet3D, AttentionGate3d
from scripts.visualize_segmentation import (
    get_val_cases, pick_best_case, best_axial_slice, NPY_DIR, OUT_DIR
)

CHECKPOINT = ROOT / "checkpoints/attention_unet/best_model.pt"


# ---------------------------------------------------------------------------
# Hook-based gate extraction
# ---------------------------------------------------------------------------

class GateExtractor:
    """Registers forward hooks on all AttentionGate3d modules."""

    def __init__(self, model: torch.nn.Module):
        self.gates: list[torch.Tensor] = []
        self._hooks = []
        for module in model.modules():
            if isinstance(module, AttentionGate3d):
                self._hooks.append(
                    module.psi.register_forward_hook(self._hook)
                )

    def _hook(self, module, input, output):
        self.gates.append(output.detach().cpu())  # (1, 1, H', W', D')

    def clear(self):
        self.gates.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Tumour-centred single-patch inference
# ---------------------------------------------------------------------------

def run_with_gates(model, image: np.ndarray, label: np.ndarray,
                   device: torch.device, patch_size=(128, 128, 128)):
    """
    Find the tumour centre, extract a single aligned patch, run one forward
    pass, and return the gate activations alongside the patch crop bounds.

    Sliding-window inference would mix gates from many different patches.
    A single centred patch keeps the gate maps spatially aligned with the
    FLAIR region being shown.
    """
    _, H, W, D = image.shape
    ph, pw, pd = patch_size

    # Tumour centre from the label (any positive voxel)
    tumour_voxels = np.argwhere(label.max(axis=0) > 0)
    if len(tumour_voxels) == 0:
        ch, cw, cd = H // 2, W // 2, D // 2
    else:
        centre = tumour_voxels.mean(axis=0).astype(int)
        ch, cw, cd = int(centre[0]), int(centre[1]), int(centre[2])

    # Crop bounds clamped so the patch fits inside the volume
    h0 = max(0, min(ch - ph // 2, H - ph))
    w0 = max(0, min(cw - pw // 2, W - pw))
    d0 = max(0, min(cd - pd // 2, D - pd))
    h1, w1, d1 = h0 + ph, w0 + pw, d0 + pd

    patch = image[:, h0:h1, w0:w1, d0:d1]
    x = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).to(device)

    extractor = GateExtractor(model)
    extractor.clear()
    with torch.no_grad():
        _ = model(x)

    gates = extractor.gates[:]   # exactly 4 tensors (one per gate level)
    extractor.remove()
    return gates, (h0, w0, d0, h1, w1, d1)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _normalise_gate(gate_slice: np.ndarray) -> np.ndarray:
    """
    Percentile-stretch the gate to [0, 1] so deep gates (which tend to be
    near-uniform at high values) don't collapse to a flat white image.
    """
    p_lo, p_hi = np.percentile(gate_slice, [2, 98])
    if p_hi > p_lo:
        return np.clip((gate_slice - p_lo) / (p_hi - p_lo), 0, 1)
    return np.clip(gate_slice, 0, 1)


def _get_gate_slice(gate_volume: torch.Tensor, patch_shape, local_d: int) -> np.ndarray:
    """Upsample gate to patch spatial dims and return one axial slice."""
    gate = gate_volume.squeeze().numpy()  # (H', W', D')
    ph, pw, pd = patch_shape

    if gate.shape != (ph, pw, pd):
        gate_t = torch.from_numpy(gate).unsqueeze(0).unsqueeze(0).float()
        gate_t = torch.nn.functional.interpolate(
            gate_t, size=(ph, pw, pd), mode="trilinear", align_corners=False
        )
        gate = gate_t.squeeze().numpy()

    d_idx = max(0, min(local_d, gate.shape[2] - 1))
    return gate[:, :, d_idx]


def plot_gate(flair_patch: np.ndarray, gate_volume: torch.Tensor,
              local_d: int, level: int, path: Path):
    """
    flair_patch : (H, W, D) FLAIR patch (tumour-centred 128³ crop)
    gate_volume : (1, 1, H', W', D') sigmoid gate output
    local_d     : axial slice index within the patch
    """
    ph, pw, _ = flair_patch.shape
    flair_slice = flair_patch[:, :, local_d]

    gate_slice = _get_gate_slice(gate_volume, (ph, pw, flair_patch.shape[2]), local_d)
    gate_vis   = _normalise_gate(gate_slice)

    vmin, vmax = flair_slice.min(), flair_slice.max()
    gray = (flair_slice - vmin) / (vmax - vmin + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)

    axes[0].imshow(gray, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR (tumour-centred patch)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gate_vis, cmap="hot", vmin=0, vmax=1, origin="lower")
    axes[1].set_title(f"Gate {level} activation", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(gray, cmap="gray", origin="lower")
    axes[2].imshow(gate_vis, cmap="hot", alpha=0.5, vmin=0, vmax=1, origin="lower")
    axes[2].set_title(f"Gate {level} overlay", fontsize=10)
    axes[2].axis("off")

    gate_raw = gate_volume.squeeze().numpy()
    plt.suptitle(
        f"Attention Gate — Level {level}  "
        f"(decoder resolution {gate_raw.shape[0]}×{gate_raw.shape[1]})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


def plot_all_gates(flair_patch: np.ndarray, gates, local_d: int, path: Path):
    n = len(gates)
    if n == 0:
        print("  No gates captured.")
        return

    ph, pw, _ = flair_patch.shape
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=130)
    if n == 1:
        axes = [axes]

    vmin, vmax = flair_patch[:, :, local_d].min(), flair_patch[:, :, local_d].max()
    gray = (flair_patch[:, :, local_d] - vmin) / (vmax - vmin + 1e-8)

    for i, gate_vol in enumerate(gates):
        gate_slice = _get_gate_slice(gate_vol, (ph, pw, flair_patch.shape[2]), local_d)
        gate_vis   = _normalise_gate(gate_slice)

        axes[i].imshow(gray, cmap="gray", origin="lower")
        axes[i].imshow(gate_vis, cmap="hot", alpha=0.55, vmin=0, vmax=1, origin="lower")
        gate_raw = gate_vol.squeeze().numpy()
        axes[i].set_title(f"Gate {i+1}\n(res {gate_raw.shape[0]}×{gate_raw.shape[1]})", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Attention Gate Activations — Attention U-Net", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case",  default=None)
    parser.add_argument("--slice", type=int, default=None)
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg   = ckpt["config"]["model"]
    model = AttentionUNet3D(**{k: v for k, v in cfg.items() if k != "name"}).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load case
    val_cases = get_val_cases()
    case_id   = args.case or pick_best_case(val_cases)
    image = np.load(NPY_DIR / f"{case_id}_image.npy").astype(np.float32)
    label = np.load(NPY_DIR / f"{case_id}_label.npy").astype(np.float32)

    # Global best slice, then translate to patch-local index
    global_slice = args.slice if args.slice is not None else best_axial_slice(label)

    print(f"\nExtracting attention gates for {case_id}, global slice {global_slice}...")
    gates, (h0, w0, d0, h1, w1, d1) = run_with_gates(model, image, label, device)
    print(f"  Patch bounds: h[{h0}:{h1}] w[{w0}:{w1}] d[{d0}:{d1}]")
    print(f"  Captured {len(gates)} gate activations")

    flair_patch = image[0, h0:h1, w0:w1, d0:d1]   # (H_p, W_p, D_p)
    local_d = max(0, min(global_slice - d0, flair_patch.shape[2] - 1))
    print(f"  Local slice index within patch: {local_d}")

    for i, gate in enumerate(gates[:4]):
        plot_gate(flair_patch, gate, local_d, i + 1,
                  OUT_DIR / f"attention_gate_{i+1}.png")

    plot_all_gates(flair_patch, gates[:4], local_d, OUT_DIR / "attention_gates_all.png")
    print(f"\nAttention gate figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
