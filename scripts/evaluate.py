"""
    Full-volume sliding-window evaluation for any trained checkpoint.

    Usage:
        python scripts/evaluate.py --checkpoint checkpoints/unet3d/best_model.pt
        python scripts/evaluate.py --checkpoint checkpoints/attention_unet/best_model.pt
        python scripts/evaluate.py --checkpoint checkpoints/swin_unetr/best_model.pt
        python scripts/evaluate.py --checkpoint checkpoints/kan_unet3d/best_model.pt
        python scripts/evaluate.py --checkpoint checkpoints/kan_unet3d_full/best_model.pt

        #! override number of val cases or sliding-window overlap
        python scripts/evaluate.py --checkpoint checkpoints/unet3d/best_model.pt --cases 50 --overlap 0.5
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import dice_score
from src.models.unet3d import UNet3D
from src.models.attention_unet3d import AttentionUNet3D
from src.models.swin_unetr import build_swin_unetr
from src.models.kan_unet3d import KANUNet3d
from src.models.kan_unet3d_full import KANUNet3dFull

_FACTORIES = {
    "UNet3D": UNet3D,
    "AttentionUNet3D": AttentionUNet3D,
    "SwinUNETR": build_swin_unetr,
    "KANUNet3d": KANUNet3d,
    "KANUNet3dFull": KANUNet3dFull,
}

NPY_DIR = ROOT / "cache_npy"
SPLITS = ROOT / "data/splits.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--cases", type=int, default=50, help="Number of val cases (default 50)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap (default 0.5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    name = cfg["name"]
    model = _FACTORIES[name](**{k: v for k, v in cfg.items() if k != "name"}).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: {name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {ckpt['epoch']}  (patch dice {ckpt['val_dice_mean']:.4f})")

    # val cases - same fixed sample used across all models
    val_ids = random.Random(43).sample(json.loads(SPLITS.read_text())["val"], args.cases)
    val_ids = [v for v in val_ids if (NPY_DIR / f"{v}_label.npy").exists()]
    print(f"Val cases:  {len(val_ids)}\n")

    wt_scores, tc_scores, et_scores = [], [], []

    for i, cid in enumerate(val_ids):
        image = np.load(NPY_DIR / f"{cid}_image.npy").astype(np.float32)
        label = np.load(NPY_DIR / f"{cid}_label.npy").astype(np.float32)

        x = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = sliding_window_inference(
                x, roi_size=(128, 128, 128), sw_batch_size=1,
                predictor=model, overlap=args.overlap, mode="gaussian",
            )

        # call straight into the canonical metric so this script can't drift from src/inference.py
        target = torch.from_numpy(label).unsqueeze(0).to(logits.device)
        scores = dice_score(logits, target)
        wt, tc, et = scores["WT"], scores["TC"], scores["ET"]
        wt_scores.append(wt)
        tc_scores.append(tc)
        et_scores.append(et)

        mean = (wt + tc + et) / 3
        print(f"  [{i+1:2d}/{len(val_ids)}] {cid}  WT {wt:.3f}  TC {tc:.3f}  ET {et:.3f}  mean {mean:.3f}")

    print(f"\n{'='*60}")
    print(f" WT: {np.mean(wt_scores):.4f}  (std {np.std(wt_scores):.4f})")
    print(f" TC: {np.mean(tc_scores):.4f}  (std {np.std(tc_scores):.4f})")
    print(f" ET: {np.mean(et_scores):.4f}  (std {np.std(et_scores):.4f})")
    print(f" Mean: {np.mean([np.mean(wt_scores), np.mean(tc_scores), np.mean(et_scores)]):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
