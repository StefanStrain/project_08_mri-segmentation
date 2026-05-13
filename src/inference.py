import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

try:
    from .metrics import dice_score, hausdorff95
    from .models.attention_unet3d import AttentionUNet3D
    from .models.kan_unet3d import KANUNet3d
    from .models.swin_unetr import build_swin_unetr
    from .models.unet3d import UNet3D
    from .models.kan_unet3d_full import KANUNet3dFull
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.metrics import dice_score, hausdorff95
    from src.models.attention_unet3d import AttentionUNet3D
    from src.models.kan_unet3d import KANUNet3d
    from src.models.swin_unetr import build_swin_unetr
    from src.models.unet3d import UNet3D
    from src.models.kan_unet3d_full import KANUNet3dFull

_MODELS = {
    "UNet3D": UNet3D,
    "AttentionUNet3D": AttentionUNet3D,
    "SwinUNETR": build_swin_unetr,
    "KANUNet3d": KANUNet3d,
    "KANUNet3dFull": KANUNet3dFull,
}


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """
        Load any supported model from a trainer checkpoint file.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    name = cfg["name"]
    model_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Supported: {list(_MODELS)}")
    model = _MODELS[name](**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {name} — epoch {ckpt['epoch']}  "
          f"(patch val_dice_mean={ckpt['val_dice_mean']:.4f})")
    return model


def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    sw_overlap: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
        model: trained model in eval mode
        image: (4, H, W, D) float32 tensor, already normalised
        patch_size: spatial size of each inference patch
        sw_overlap: fraction of overlap between adjacent patches
        device: defaults to the model's device
    """
    if device is None:
        device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device) # (1, 4, H, W, D)
    with torch.no_grad():
        logits = sliding_window_inference(
            x,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=sw_overlap,
            mode="gaussian", # gaussian weighting blends patches at boundaries, reduces edge artefacts
        )
    return logits.squeeze(0).cpu()  # (3, H, W, D)


def evaluate_checkpoint(
    checkpoint_path: Path,
    npy_cache_dir: Path,
    splits_path: Path,
    split: str = "val",
    max_cases: int | None = 50,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    sw_overlap: float = 0.5,
    compute_hd95: bool = False,
    seed: int = 42,
) -> dict[str, float]:
    """
        Evaluate a checkpoint using full-volume sliding-window inference.
        HD95 is implemented via compute_hd95=True but was never used for the final results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint_path), device)

    splits = json.loads(Path(splits_path).read_text())
    case_ids = splits[split]
    if max_cases is not None:
        rng_seed = seed + 1 if split == "val" else seed
        rng = random.Random(rng_seed)
        case_ids = rng.sample(case_ids, min(max_cases, len(case_ids)))

    npy_dir = Path(npy_cache_dir)
    wt_scores, tc_scores, et_scores = [], [], []
    hd_wt, hd_tc, hd_et = [], [], []
    skipped = 0

    bar = tqdm(case_ids, desc=f"Evaluating [{split}]", unit="case")
    for case_id in bar:
        img_path = npy_dir / f"{case_id}_image.npy"
        lbl_path = npy_dir / f"{case_id}_label.npy"

        if not img_path.exists() or not lbl_path.exists():
            bar.write(f" skip {case_id}  (not in npy cache)")
            skipped += 1
            continue

        image = torch.from_numpy(np.load(img_path).astype(np.float32))  # (4, H, W, D)
        label = torch.from_numpy(np.load(lbl_path).astype(np.float32))  # (3, H, W, D)

        logits = predict_volume(model, image, patch_size, sw_overlap, device)

        scores = dice_score(logits.unsqueeze(0), label.unsqueeze(0))
        wt_scores.append(scores["WT"])
        tc_scores.append(scores["TC"])
        et_scores.append(scores["ET"])

        # HD95 is slow (~30s per case on CPU), never ran it for the final results
        #TODO: run this once HD95 gets added to the portfolio page
        if compute_hd95:
            hd = hausdorff95(logits.unsqueeze(0), label.unsqueeze(0))
            hd_wt.append(hd["WT"])
            hd_tc.append(hd["TC"])
            hd_et.append(hd["ET"])

        n_done = len(wt_scores)
        bar.write(
            f" [{n_done:>3}/{len(case_ids)}] {case_id} | "
            f"  WT {scores['WT']:.3f} TC {scores['TC']:.3f} ET {scores['ET']:.3f}  "
            f"(running mean: {sum(wt_scores)/n_done:.3f} / "
            f"{sum(tc_scores)/n_done:.3f} / "
            f"{sum(et_scores)/n_done:.3f})"
        )
        bar.set_postfix(
            WT=f"{sum(wt_scores)/n_done:.3f}",
            TC=f"{sum(tc_scores)/n_done:.3f}",
            ET=f"{sum(et_scores)/n_done:.3f}",
        )

    n = len(wt_scores)
    if n == 0:
        raise RuntimeError(
            f"No cases evaluated. Check that npy_cache_dir={npy_cache_dir} "
            f"contains files for the {split} split."
        )

    wt = sum(wt_scores) / n
    tc = sum(tc_scores) / n
    et = sum(et_scores) / n

    results: dict = {
        "WT": wt, "TC": tc, "ET": et,
        "mean": (wt + tc + et) / 3,
        "n_cases": n,
        "skipped": skipped,
    }
    if compute_hd95:
        results["hd95_WT"] = sum(hd_wt) / n
        results["hd95_TC"] = sum(hd_tc) / n
        results["hd95_ET"] = sum(hd_et) / n

    return results


def _print_results(results: dict, hd95: bool) -> None:
    n = results["n_cases"]
    print(f"\n{'='*52}")
    print(f" Full-volume evaluation  ({n} cases)")
    print(f"{'='*52}")
    print(f" Dice  WT : {results['WT']:.4f}")
    print(f" Dice  TC : {results['TC']:.4f}")
    print(f" Dice  ET : {results['ET']:.4f}")
    print(f" Dice mean: {results['mean']:.4f}")
    if hd95:
        print(f" HD95 WT : {results['hd95_WT']:.2f} mm")
        print(f" HD95 TC : {results['hd95_TC']:.2f} mm")
        print(f" HD95 ET : {results['hd95_ET']:.2f} mm")
    if results["skipped"]:
        print(f" (skipped {results['skipped']} cases not found in cache)")
    print(f"{'='*52}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--npy-cache-dir", default="cache_npy")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--hd95", action="store_true", help="Also compute HD95 (slow, ~30s/case on CPU)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        npy_cache_dir=Path(args.npy_cache_dir),
        splits_path=Path(args.splits_path),
        split=args.split,
        max_cases=args.max_cases,
        patch_size=tuple(args.patch_size),
        sw_overlap=args.sw_overlap,
        compute_hd95=args.hd95,
        seed=args.seed,
    )
    _print_results(results, hd95=args.hd95)


if __name__ == "__main__":
    main()
