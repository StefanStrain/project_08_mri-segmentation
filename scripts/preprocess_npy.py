"""
    One-time preprocessing: convert BraTS NIfTI files to float16 numpy arrays.

    Saves {case_id}_image.npy (float16, 4x240x240x155) and
    {case_id}_label.npy (uint8, 3x240x240x155) to the output directory.
    Already-processed cases are skipped so you can safely re-run if interrupted.

    Usage:
        python scripts/preprocess_npy.py
        python scripts/preprocess_npy.py --max-train 200 --max-val 50
        python scripts/preprocess_npy.py --max-train 0 --max-val 0 #! dry run
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import _build_case_dicts, generate_splits
from src.transforms import get_preprocessing_transforms


def _process_cases(case_dicts: list[dict], output_dir: Path, transform) -> int:
    """Process and save each case. Returns number of newly written files."""
    written = 0
    for case_dict in tqdm(case_dicts, unit="case", leave=True):
        case_id = Path(case_dict["image"][0]).parent.name
        img_out = output_dir / f"{case_id}_image.npy"
        lbl_out = output_dir / f"{case_id}_label.npy"

        if img_out.exists() and lbl_out.exists():
            tqdm.write(f"skip {case_id}")
            continue

        result = transform(case_dict)
        np.save(img_out, np.asarray(result["image"].cpu()).astype(np.float16))
        np.save(lbl_out, np.asarray(result["label"].cpu()).astype(np.uint8))
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/src_1/BraTS2021_Training_Data")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument("--output-dir", default="cache_npy")
    parser.add_argument("--max-train", type=int, default=150, help="Train cases to preprocess (default 150, ~14 GB)")
    parser.add_argument("--max-val", type=int, default=50, help="Val cases to preprocess (default 50, ~4.7 GB)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = generate_splits(data_root, seed=args.seed, output_path=Path(args.splits_path))
    splits = json.loads(splits_path.read_text())

    # use same sampling logic as get_loaders so the right cases get preprocessed
    train_ids = splits["train"]
    if args.max_train > 0:
        rng = random.Random(args.seed)
        train_ids = rng.sample(train_ids, min(args.max_train, len(train_ids)))
    else:
        train_ids = []

    val_ids = splits["val"]
    if args.max_val > 0:
        rng_val = random.Random(args.seed + 1)
        val_ids = rng_val.sample(val_ids, min(args.max_val, len(val_ids)))
    else:
        val_ids = []

    total = len(train_ids) + len(val_ids)
    est_gb = total * 93 / 1024
    print(f"Will process {len(train_ids)} train + {len(val_ids)} val = {total} cases")
    print(f"Estimated disk usage: ~{est_gb:.1f} GB -> {output_dir}/")
    print(f"Estimated time: ~{total * 18 / 60:.0f} minutes\n")

    transform = get_preprocessing_transforms()

    if train_ids:
        print(f"[1/2] Training cases ({len(train_ids)})")
        _process_cases(_build_case_dicts(train_ids, data_root), output_dir, transform)

    if val_ids:
        print(f"\n[2/2] Validation cases ({len(val_ids)})")
        _process_cases(_build_case_dicts(val_ids, data_root), output_dir, transform)

    npy_files = list(output_dir.glob("*.npy"))
    size_gb = sum(f.stat().st_size for f in npy_files) / 1e9
    print(f"\nDone. {len(npy_files) // 2} cases saved -> {output_dir}/ ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
