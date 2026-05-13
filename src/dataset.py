import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate

from .transforms import get_npy_transforms, get_train_transforms, get_val_transforms


class NpyDataset(torch.utils.data.Dataset):
    """
        Dataset that loads pre-processed float16 numpy arrays instead of NIfTI files.
        About 100x faster than loading NIfTI on every epoch.
        !Run scripts/preprocess_npy.py once to generate the cache before training.
    """

    def __init__(
        self,
        case_ids: list[str],
        npy_dir: Path,
        patch_size: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        self.npy_dir = Path(npy_dir)
        self.case_ids = case_ids
        self.transform = get_npy_transforms(patch_size)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> dict:
        case_id = self.case_ids[idx]
        img_path = self.npy_dir / f"{case_id}_image.npy"
        if not img_path.exists():
            raise FileNotFoundError(
                f"Pre-processed file not found: {img_path}\n"
                f"Run: python scripts/preprocess_npy.py"
            )
        image = np.load(img_path).astype(np.float32)
        label = np.load(self.npy_dir / f"{case_id}_label.npy").astype(np.float32)
        result = self.transform({"image": image, "label": label})
        # MONAI returns a list when num_samples > 1 in RandCropByPosNegLabeld
        # num_samples=1 so this is always a single dict, but the check avoids a crash
        return result[0] if isinstance(result, list) else result


def generate_splits(
    data_root: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    output_path: Path = Path("data/splits.json"),
) -> Path:
    """
        Scan data_root for BraTS2021_* dirs and write a fixed train/val split to output_path.
        Returns immediately if splits.json already exists)
        The fixed split is what makes the cross-architecture comparisons at least somewhat fair.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    cases = sorted(
        p.name
        for p in Path(data_root).iterdir()
        if p.is_dir() and p.name.startswith("BraTS2021_")
    )

    rng = random.Random(seed)
    rng.shuffle(cases)

    split_idx = int(len(cases) * (1 - val_fraction))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "val_fraction": val_fraction,
                "train": cases[:split_idx],
                "val": cases[split_idx:],
            },
            indent=2,
        )
    )

    return output_path


def _build_case_dicts(case_ids: list[str], data_root: Path) -> list[dict]:
    """
        Build MONAI case dicts for loading directly from NIfTI.
        Only used in the fallback path of get_loaders (when npy_cache_dir is None).
        !In practice never called since all configs set npy_cache_dir.
    """
    dicts = []
    for case_id in case_ids:
        case_dir = Path(data_root) / case_id
        dicts.append(
            {
                "image": [
                    str(case_dir / f"{case_id}_flair.nii.gz"),
                    str(case_dir / f"{case_id}_t1.nii.gz"),
                    str(case_dir / f"{case_id}_t1ce.nii.gz"),
                    str(case_dir / f"{case_id}_t2.nii.gz"),
                ],
                "label": str(case_dir / f"{case_id}_seg.nii.gz"),
            }
        )
    return dicts


def get_loaders(
    data_root: Path,
    splits_path: Path = Path("data/splits.json"),
    batch_size: int = 1,
    num_workers: int = 0,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    val_fraction: float = 0.2,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    npy_cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
        Build train and val DataLoaders.
        When npy_cache_dir is set (always in practice), uses NpyDataset.
        Otherwise falls back to loading NIfTI directly via MONAI Dataset.
    """
    splits_path = generate_splits(
        data_root,
        val_fraction=val_fraction,
        seed=seed,
        output_path=Path(splits_path),
    )

    splits = json.loads(Path(splits_path).read_text())

    train_ids = splits["train"]
    if max_train_samples is not None:
        rng = random.Random(seed)
        train_ids = rng.sample(train_ids, min(max_train_samples, len(train_ids)))

    val_ids = splits["val"]
    if max_val_samples is not None:
        # seed+1 so val sampling uses different ordering than train
        rng_val = random.Random(seed + 1)
        val_ids = rng_val.sample(val_ids, min(max_val_samples, len(val_ids)))

    if npy_cache_dir is not None:
        npy_cache_dir = Path(npy_cache_dir)
        train_ds = NpyDataset(train_ids, npy_cache_dir, patch_size)
        val_ds = NpyDataset(val_ids, npy_cache_dir, patch_size)
    else:
        # fallback - load from NIfTI directly (much slower)
        train_ds = Dataset(data=_build_case_dicts(train_ids, data_root), transform=get_train_transforms(patch_size))
        val_ds = Dataset(data=_build_case_dicts(val_ids, data_root), transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), # for faster GPU transfer
        collate_fn=list_data_collate, #! needed bcs MONAI transforms can return lists
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader
