from pathlib import Path

import mlflow
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    PolynomialLR,
    ReduceLROnPlateau,
)

from .dataset import get_loaders
from .losses import BCEDiceLoss, FocalDiceLoss
from .metrics import dice_score
from .models.attention_unet3d import AttentionUNet3D
from .models.kan_unet3d import KANUNet3d
from .models.kan_unet3d_full import KANUNet3dFull
from .models.swin_unetr import build_swin_unetr
from .models.unet3d import UNet3D

_MODELS = {
    "UNet3D": UNet3D,
    "AttentionUNet3D": AttentionUNet3D,
    "SwinUNETR": build_swin_unetr,
    "KANUNet3d": KANUNet3d,
    "KANUNet3dFull": KANUNet3dFull,
}

_LOSSES = {
    "BCEDiceLoss": BCEDiceLoss,
    "FocalDiceLoss": FocalDiceLoss,
}


class Trainer:
    def __init__(self, config: dict, resume_path: str | None = None) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_cfg = config["data"]
        npy_cache_dir = data_cfg.get("npy_cache_dir")
        self.train_loader, self.val_loader = get_loaders(
            data_root=Path(data_cfg["data_root"]),
            splits_path=Path(data_cfg["splits_path"]),
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            patch_size=tuple(data_cfg["patch_size"]),
            val_fraction=data_cfg["val_fraction"],
            seed=data_cfg["seed"],
            max_train_samples=data_cfg.get("max_train_samples"),
            max_val_samples=data_cfg.get("max_val_samples"),
            npy_cache_dir=Path(npy_cache_dir) if npy_cache_dir else None,
        )

        model_cfg = config["model"]
        model_kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
        self.model = _MODELS[model_cfg["name"]](**model_kwargs).to(self.device)

        loss_cfg = config["loss"]
        loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
        self.criterion = _LOSSES[loss_cfg["name"]](**loss_kwargs)

        train_cfg = config["training"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.epochs = train_cfg["epochs"]
        self.scheduler = self._build_scheduler(train_cfg)

        use_amp = train_cfg.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self._use_amp = use_amp
        #! KAN models cast to fp32 internally for spline computation regardless of this flag.
        # AMP still helps the rest of the forward pass though.

        log_cfg = config["logging"]
        self.checkpoint_dir = Path(log_cfg["checkpoint_dir"])
        self.output_dir = Path(log_cfg["output_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_dice = 0.0
        self._start_epoch = 1
        self._current_epoch = 0

        if resume_path is not None:
            #TODO: Find a way to merge two different mlflow runs into one when resuming. 
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.best_dice = ckpt.get("val_dice_mean", 0.0)
            self._start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {ckpt['epoch']} (best dice so far: {self.best_dice:.3f})")

        run_name = log_cfg["run_name"]
        if resume_path is not None:
            run_name = f"{run_name}_resumed_ep{self._start_epoch}"
        mlflow.set_experiment(log_cfg["experiment_name"])
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(self._flatten_config(config))
        

    def _build_scheduler(self, train_cfg: dict):
        """Instantiate the LR scheduler named in config, raises ValueError for unknown names."""
        name = train_cfg["scheduler"]
        if name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                patience=train_cfg.get("plateau_patience", 10),
                factor=train_cfg.get("plateau_factor", 0.5),
            )
        elif name == "poly":
            return PolynomialLR(self.optimizer, total_iters=self.epochs, power=0.9)
        else:
            raise ValueError(
                f"Unknown scheduler: {name!r}. Choose cosine, plateau, or poly."
            )

    def fit(self) -> None:
        """Run the full training loop, logging to MLflow and saving checkpoints."""
        epoch_bar = tqdm(range(self._start_epoch, self.epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_bar:
            self._current_epoch = epoch
            train_loss = self._train_epoch(epoch_bar)
            val_dice = self._val_epoch(epoch_bar)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_dice["mean"])
            else:
                self.scheduler.step()

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "val/dice_WT": val_dice["WT"],
                    "val/dice_TC": val_dice["TC"],
                    "val/dice_ET": val_dice["ET"],
                    "val/dice_mean": val_dice["mean"],
                },
                step=epoch,
            )

            self._save_checkpoint("last", val_dice["mean"])
            is_best = val_dice["mean"] > self.best_dice
            if is_best:
                self.best_dice = val_dice["mean"]
                self._save_checkpoint("best", val_dice["mean"])

            epoch_bar.write(
                f"Epoch {epoch}/{self.epochs} | "
                f"loss {train_loss:.4f} | "
                f"WT {val_dice['WT']:.3f}  "
                f"TC {val_dice['TC']:.3f}  "
                f"ET {val_dice['ET']:.3f}  "
                f"mean {val_dice['mean']:.3f}"
                + ("  ★ best" if is_best else "")
            )

        mlflow.log_artifact(str(self.checkpoint_dir / "best_model.pt"))
        mlflow.log_artifact(str(self.checkpoint_dir / "last_model.pt"))
        mlflow.end_run()

    def _train_epoch(self, epoch_bar: tqdm) -> float:
        """One training epoch with mixed precision; returns mean batch loss."""
        self.model.train()
        total_loss = 0.0
        n = len(self.train_loader)
        batch_bar = tqdm(self.train_loader, desc=f" Train ep{self._current_epoch}", unit="batch", leave=False)
        for i, batch in enumerate(batch_bar, 1):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
                preds = self.model(images)
                loss = self.criterion(preds, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{total_loss / i:.4f}")
        return total_loss / n

    def _val_epoch(self, epoch_bar: tqdm) -> dict[str, float]:
        """
            One validation epoch without gradients, returns Dice per BraTS region plus mean.
            !Important thing to note is that validation runs on random patches, not full volume sliding window inference!
            Dice will be .02-03 lower than the final number from evaluate.py
            Full volume eval is too slow to run every epoch on this hardware.
            TODO: add the HD95 metric (could be expensive / epoch...)
        """
        self.model.eval()
        wt_scores, tc_scores, et_scores = [], [], []
        batch_bar = tqdm(self.val_loader, desc=f"  Val   ep{self._current_epoch}", unit="batch", leave=False)
        with torch.no_grad():
            for batch in batch_bar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                preds = self.model(images)
                scores = dice_score(preds, labels)
                wt_scores.append(scores["WT"])
                tc_scores.append(scores["TC"])
                et_scores.append(scores["ET"])
                batch_bar.set_postfix(
                    WT=f"{sum(wt_scores)/len(wt_scores):.3f}",
                    TC=f"{sum(tc_scores)/len(tc_scores):.3f}",
                    ET=f"{sum(et_scores)/len(et_scores):.3f}",
                )
        wt = sum(wt_scores) / len(wt_scores)
        tc = sum(tc_scores) / len(tc_scores)
        et = sum(et_scores) / len(et_scores)
        return {"WT": wt, "TC": tc, "ET": et, "mean": (wt + tc + et) / 3}

    def _save_checkpoint(self, tag: str, val_dice_mean: float = 0.0) -> None:
        """Save model + optimizer + scheduler state to {tag}_model.pt."""
        path = self.checkpoint_dir / f"{tag}_model.pt"
        torch.save(
            {
                "epoch": self._current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_dice_mean": val_dice_mean,
                "config": self.config,
            },
            path,
        )

    @staticmethod
    def _flatten_config(config: dict, prefix: str = "") -> dict[str, str]:
        """Flatten nested config dict to dot separated keys for MLflow param logging."""
        flat: dict[str, str] = {}
        for k, v in config.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(Trainer._flatten_config(v, key))
            else:
                flat[key] = str(v)
        return flat
