import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
            Soft Dice loss averaged over all channels. Sigmoid applied internally, pass raw logits.
        """
        probs = torch.sigmoid(preds)
        # sum over spatial dims only (2,3,4), not batch or channel
        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = (probs + targets).sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.eps) / (union + self.eps) # div zero prevention with eps 
        # subtract from 1 because we want to minimise loss but maximise Dice
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self._dice = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 50/50 split between BCE and Dice (BCE gives a stable gradient from epoch 1)
        # when predictions are around zero (pure Dice loss gradient basically vanishes early on)
        # binary_cross_entropy_with_logits is more numerically stable than sigmoid + BCE separately
        bce = F.binary_cross_entropy_with_logits(preds, targets)
        dice = self._dice(preds, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    #! never used in training, all five models used BCEDiceLoss
    #TODO: try this for ET — focal loss down-weights easy background voxels which might help on the smallest region
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0, # gamma=2 is the standard value from the original focal loss paper
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.eps = eps
        self._dice = DiceLoss(eps=eps)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss down-weights easy background voxels; gamma controls the strength."""
        p = torch.sigmoid(preds)
        focal_pos = -(1.0 - p).pow(self.gamma) * torch.log(p + self.eps)
        focal_neg = -p.pow(self.gamma) * torch.log(1.0 - p + self.eps)
        focal = (targets * focal_pos + (1.0 - targets) * focal_neg).mean()
        dice = self._dice(preds, targets)
        return self.focal_weight * focal + self.dice_weight * dice
