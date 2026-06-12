import torch
from monai.metrics import HausdorffDistanceMetric

# worst-case HD95: diagonal of the 240x240x155 BraTS volume
_BRATS_VOLUME_DIAGONAL = 374.0


def _build_brats_regions(
    binary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Split a (B, 3, H, W, D) binary channel mask into WT, TC, ET region tensors.
        ConvertToMultiChannelBasedOnBratsClassesd already emits the overlapping BraTS
        regions, so we just pick the channels out, no reconstruction needed:
        channel 0 = TC (labels 1+4), channel 1 = WT (labels 1+2+4), channel 2 = ET (label 4).
    """
    tc = binary[:, 0].float()
    wt = binary[:, 1].float()
    et = binary[:, 2].float()
    return wt, tc, et


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5,
) -> dict[str, float]:
    """
        Hard Dice score for WT, TC, ET BraTS regions, averaged over batch.
        preds: raw logits (B, 3, H, W, D)
        targets: binary targets (B, 3, H, W, D)
        Returns {"WT": float, "TC": float, "ET": float, "mean": float}

        Empty regions follow the BraTS-official convention:
        both empty (GT and pred) scores 1.0 (perfect agreement), GT empty but
        something predicted scores 0.0 (false positive). This is the single canonical
        Dice used everywhere - scripts/evaluate.py calls straight into here so the two
        evaluation paths can't drift apart.
    """
    binary_preds = torch.sigmoid(preds) > 0.5 # threshold at 0.5 to get hard predictions
    pred_wt, pred_tc, pred_et = _build_brats_regions(binary_preds)
    gt_wt, gt_tc, gt_et = _build_brats_regions(targets.bool())

    results = {}
    for name, pred_r, gt_r in [
        ("WT", pred_wt, gt_wt),
        ("TC", pred_tc, gt_tc),
        ("ET", pred_et, gt_et),
    ]:
        intersection = (pred_r * gt_r).sum(dim=(1, 2, 3))
        denom = pred_r.sum(dim=(1, 2, 3)) + gt_r.sum(dim=(1, 2, 3))
        # denom == 0 means both pred and GT are empty -> perfect agreement (1.0)
        # everything else is standard Dice; GT-empty-but-predicted falls out as 0.0 on its own
        # clamp keeps the division safe in the branch torch.where doesn't end up using
        dice = torch.where(
            denom > 0,
            2.0 * intersection / denom.clamp(min=eps),
            torch.ones_like(denom),
        )
        results[name] = dice.mean().item()

    results["mean"] = (results["WT"] + results["TC"] + results["ET"]) / 3
    return results


def hausdorff95(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """
        HD95 for WT, TC, ET BraTS regions, averaged over batch.
        preds: raw logits (B, 3, H, W, D)
        targets: binary targets (B, 3, H, W, D)

        Not used for the final results on the portfolio page, Dice only.
        #TODO: run this once it gets added to the portfolio page
    """
    binary_preds = torch.sigmoid(preds) > 0.5
    pred_wt, pred_tc, pred_et = _build_brats_regions(binary_preds)
    gt_wt, gt_tc, gt_et = _build_brats_regions(targets.bool())

    hd_fn = HausdorffDistanceMetric(percentile=95, reduction="mean", get_not_nans=False)

    results = {}
    for name, pred_r, gt_r in [
        ("WT", pred_wt, gt_wt),
        ("TC", pred_tc, gt_tc),
        ("ET", pred_et, gt_et),
    ]:
        batch_scores = []
        for b in range(pred_r.shape[0]):
            p = pred_r[b]
            g = gt_r[b]
            if p.sum() == 0 and g.sum() == 0:
                batch_scores.append(0.0) # both empty = perfect agreement
            elif p.sum() == 0 or g.sum() == 0:
                batch_scores.append(_BRATS_VOLUME_DIAGONAL) # one empty = worst case
            else:
                hd_fn(y_pred=p[None, None].float(), y=g[None, None].float())
                score = hd_fn.aggregate()
                hd_fn.reset()
                batch_scores.append(float(score))
        results[name] = sum(batch_scores) / len(batch_scores)

    return results
