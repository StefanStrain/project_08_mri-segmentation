"""
    Swin UNETR (Hatamizadeh et al. 2022, arXiv:2201.01266)
    Using MONAI's built-in implementation rather than reimplementing from scratch, though that could be a fun exercise.
    Reference: https://arxiv.org/abs/2201.01266
    MONAI tutorial: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb
"""
from monai.networks.nets import SwinUNETR


def build_swin_unetr(
    in_channels=4,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True,
):
    # wrapper function so the trainer registry can call it the same way as the other models
    # feature_size=48 is the BraTS21 paper default, gives ~62M params 
    # TODO: try feature_size=24 (~16M params) if VRAM is a problem, might also overfit less
    # use_checkpoint=True enables gradient checkpointing, cuts VRAM at the cost of slightly slower training
    # img_size is not passed here because this MONAI version infers it from the input tensor at forward time
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )
