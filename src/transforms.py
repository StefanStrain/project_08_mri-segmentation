from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ToTensord,
)

def get_preprocessing_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        # converts single-channel mask (values 0,1,2,4) into 3 binary channels: WT, TC, ET
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        # nonzero=True: only normalise non-zero voxels — skull-stripped volumes have large
        # zero borders that would skew mean/std if included.
        # channel_wise=True: each modality normalised independently since FLAIR/T1/T1CE/T2
        # have completely different intensity ranges.
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

def get_npy_transforms(
    patch_size: tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    return Compose([
        # pos=1, neg=1: 50/50 foreground/background patch split, standard BraTS starting point.
        # TODO: experiment with higher pos ratio (e.g. pos=2, neg=1) to show the model more tumour.
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # prob=1.0: intensity augmentation always fires, not randomly applied.
        # TODO: could add elastic deformation here, common in BraTS augmentation pipelines.
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ToTensord(keys=["image", "label"]),
    ])

def get_train_transforms(
    patch_size: tuple[int, int, int] = (128, 128, 128),
) -> Compose:
    # full pipeline in one go: load -> normalise -> crop -> augment
    return Compose([
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    # deterministic only, no cropping or augmentation for validation
    return Compose([
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])
