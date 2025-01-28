from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    MapTransform,
    ToNumpyd,
    ToTensord,
    ScaleIntensityd,
    ToTensord,
    RandAdjustContrastd,
    RandRotated,
    RandCropByLabelClassesd,
    RandFlipd,
    RandShiftIntensityd,
    )
from monai.data import PatchIterd


import numpy as np
import cv2
from skimage.color import rgb2gray
from monai.transforms import MapTransform
from scipy.ndimage import label

transforms3D = Compose([
    LoadImaged(keys=['image','mask'], ensure_channel_first=True),
    ScaleIntensityd(keys=['image']),
    RandCropByLabelClassesd(
    keys=["image", "mask"],
    label_key="mask",
    spatial_size=[64,64,64],
    ratios=[0, 22, 2, 2],
    num_classes=4,
    num_samples=8),
    RandFlipd(
            keys=["image", "mask"],
            spatial_axis=[0],
            prob=0.10,
        ),
    RandFlipd(
        keys=["image", "mask"],
        spatial_axis=[1],
        prob=0.10,
    ),
    RandFlipd(
        keys=["image", "mask"],
        spatial_axis=[2],
        prob=0.10,),
   
    ToTensord(keys=['image','mask']),
])
transforms3D_val = Compose([
    LoadImaged(keys=['image','mask'], ensure_channel_first=True),  
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image','mask']),
])
transforms3D_test=Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image']),
])