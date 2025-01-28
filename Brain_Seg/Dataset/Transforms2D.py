from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    MapTransform,
    ToNumpyd,
    ToTensord,
    ScaleIntensityd,
    ToTensord,
    EnsureChannelFirstd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandRotated,
    RandFlipd,
    RandGaussianSharpend,
    RandBiasFieldd,
    SpatialPad
    )
import numpy as np
import cv2
from skimage.color import rgb2gray
from monai.transforms import MapTransform
from scipy.ndimage import label
import torch
from monai.transforms import MapTransform
from configs import config
import matplotlib.pyplot as plt
import SimpleITK as sitk
  
class crop_backgroundd(MapTransform):
    def __init__(self, keys, threshold, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold
        self.output_key = output_key

    def crop(self, input):
        if isinstance(input, torch.Tensor):
            input = input.numpy()
        channels, height, width = input.shape
        lower_limit = self.threshold
        upper_limit = height - self.threshold
        cropped_image = np.zeros((channels, upper_limit - lower_limit, width), dtype=input.dtype)
        cropped_image[0] = input[:, lower_limit:upper_limit, :]
        return cropped_image

    def __call__(self, data):
        for key in self.key_iterator(data):
            output_key = self.output_key or key
            data[output_key] = self.crop(data[key])
        return data

class N4BiasFieldCorrectiond(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def correct_bias(self, input_tensor):
        corrected_channels = []
        for c in range(input_tensor.shape[0]): 
            channel_tensor = input_tensor[c]
            input_image = sitk.GetImageFromArray(channel_tensor.numpy())
            mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
            input_image = sitk.Cast(input_image, sitk.sitkFloat32)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected_image = corrector.Execute(input_image, mask_image)
            corrected_tensor = torch.from_numpy(sitk.GetArrayFromImage(corrected_image))
            corrected_channels.append(corrected_tensor)
        return torch.stack(corrected_channels, dim=0)
    def __call__(self, data):
      for key in self.keys:
          if key not in data:
              raise KeyError(f"Key {key} not found in data.")
          input_tensor = data[key]
          if isinstance(input_tensor, list):
              input_tensor = torch.tensor(input_tensor)
          output_key = self.output_key or key
          data[output_key] = self.correct_bias(input_tensor)
      return data

class AddCroppedRows:
    def __init__(self, threshold):
        self.threshold = threshold

    def pad(self, input):
        n, c, h, w = input.shape
        top_rows = torch.zeros((n, c, self.threshold, w), device=input.device, dtype=input.dtype)
        bottom_rows = torch.zeros((n, c, self.threshold, w), device=input.device, dtype=input.dtype)
        padded_tensor = torch.cat([top_rows, input, bottom_rows], dim=2)
        return padded_tensor

    def __call__(self, input):
        return self.pad(input)
        
padder = SpatialPad((256,128,256))

transforms2D = Compose([
    ScaleIntensityd(keys=['image']),
    RandFlipd(keys = ["image", "mask"], 
          prob = 0.3, 
          spatial_axis = (0, 1)),
    RandBiasFieldd(keys=['image'],prob=0.3,coeff_range=(0.1,0.25)),
    ToTensord(keys=['image','mask']),

])
transforms2D_val = Compose([
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image','mask']),
])
transforms2D_test = Compose([
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image']),
])

