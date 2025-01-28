import os
import random
import numpy as np
import torch
import monai
from monai.data import Dataset, DataLoader
import nibabel as nib
from Dataset.Transforms2D import transforms2D
from monai.data import pad_list_data_collate
from monai.transforms import LoadImage


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)


def load_nifti_slices(file_path):
    loader = LoadImage(image_only=True,ensure_channel_first=True)  
    img = loader(file_path)  
    return [img[:,:, i,:] for i in range(img.shape[2])]

def get_data(data_path):
    train_dataset = []
    val_dataset = []
    images = {}
    images_val = {}
    masks = {}
    masks_val = {}
    for class_file in os.listdir(data_path):
        class_path = os.path.join(data_path, class_file)
        if class_file == "imagesTr":
            for image_file in os.listdir(class_path):
                if image_file.endswith((".nii.gz", ".nii")):
                    prefix = "_".join(image_file.split("_")[:2])
                    images[prefix] = os.path.join(class_path, image_file)
        elif class_file == "labelsTr":
            for label_file in os.listdir(class_path):
                if label_file.endswith((".nii.gz", ".nii")):
                    prefix = label_file.split(".")[0]
                    masks[prefix] = os.path.join(class_path, label_file)
        elif class_file == "imagesVal":
            for image_file in os.listdir(class_path):
                if image_file.endswith((".nii.gz", ".nii")):
                    prefix = "_".join(image_file.split("_")[:2])
                    images_val[prefix] = os.path.join(class_path, image_file)
        elif class_file == "labelsVal":
            for label_file in os.listdir(class_path):
                if label_file.endswith((".nii.gz", ".nii")):
                    prefix = label_file.split(".")[0]
                    masks_val[prefix] = os.path.join(class_path, label_file)

    for prefix, image_path in images.items():
        label_path = masks.get(prefix, None)
        image_slices = load_nifti_slices(image_path)
        label_slices = load_nifti_slices(label_path) if label_path else [None] * len(image_slices)
        for img_slice, lbl_slice in zip(image_slices, label_slices):
            train_dataset.append({"image": img_slice, "mask": lbl_slice})

    for prefix, image_path in images_val.items():
        label_path = masks_val.get(prefix, None)
        image_slices = load_nifti_slices(image_path)
        label_slices = load_nifti_slices(label_path) if label_path else [None] * len(image_slices)
        for img_slice, lbl_slice in zip(image_slices, label_slices):
            val_dataset.append({"image": img_slice, "mask": lbl_slice})  

    return train_dataset, val_dataset


def get_loader2D(data_path, transforms=transforms2D, shuffle=False, batch_size=1, seed=42):
    set_seed(seed)
    train_data, val_data = get_data(data_path)
    
    train_dataset = Dataset(train_data, transforms)
    val_dataset = Dataset(val_data, transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
