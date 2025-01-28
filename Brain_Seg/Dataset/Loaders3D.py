import os
import random
import numpy as np
import torch
import monai
from monai.data import Dataset, DataLoader
from monai.transforms import LoadImage
from monai.utils import set_determinism
from configs import config



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed=seed)



def get_data(data_path, patch_size=None):
    if patch_size is None:
        patch_size = tuple(config.transforms['3D_patch_size'])
    train_dataset = []
    val_dataset = []
    test_dataset=[]
    images = {}
    images_val = {}
    masks = {}
    masks_val={}
    images_test={}
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
        elif class_file == "imagesTs":
            for image_file in os.listdir(class_path):
                if image_file.endswith((".nii.gz", ".nii")):
                    prefix = "_".join(image_file.split("_")[:2])
                    images_test[prefix] = os.path.join(class_path, image_file)
       
    for prefix, image_path in images.items():
        label_path = masks.get(prefix, None)
        train_dataset.append({"image": image_path, "mask": label_path})

    for prefix, image_path in images_val.items():
        label_path = masks_val.get(prefix, None)
        val_dataset.append({"image": image_path, "mask": label_path})

    for prefix, image_path in images_test.items():
        test_dataset.append({"image": image_path,"prefix":prefix})

    return train_dataset, val_dataset,test_dataset

def get_loader3D(data_path, transforms=None, shuffle=False, batch_size=1, seed=42):
    set_seed(seed)
    train_data, val_data,test_data = get_data(data_path)

    train_dataset = Dataset(train_data, transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_dataset = Dataset(val_data, transforms)  
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    test_dataset = Dataset(test_data, transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader

