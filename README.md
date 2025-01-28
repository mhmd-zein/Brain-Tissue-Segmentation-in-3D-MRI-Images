# Brain-Tissue-Segmentation-in-3D-MRI-Images
In this project we evaluated different data handling techniques on brain tissue segmentation in 3D MRI images. Pytorch was used for building that project alongside with MONAI library which facilitates the work on 3D medical images.
#  Dataset
For this project we worked on IBRS18 dataset which consists of 18 T1 weighted MRI of skull stripped brain images of healthy subjects of shape 256*256*128. We had the masks of 15 of these 18 images which consists of erebrospinal fluid (CSF), gray matter (GM), and white matter (WM).
10 of these images were used for our training, 5 were used for validation and 3 images without masks were used for our final testing.
# Preprocessing and Data Handling
1- **2D**
The 3D volumetric data, with a shape of (256, 128, 256), was sliced along the axial dimension to create 2D images. Each slice had a shape of (256, 128), representing the spatial resolution in two dimensions. These slices were used as independent samples for training 2D models. 

2- **2.5D**
The 3D volumetric data, with a shape of (256, 128, 256), was processed to create 2.5D slices by stacking each slice along the axial dimension with its adjacent slices (previous and next) across the channel dimension. This resulted in input tensors with a shape of (3, 256, 128), where the three channels represented the current slice and its neighboring slices. These 2.5D slices were used as input for training 2D convolutional neural networks (CNNs), enabling the model to capture inter-slice dependencies while maintaining the efficiency of 2D processing.

3- **3D Patch Based**
The 3D volumetric data, with a shape of (256, 128, 256), was processed by extracting 3D patches using random spatial cropping. From each volume, four patches of size (64, 64, 64) were randomly sampled, ensuring diverse coverage of the original image. This patching strategy provided manageable input sizes for the model while preserving the 3D spatial context within each patch. The extracted patches served as input for training 3D convolutional neural networks (CNNs), leveraging the full volumetric information of the data.

# UNets training
The explanation of training and validation process for each approach is summarized in the following Block Diagrams.
![image](https://github.com/user-attachments/assets/080cc7ce-2ee6-4135-9e30-376291cdaad0)
2D Apprach for Training UNet
![image](https://github.com/user-attachments/assets/f673159a-4b4e-46b7-9849-f4b303694806)
2.5D Apprach for Training UNet
![image](https://github.com/user-attachments/assets/eb09286e-72c2-4428-802c-18ad8316d1e1)
3D Patch Based Apprach for Training UNet
# Results
The results of this project are shown in the Table below. We evaluated our segmentation on some convolutional UNet architecture and we compared our approaches to a reference which is nnUNet.
![image](https://github.com/user-attachments/assets/faa83dd4-62ed-4779-98b8-98fa13f6e303)




