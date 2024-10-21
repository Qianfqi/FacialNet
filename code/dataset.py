# dataset.py

# Define custom dataset classes and data loading functions.
# This file should include:
# 1. Custom Dataset class(es) inheriting from torch.utils.data.Dataset
# 2. Data loading and preprocessing functions
# 3. Data augmentation techniques (if applicable)
# 4. Functions to split data into train/val/test sets
# 5. Any necessary data transformations
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
from PIL import ImageEnhance
import torchvision.transforms.functional as TF


def show_edge(mask_ori):
    mask = np.array(mask_ori).copy()
    # make sure the mask is 2D image
    if len(mask.shape) == 4:
        mask = mask[0, 0] 
    elif len(mask.shape) == 3:
        mask = mask[0]  
    else:
        mask = mask
    # binary
    binary = (mask > 0.5).astype(np.uint8) * 255
    # find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create output image and draw contours
    myImg = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(myImg, contours, -1, 1, 4)
    return myImg

def deformation(image, mask):
    # random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    
    # random rotation
    angle = random.uniform(-45, 45)
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    # random scale
    scale = random.uniform(0.5, 1.5)
    new_size = (int(224 * scale), int(224 * scale))
    image = TF.resize(image, new_size)
    mask = TF.resize(mask, new_size)
    # random translate
    translate = (random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25))
    image = TF.affine(image, angle=0, translate=translate, scale=1, shear=0)
    mask = TF.affine(mask, angle=0, translate=translate, scale=1, shear=0)
    # resize to 224x224
    image = TF.resize(image, (224, 224))
    mask = TF.resize(mask, (224, 224))
    return image, mask

def texture_enhancement(image):
        # randomly select the enhancement method
        enhancements = ['brightness', 'contrast', 'sharpness', 'noise', 'gaussian']
        num_enhancements = random.randint(1, len(enhancements))
        selected_enhancements = random.sample(enhancements, num_enhancements)
        
        for enhancement in selected_enhancements:
            if enhancement == 'brightness':
                factor = random.uniform(0.4, 1.7)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)
            
            elif enhancement == 'contrast':
                factor = random.uniform(0.5, 1.5)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(factor)
            
            elif enhancement == 'sharpness':
                factor = random.uniform(0.8, 1.3)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(factor)
            
            elif enhancement == 'noise':
                image_array = np.array(image)
                noise = np.random.normal(0, 10, image_array.shape).astype(np.uint8)
                noisy_image = cv2.add(image_array, noise)
                image = Image.fromarray(noisy_image)
            
            elif enhancement == 'gaussian':
                image_array = np.array(image)
                blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
                image = Image.fromarray(blurred)
        return image
    
def normalize_image(image):
    # make sure the image is numpy array format
    if isinstance(image, Image.Image):
        image = np.array(image)
    # make sure the image is float32 type
    image = image.astype(np.float32)
    
    # RGB mean
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    # scale value
    val = 0.017
    
    # execute normalization: (image - mean) * val
    normalized_image = (image - mean) * val
    
    return normalized_image

class EasyPortraitDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.annotations_dir = os.path.join(root_dir, 'annotations', split)
        self.image_files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        annotation = Image.open(ann_path)
        image = image.resize((224, 224))
        annotation = annotation.resize((224, 224), Image.NEAREST)
        if self.transform:
            p = random.randint(0, 1)
            if p == 1:
                image, annotation = deformation(image, annotation)
                image_hat = texture_enhancement(image)
            else:
                image_hat = texture_enhancement(image)
        else:
            image_hat = texture_enhancement(image)
            
        # convert to numpy array
        annotation_array = np.array(annotation)
        # convert to PyTorch tensor
        annotation_tensor = torch.from_numpy(annotation_array).float()
        # add channel dimension
        annotation_tensor = annotation_tensor.unsqueeze(0) # 1 x 224 x 224
        
        # normalize
        image = normalize_image(image)
        image_hat = normalize_image(image_hat)
        # convert to PyTorch tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        image_hat = torch.from_numpy(image_hat).float().permute(2, 0, 1)
        
        assert image.shape == (3, 224, 224) and type(image) == torch.Tensor
        assert image_hat.shape == (3, 224, 224) and type(image_hat) == torch.Tensor
        assert annotation_tensor.shape == (1, 224, 224) and type(annotation_tensor) == torch.Tensor
        assert annotation_tensor.max() == 8 and annotation_tensor.min() == 0
        return image, image_hat, annotation_tensor











