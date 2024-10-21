
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from FacialNet import *
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    image = image.resize((224, 224))
    save = image.copy()
    image = normalize_image(image)
    image = torch.from_numpy(image).float().permute(2, 0, 1)
    return image.unsqueeze(0), save  # add batch dimension

def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output

def visualize_result(original_image, prediction, num_classes=9):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # show original image
    ax1.imshow(original_image)
    ax1.axis('off')
    ax1.set_title('Original')

    # show segmentation result
    custom_colors = [
        [0, 0, 0, 1],       
        [1, 0.05, 0.58, 1],       
        [0, 0, 1, 1],       
        [0, 1, 1, 1],       
        [1, 1, 0, 1],       
        [1, 0, 1, 1],       
        [0, 1, 1, 1],       
        [1, 0.5, 0, 1],     
        [1, 1, 1, 1],     
    ]
    color_map = np.array(custom_colors)
    pred = prediction.argmax(dim=1).squeeze().cpu().numpy()
    segmentation = np.zeros((pred.shape[0], pred.shape[1], 4))
    for class_id in range(num_classes):
        mask = pred == class_id
        segmentation[mask] = color_map[class_id]
    
    ax2.imshow(original_image)
    ax2.imshow(segmentation, alpha=0.8)
    ax2.axis('off')
    ax2.set_title('Segmentation')
    
    plt.tight_layout()
    plt.show()

def normalize_image(image):
    # ensure image is numpy array format
    if isinstance(image, Image.Image):
        image = np.array(image)
    # ensure image is float32 type
    image = image.astype(np.float32)
    
    # RGB mean
    mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    # scale value
    val = 0.017
    
    # execute normalization: (image - mean) * val
    normalized_image = (image - mean) * val
    
    return normalized_image


def main():
    # load model
    model = FacialNet()  # use your model class
    model.load_state_dict(torch.load('checkpoints/FacialNet.pth'))
    model.eval()
    # load image
    image_path = 'code/task3/image/00001.png'  # test image path
    image, save = load_image(image_path)
    # predict
    prediction = predict(model, image)
    # visualize result
    visualize_result(save, prediction)

if __name__ == '__main__':
    main()
