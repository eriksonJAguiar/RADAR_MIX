import json
import sys
sys.path.append("../utils")
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms

import shap
import utils

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def __get_layer(model, model_name):
    
    layer = None
    
    if model_name == "resnet50":
        layer = [model.layer4[-1]]
    elif model_name == "vgg16":
        layer = [model.features[23]]
    elif model_name == "vgg19":
        layer = [model.features[35]]
    elif model_name == "inceptionv3":
        layer = [model.Mixed_7c]
    elif model_name == "efficientnet":
        layer = [model._blocks[17]]
    elif model_name == "densenet":
        layer = [model.features.denseblock4.denselayer16.conv2]
    
    return layer

def shap_explainer(model_path, model_name, nb_class, images):
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed.eval() 
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_images = preprocess(images).unsqueeze(0)
    
    xai_shap  = shap.DeepExplainer(model_trianed, data=input_images[-10:-1])
    
    shap_values = xai_shap.shap_values(input_images[0])
    
    print(shap_values)
    

def grad_cam_explainer(model_path, model_name, nb_class, images, labels):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval() 
    
    #rgb_img = images[0] / 255
    input_image = images[7]
    input_image = np.clip(input_image, 0, 1)
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
    
    
    target_layers = __get_layer(model_trianed, model_name)
    
    cam = GradCAM(model=model_trianed, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(labels[7])]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(input_image.transpose((1, 2, 0)), grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_image.transpose(1, 2, 0), cmap='viridis')
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(visualization, alpha=0.6, cmap='viridis')
    axes[1].axis('off')
    axes[1].set_title('GradCAM Visualization')

    # Save the GradCAM visualization image
    plt.savefig(f'gradcam_visualization_{model_name}.png')

    plt.show()
    
    