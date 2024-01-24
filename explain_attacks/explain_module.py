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
        layer = [model.features[7][0].block[0]]
    elif model_name == "densenet":
        layer = [model.features.denseblock4.denselayer16.conv2]
    
    return layer

def shap_explainer(model_path, model_name, nb_class, images, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval() 
    
    def predict(img: np.ndarray) -> torch.Tensor:
        img = nhwc_to_nchw(torch.Tensor(img))
        img = img.to(device)
        output = model_trianed(img)
        return output


    classes = np.unique(labels)
    
    input_image_data = torch.from_numpy(images)
    input_image_data = torch.clip(input_image_data, 0, 1).to(device)
    #input_image = np.clip(input_image, 0, 1)
    #input_tensor = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
    
    print(input_image_data[0].shape)
    
    masker_blur = shap.maskers.Image("blur(224,224)", input_image_data[0].shape)
    explainer = shap.Explainer(predict, masker_blur, output_names=classes)
    
    
    shap_values = explainer(
        input_image_data[0],
        max_evals=1000,
        batch_size=32,
        outputs=0,
    )
    
    shap_values.data = shap_values.data.cpu().numpy()[0]
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]
    
    print(shap_values.data)
    print(shap_values.values)
    
    shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        true_labels=[labels[0]],
    )
    plt.savefig("shap_image_plot.png")
    
    # print(shap_values)
    

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
    
    