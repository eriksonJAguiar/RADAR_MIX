import json
import sys
sys.path.append("../utils")
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import shap
import utils

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.manifold import TSNE
import umap


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

def shap_explainer(model_path, model_name, nb_class, image_target, label_target, class_names_path, save_path_shap=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval()

    #file = open("./dataset/MelanomaDB/class_name.json")
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    
    input_image = torch.from_numpy(image_target).unsqueeze(0).float().to(device)
    background = torch.from_numpy(image_target).float().to(device)
    
    explainer = shap.GradientExplainer(model_trianed, background)
    shap_values = explainer.shap_values(input_image)
    
    #shap_values_abs = [np.abs(s) for s in shap_values]
    
    class_index = np.argmax(model_trianed(input_image).detach().cpu().numpy())
    cls = class_names[class_index]

    input_non_norm =  __normalize(input_image[0])
    
    shap_values_norm = np.clip(shap_values[class_index][0], 0, 1)
    input_norm = np.clip(input_non_norm.cpu().numpy(), 0, 1)

    #plt.savefig("shap_explanation.png")
    if not save_path_shap is None:
        shap.image_plot(
            shap_values_norm.transpose(1, 2, 0),
            input_norm.transpose(1, 2, 0),
            labels=[cls],
            true_labels=[class_names[label_target]]
        )
        plt.savefig('{}.png'.format(save_path_shap), bbox_inches = 'tight', dpi = 300)
        
    
    return input_norm.transpose(1, 2, 0), shap_values_norm.transpose(1, 2, 0)
    
    # print(shap_values)
    
def grad_cam_explainer(model_path, model_name, nb_class, image_target, label_target, save_path_cam=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval() 
    
    #rgb_img = images[0] / 255
    input_image = image_target
    #input_image = np.clip(input_image, 0, 1)
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
    
    input_tensor = __normalize(input_tensor)
    input_image = __normalize(input_image, is_cuda=False)
    
    
    target_layers = __get_layer(model_trianed, model_name)
    
    cam = GradCAM(model=model_trianed, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(label_target)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(input_image.transpose((1, 2, 0)), grayscale_cam, use_rgb=True)
    
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(input_image.transpose(1, 2, 0), cmap='viridis')
    # axes[0].axis('off')
    # axes[0].set_title('Original Image')

    # axes[1].imshow(visualization, alpha=0.6, cmap='viridis')
    # axes[1].axis('off')
    # axes[1].set_title('GradCAM Visualization')
    
    
    # Save the GradCAM visualization image
    if not save_path_cam is None:
        plt.imshow(visualization, alpha=0.6, cmap='viridis')
        plt.axis("off")
        plt.savefig('{}.png'.format(save_path_cam), bbox_inches = 'tight', dpi = 300)
        
    return input_image.transpose(1, 2, 0), visualization
    
def lime_explainer(model_path, model_name, nb_class, image_target, save_path_lime=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval()
    
    def predict(image):
    
        input_image = torch.from_numpy(image.transpose(0, 3, 1, 2)).float().to(device)
        logits = model_trianed(input_image)
    
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        return probs.detach().cpu().numpy()
    
    explainer = lime_image.LimeImageExplainer()
    
    explanation = explainer.explain_instance(image_target.transpose(1,2,0), 
                                            predict,
                                            top_labels=5, 
                                            hide_color=0,
                                            num_samples=1000)
    
    #explanation = __normalize(explanation, is_cuda=False)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,num_features=10, hide_rest=False)  
    temp = __normalize(temp.transpose(2, 0, 1), is_cuda=False).transpose(1, 2, 0)
    img_boundry1 = mark_boundaries(temp, mask)
    
    if not save_path_lime is None:
        plt.imshow(img_boundry1)
        plt.axis("off")
        plt.savefig(save_path_lime)
    
def __load_embeddings(model, model_name, image):
    
    last_layer = {
        "resnet50": 'layer4.2.conv3',
        "vgg16": "features.24",
        "vgg19": "features.34",
        "efficientnet": "features.7.0.block.0",
        "densenet": "features.denseblock4.denselayer16.conv2",
        "inceptionv3": "Mixed_7c.branch_pool.conv"
    }
    
    # _, eval_nodes = get_graph_node_names(model)
    # print(eval_nodes)
    model_feat = create_feature_extractor(model, return_nodes=[last_layer[model_name]])
    
    with torch.no_grad():
        features_dict = model_feat(image)
    
    features = list(features_dict.items())[-1][-1]
    
    features_flatten = features.view(features.size(0), -1)
    
    return features_flatten.detach().cpu().numpy()

def tsne_visualizer(model_path, model_name, nb_class, images_target, labels_target, class_names_path, save_path_tsne=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    
    labels_names = list(map(lambda x: class_names[x], labels_target))
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval()

    features_list = []
    for img in images_target:
        input_image = torch.from_numpy(img).unsqueeze(0).float().to(device)
        
        features = __load_embeddings(model_trianed, model_name, input_image)
        features_list.append(features)
    
    all_features = np.vstack(features_list)
    
    method_tsne = TSNE(n_components=2, learning_rate='auto', init="random", perplexity=2).fit_transform(all_features)
    
    __plot_tsne_umap(method_tsne, labels_names, attack_title="None", class_names=labels_names, save_path="./tnse_visualizer")
    
def umap_visualizer(model_path, model_name, nb_class, images_target, labels_target, class_names_path, save_path_tsne=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    
    labels_names = list(map(lambda x: class_names[x], labels_target))
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval()

    features_list = []
    for img in images_target:
        input_image = torch.from_numpy(img).unsqueeze(0).float().to(device)
        
        features = __load_embeddings(model_trianed, model_name, input_image)
        features_list.append(features)
    
    all_features = np.vstack(features_list)
    
    method_umap = umap.UMAP(n_components=2, n_neighbors=2, min_dist=0.1, metric='euclidean').fit_transform(all_features)
    
    __plot_tsne_umap(method_umap, attack_title="None", class_names=labels_names, save_path="./umap_visualizer")

def __plot_tsne_umap(tnse_features, attack_title, class_names, save_path=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), constrained_layout=True)
    
    category_colors = {
        category: plt.cm.tab10(i) for i, category in enumerate(set(class_names))
    }

    point_colors = [category_colors[category] for category in class_names]


    scatter = ax.scatter(tnse_features[:, 0], tnse_features[:, 1], c=point_colors, alpha=0.5)
    #ax.set_title(f'Clean', fontsize=16)  

    plt.title(f'T-SNE Dimensionality reduction ({attack_title})', fontweight='bold', fontsize=10)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in category_colors.values()]
    plt.legend(legend_handles,category_colors.keys(), title="Classes")
    
    if not save_path is None:
        fig.savefig('{}.png'.format(save_path), bbox_inches = 'tight', dpi = 300)

def __normalize(image, is_cuda=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).to(device)

    if is_cuda:
        img_norm = image * STD[:, None, None] + MEAN[:, None, None]
    else:
        img_norm = image * STD[:, None, None].cpu().numpy() + MEAN[:, None, None].cpu().numpy()
        
    return img_norm
    
    