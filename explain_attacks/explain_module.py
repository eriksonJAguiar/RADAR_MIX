import json
import sys
sys.path.append("../utils")
sys.path.append("./evaluate_explaination")
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

import shap
import utils
from evaluate_explaination import iou_score, ssim_score

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

def shap_explainer(model_path, model_name, nb_class, image_target, background_imgs, label_target, class_names_path, save_path_shap=None, is_attack=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval()

    #file = open("./dataset/MelanomaDB/class_name.json")
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    
    input_image = torch.from_numpy(image_target).unsqueeze(0).float().to(device)
    background = torch.from_numpy(background_imgs).float().to(device)
    
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
        plt.figure(figsize=(4,6))
        shap.image_plot(
            shap_values_norm.transpose(1, 2, 0),
            input_norm.transpose(1, 2, 0),
            labels=[cls],
            true_labels=[class_names[label_target]]
        )
        plt.savefig(os.path.join(save_path_shap, f"shap_attack_explain_label_{class_names[label_target]}" if is_attack else f"shap_explain_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
        
        normalized_shap_values = (shap_values[class_index][0] - np.min(shap_values[class_index][0])) / (np.max(shap_values[class_index][0]) - np.min(shap_values[class_index][0]))
        plt.figure(figsize=(4,6))
        plt.imshow(normalized_shap_values.transpose(1, 2, 0), cmap="jet")
        plt.axis("off")
        plt.savefig(os.path.join(save_path_shap, f"shap_attack_saliency_label_{class_names[label_target]}" if is_attack else  f"shap_saliency_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
    
    return input_norm.transpose(1, 2, 0), shap_values_norm.transpose(1, 2, 0)
    
    # print(shap_values)
    
def grad_cam_explainer(model_path, model_name, nb_class, image_target, label_target, class_names_path=None, save_path_cam=None, is_attack=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval() 
    
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    
    #rgb_img = images[0] / 255
    input_image = image_target
    #input_image = np.clip(input_image, 0, 1)
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
    
    input_tensor = __normalize(input_tensor)
    input_image = __normalize(input_image, is_cuda=False)
    
    input_image = np.clip(input_image, 0, 1)
    
    target_layers = __get_layer(model_trianed, model_name)
    
    cam = GradCAM(model=model_trianed, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(label_target)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    #visualization = show_cam_on_image(input_image.transpose(1, 2, 0), grayscale_cam, use_rgb=True)
    
    normalized_heatmap = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
    heatmap_rgb = plt.cm.jet(normalized_heatmap)[..., :3]
    
    alpha = 0.3  # Adjust the transparency
    visualization = alpha * heatmap_rgb + (1 - alpha) * input_image.transpose(1, 2, 0)
    
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(input_image.transpose(1, 2, 0), cmap='viridis')
    # axes[0].axis('off')
    # axes[0].set_title('Original Image')

    # axes[1].imshow(visualization, cmap='jet')
    # axes[1].axis('off')
    # axes[1].set_title('GradCAM Visualization')
    # plt.savefig(save_path_cam, bbox_inches = 'tight', dpi = 300)
    
    
    # Save the GradCAM visualization image
    if not save_path_cam is None:
        plt.figure(figsize=(4,6))
        plt.imshow(visualization, cmap='viridis')
        plt.axis("off")
        plt.savefig(os.path.join(save_path_cam, f"cam_attack_explain_label_{class_names[label_target]}" if is_attack else f"cam_explain_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
        
        plt.figure(figsize=(4,6))
        plt.imshow(input_image.transpose(1, 2, 0), cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(save_path_cam, f"cam_attack_input_label_{class_names[label_target]}" if is_attack else f"cam_input_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
        
        plt.figure(figsize=(4,6))
        plt.imshow(grayscale_cam, cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(save_path_cam, f"cam_attac_saliency_label_{class_names[label_target]}" if is_attack else f"cam_saliency_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
        
    return input_image.transpose(1, 2, 0), visualization, grayscale_cam
    
def lime_explainer(model_path, model_name, nb_class, image_target, label_target, class_names_path=None, save_path_lime=None, is_attack=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_trianed = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    model_trianed = model_trianed.to(device)
    model_trianed.eval()
    
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    
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
    
    mask = np.zeros_like(image_target)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)  
    temp = __normalize(temp.transpose(2, 0, 1), is_cuda=False).transpose(1, 2, 0)
    img_boundry1 = mark_boundaries(temp, mask)
    
    if not save_path_lime is None:
        plt.figure(figsize=(4,6))
        plt.imshow(img_boundry1)
        plt.axis("off")
        plt.savefig(os.path.join(save_path_lime, f"lime_attack_explain_label_{class_names[label_target]}" if is_attack else f"lime_explain_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
        
        norm_input = __normalize(image_target, is_cuda=False)
        plt.figure(figsize=(4,6))
        plt.imshow(norm_input.transpose(1,2,0))
        plt.axis("off")
        plt.savefig(os.path.join(save_path_lime, f"lime_attack_input_label_{class_names[label_target]}" if is_attack else f"lime_input_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
        
        plt.figure(figsize=(4,6))
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(save_path_lime, f"lime_attack_saliency_label_{class_names[label_target]}" if is_attack else f"lime_saliency_label_{class_names[label_target]}"), bbox_inches = 'tight', pad_inches=0, dpi = 300)
    
    return img_boundry1, mask
    
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

def tsne_visualizer(model, model_name, nb_class, images_target):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    features_list = []
    for img in images_target:
        input_image = torch.from_numpy(img).unsqueeze(0).float().to(device)
        
        features = __load_embeddings(model, model_name, input_image)
        features_list.append(features)
    
    all_features = np.vstack(features_list)
    
    method_tsne = TSNE(n_components=2, learning_rate='auto', init="random", perplexity=2).fit_transform(all_features)
    
    #__plot_tsne_umap(method_tsne, labels_names, attack_title=attack_title, class_names=labels_names, save_path="./tnse_visualizer")
    
    return method_tsne
    
def umap_visualizer(model, model_name, nb_class, images_target):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    features_list = []
    for img in images_target:
        input_image = torch.from_numpy(img).unsqueeze(0).float().to(device)
        
        features = __load_embeddings(model, model_name, input_image)
        features_list.append(features)
    
    all_features = np.vstack(features_list)
    
    method_umap = umap.UMAP(n_components=2, n_neighbors=2, min_dist=0.1, metric='euclidean').fit_transform(all_features)
    
    #__plot_tsne_umap(method_umap, attack_title=attack_title, class_names=labels_names, save_path="umap_visualizer.png")
    
    return method_umap

def plot_tsne_umap(features, method, attack_title, class_names, save_path=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), constrained_layout=True)
    
    category_colors = {
        category: plt.cm.tab10(i) for i, category in enumerate(set(class_names))
    }

    point_colors = [category_colors[category] for category in class_names]


    scatter = ax.scatter(features[:, 0], features[:, 1], c=point_colors, alpha=0.5)
    #ax.set_title(f'Clean', fontsize=16)  

    plt.title(f'{method} Dimensionality reduction Attack: {attack_title}', fontweight='bold', fontsize=10)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in category_colors.values()]
    plt.legend(legend_handles, category_colors.keys(), loc="lower center", bbox_to_anchor=(.5, 1.10), ncol=3)
    
    if not save_path is None:
        fig.savefig(save_path, bbox_inches = 'tight', dpi = 300)

def __normalize(image, is_cuda=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).to(device)

    if is_cuda:
        img_norm = image * STD[:, None, None] + MEAN[:, None, None]
    else:
        img_norm = image * STD[:, None, None].cpu().numpy() + MEAN[:, None, None].cpu().numpy()
        
    return img_norm

def run_explainer(weights_path, model_name, dataset_name, images_target, images_adv_target, labels_target, nb_class, class_names_path=None, root_save_path=None):
    
    model_path = os.path.join(weights_path, "{}-{}-exp1.ckpt".format(model_name, dataset_name))
    cls_searh = list(filter(lambda x: x in list(labels_target), range(nb_class)))
    idx_rand = list(map(lambda x: list(labels_target).index(x), cls_searh))
    #idx_rand = [-1]
    
    iou_lime, ssim_lime = [], []
    iou_shap, ssim_shap = [], []
    iou_cam, ssim_cam = [], []
    
    for i, (x, y, x_adv) in enumerate(zip(images_target, labels_target, images_adv_target)):
        _, shap_scores = shap_explainer(model_path=model_path, 
                                        model_name=model_name, 
                                        image_target=x,
                                        background_imgs=images_target[:32], 
                                        label_target=y, 
                                        nb_class=nb_class, 
                                        class_names_path=class_names_path,
                                        save_path_shap = root_save_path if i in idx_rand else None)
        
        
        _, shap_scores_adv = shap_explainer(model_path=model_path, 
                                                model_name=model_name, 
                                                image_target=x_adv,
                                                background_imgs=images_adv_target[:32], 
                                                label_target=y, 
                                                nb_class=nb_class, 
                                                class_names_path=class_names_path,
                                                save_path_shap=root_save_path if i in idx_rand else None,
                                                is_attack=True)
        
        _, grad_image, scores_cam = grad_cam_explainer(model_path=model_path, 
                                                        model_name=model_name, 
                                                        image_target=x, 
                                                        label_target=y, 
                                                        nb_class=nb_class,
                                                        class_names_path=class_names_path,
                                                        save_path_cam=root_save_path if i in idx_rand else None)
        
        _, grad_image_adv, scores_cam_adv = grad_cam_explainer(model_path=model_path, 
                                                           model_name=model_name, 
                                                           image_target=x_adv, 
                                                           label_target=y, 
                                                           nb_class=nb_class,
                                                           class_names_path=class_names_path,
                                                           save_path_cam=root_save_path if i in idx_rand else None,
                                                           is_attack=True)
        
        lime_img, score_lime = lime_explainer(model_path=model_path, 
                                              model_name=model_name, 
                                              nb_class=nb_class, 
                                              image_target=x,
                                              label_target=y,
                                              class_names_path=class_names_path,
                                              save_path_lime=root_save_path if i in idx_rand else None)
        
        lime_img, score_lime_adv = lime_explainer(model_path=model_path, 
                                              model_name=model_name, 
                                              nb_class=nb_class,
                                               class_names_path=class_names_path, 
                                              image_target=x_adv,
                                              label_target=y,
                                              save_path_lime=root_save_path if i in idx_rand else None,
                                              is_attack=True)
        
      
        iou_lime.append(iou_score(image_original=score_lime, image_attacked=score_lime_adv))
        iou_shap.append(iou_score(image_original=shap_scores, image_attacked=shap_scores_adv))
        iou_cam.append(iou_score(image_original=scores_cam, image_attacked=scores_cam_adv))
        ssim_shap.append(ssim_score(image_original=shap_scores, image_attacked=shap_scores_adv, win_size=5))
        ssim_cam.append(ssim_score(image_original=scores_cam, image_attacked=scores_cam_adv, win_size=5))
        ssim_lime.append(ssim_score(image_original=score_lime, image_attacked=score_lime_adv, win_size=5))
        
    metrics_xai = pd.DataFrame()
    metrics_xai["images_idx"] = np.concatenate([range(len(iou_lime)), range(len(iou_shap)), range(len(iou_cam))])
    metrics_xai["Method"] = np.concatenate([np.repeat("Lime", len(iou_lime)), np.repeat("SHAP", len(iou_shap)), np.repeat("Grad-CAM", len(iou_cam))])
    metrics_xai["IoU"] =  np.concatenate([iou_lime, iou_shap, iou_cam])
    metrics_xai["SSIM"] = np.concatenate([ssim_lime, ssim_shap, ssim_cam])
    
    print(metrics_xai)
    
    return metrics_xai
    
        
def manifold_visualization(weights_path, model_name, dataset_name, attack_title, images_target, images_adv_target, labels_target, nb_class, class_names_path, root_save_path):
    
    file = open(class_names_path)
    class_names = list([v for v in json.load(file).values()])
    class_names_adv = ["Clean", "Adversarial"]
    
    labels_names = list(map(lambda x: class_names[x], labels_target))
    
    clean_adv = np.vstack([images_target, images_adv_target])
    clean_adv_labels = np.concatenate([np.repeat(0, len(labels_target)), np.repeat(1, len(labels_target))])
    clean_adv_names = list(map(lambda x: class_names_adv[int(x)], clean_adv_labels))
    
    model_path = os.path.join(weights_path, "{}-{}-exp1.ckpt".format(model_name, dataset_name))
    model = utils.read_model_from_checkpoint(model_path, model_name, nb_class)
    
    tnse_embeddings = tsne_visualizer(model=model,
                                      model_name=model_name,
                                      nb_class=nb_class,
                                      images_target=images_target)
    
    tnse_embeddings_adv = tsne_visualizer(model=model,
                                      model_name=model_name,
                                      nb_class=nb_class,
                                      images_target=images_adv_target)
    
    tnse_embeddings_clean_adv = tsne_visualizer(model=model,
                                                model_name=model_name,
                                                nb_class=2,
                                                images_target=clean_adv)
    
    umap_embeddings = umap_visualizer(model=model,
                                      model_name=model_name,
                                      nb_class=nb_class,
                                      images_target=images_target)
    
    umap_embeddings_adv = umap_visualizer(model=model,
                                          model_name=model_name,
                                          nb_class=nb_class,
                                          images_target=images_adv_target)
    
    umap_embeddings_clean_adv = umap_visualizer(model=model,
                                                model_name=model_name,
                                                nb_class=2,
                                                images_target=clean_adv)
    
    plot_tsne_umap(features=tnse_embeddings,
                   method="T-SNE",
                   attack_title="None",
                   class_names=labels_names,
                   save_path=os.path.join(root_save_path, f"tsne_clean.png"))
    
    
    plot_tsne_umap(features=umap_embeddings,
                   method="UMAP",
                   attack_title="None",
                   class_names=labels_names,
                   save_path=os.path.join(root_save_path, f"umap_clean.png"))
    
    
    plot_tsne_umap(features=tnse_embeddings_adv,
                   method="T-SNE",
                   attack_title=attack_title,
                   class_names=labels_names,
                   save_path=os.path.join(root_save_path, f"tsne_adv_{attack_title}.png"))
    
    
    plot_tsne_umap(features=umap_embeddings_adv,
                   method="UMAP",
                   attack_title=attack_title,
                   class_names=labels_names,
                   save_path=os.path.join(root_save_path, f"umap_adv_{attack_title}.png"))
    
    plot_tsne_umap(features=tnse_embeddings_clean_adv,
                   method="T-SNE",
                   attack_title=attack_title,
                   class_names=clean_adv_names,
                   save_path=os.path.join(root_save_path, f"tsne_clean_adv_{attack_title}.png"))
    
    
    plot_tsne_umap(features=umap_embeddings_clean_adv,
                   method="UMAP",
                   attack_title=attack_title,
                   class_names=clean_adv_names,
                   save_path=os.path.join(root_save_path, f"umap_clean_adv_{attack_title}.png"))
    