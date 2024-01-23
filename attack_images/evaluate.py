import torch
import pandas as pd
import numpy as np

from torchmetrics import Accuracy, Recall, Specificity, Precision, F1Score, AUROC
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

def get_last_layer_features(model, model_name, image):
    
    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    # features = feature_extractor(image)
        
    # return features.detach().cpu().numpy()
    
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
    features_dict = model_feat(image)
    
    features = list(features_dict.items())[-1][-1]
    
    return features.detach().cpu()

def evaluate_model(model, model_name, dataset_clean, dataset_adv, nb_class):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #3rd predict attacked images
    #define loss of the model
    avg_accuracy_clean, avg_accuracy_adv = [], []
    avg_auc_clean, avg_auc_adv = [], []
    asr = []
    
    adv_auc = AUROC(task="binary") if not nb_class > 2 else AUROC(task="multiclass", num_classes=nb_class, average="macro")
    val_auc = AUROC(task="binary") if not nb_class > 2 else AUROC(task="multiclass", num_classes=nb_class, average="macro")
    
    logits_clean, logits_adv  = [], []
    feat_clean, feat_adv = [], []
    
    model.eval()
    with torch.no_grad():
        for i, (data_clean, data_adv) in enumerate(zip(dataset_clean, dataset_adv)):
            #test images for clean examples
            x_clean, y_clean = data_clean
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
            pred_clean = model(x_clean)
            
            #get logits
            logits_clean.append(pred_clean.detach().cpu())
            
            #calculate metrics for clean
            y_clean = y_clean if nb_class > 2 else y_clean.view(-1, 1).float()
            y_pred = torch.argmax(pred_clean, dim=1) if nb_class > 2 else (pred_clean > 0.5).float()
            y_prob = torch.softmax(pred_clean, dim=1) if nb_class > 2 else torch.sigmoid(pred_clean)
            
            #get features clean
            feat_clean.append(get_last_layer_features(model, model_name, x_clean))
            
            #calcualte metrics for clean
            accuracy_clean = np.sum(y_pred.cpu().numpy() == y_clean.cpu().numpy()) / len(y_clean)
            avg_accuracy_clean.append(accuracy_clean)
            val_auc(y_prob, y_clean)
            avg_auc_clean.append(val_auc.compute().cpu().numpy())
        
            #test images for adversarial examples
            x_adv, y_adv = data_adv
            x_adv, y_adv = x_adv.to(device), y_adv.to(device)
            pred_adv = model(x_adv)
            
            #get logits for adv
            logits_adv.append(pred_adv.detach().cpu())
            
            #calcualte metrics for adv
            y_adv = y_adv if nb_class > 2 else y_adv.view(-1, 1).float()
            y_pred = torch.argmax(pred_adv, dim=1) if nb_class > 2 else (pred_adv > 0.5).float()
            y_prob = torch.softmax(pred_adv, dim=1) if nb_class > 2 else torch.sigmoid(pred_adv)
            
            #get features
            feat_adv.append(get_last_layer_features(model, model_name, x_adv))

            #calculate metrics for adv
            accuracy_adv = np.sum(y_pred.cpu().numpy() == y_adv.cpu().numpy()) / len(y_adv)
            avg_accuracy_adv.append(accuracy_adv)
            adv_auc(y_prob, y_adv)
            avg_auc_adv.append(adv_auc.compute().cpu().numpy())
            
            #evaluate the attack sucess rate (asr)
            asr.append(1 - accuracy_adv)
        
    epochs_metrics = pd.DataFrame()
    epochs_metrics["epochs"] = list(range(len(dataset_clean)))
    epochs_metrics["val_acc"] = avg_accuracy_clean
    epochs_metrics["val_acc_adv"] = avg_accuracy_adv
    epochs_metrics["val_auc"] = avg_auc_clean
    epochs_metrics["val_auc_adv"] = avg_auc_adv
    epochs_metrics["asr"] = asr
    
    #logits_clean = np.asanyarray(logits_clean)
    logits_clean = torch.cat(logits_clean, dim=0).numpy()
    logits_adv = torch.cat(logits_adv, dim=0).numpy()
    feat_clean = torch.cat(feat_clean, dim=0).numpy()
    feat_adv = torch.cat(feat_adv, dim=0).numpy()
    
    return epochs_metrics, logits_clean, logits_adv, feat_clean, feat_adv