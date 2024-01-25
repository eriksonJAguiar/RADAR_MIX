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

def evaluate_model(model, dataset, nb_class, is_attacked=False):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #3rd predict attacked images
    #define loss of the model
    avg_accuracy = []
    avg_auc = []
    asr = []
    
    auc = AUROC(task="binary") if not nb_class > 2 else AUROC(task="multiclass", num_classes=nb_class, average="weighted")
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            #test images for adversarial examples
            x, y = data
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            #calcualte metrics for adv
            y = y if nb_class > 2 else y.view(-1, 1).float()
            y_pred = torch.argmax(pred, dim=1) if nb_class > 2 else (pred > 0.5).float()
            y_prob = torch.softmax(pred, dim=1) if nb_class > 2 else torch.sigmoid(pred)

            #calculate metrics for adv
            accuracy = np.sum(y_pred.cpu().numpy() == y.cpu().numpy()) / len(y)
            avg_accuracy.append(accuracy)
            auc(y_prob, y)
            avg_auc.append(auc.compute().cpu().numpy())
            
            if is_attacked:
                #evaluate the attack sucess rate (asr)
                asr.append(1 - accuracy)
        
    epochs_metrics = pd.DataFrame()
    epochs_metrics["epochs"] = list(range(len(dataset)))
    epochs_metrics["val_acc"] = avg_accuracy
    epochs_metrics["val_auc"] = avg_auc
    
    if is_attacked:
        epochs_metrics["asr"] = asr
    
    #logits_clean = np.asanyarray(logits_clean)
    # logits_adv = torch.cat(logits_adv, dim=0).numpy()
    # feat_adv = torch.cat(feat_adv, dim=0).numpy()
    
    return epochs_metrics