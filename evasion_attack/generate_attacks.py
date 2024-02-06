import sys
sys.path.append("../utils")

import torch
import numpy as np
import pandas as pd
import utils
import os
import time
import evaluate

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool, CarliniL2Method, UniversalPerturbation, ProjectedGradientDescent, BasicIterativeMethod, Wasserstein

def generate_attack(model, data_loader, input_shape, lr, nb_class, attack_name, eps):
    """selecting and generate adversarial attack

    Args:
        model (torch.nn.Module): pytorch pre-trained model 
        data_loader (torch.utils.data.Dataloader): Image dataloader
        input_shape (tuple): size of image, i.e., (128, 128)
        lr (float): learning rate
        nb_class (int): number of class in dataset
        attack_name (str): attack will be performed
        eps (float): noise level

    Returns:
        images (np.ndarray): clean images 
        adv_images (np.ndarray) crafted image attacked by adversarial attacks 
        true_labels (np.ndarray): labels that indicate true values
    """
    #define the loss and optimizer
    loss = torch.nn.CrossEntropyLoss() if nb_class > 2 else torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    #create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=opt,
        #clip_values=[0,1],
        input_shape=input_shape,
        nb_classes=nb_class
    )
    
    #4th execute the attack type
    images, adv_images, true_labels = __get_adv_attack(attack_name=attack_name, 
                                               data_loader=data_loader, 
                                               classifier=classifier, 
                                               eps=eps)
    
    return images, adv_images, true_labels

def __get_adv_attack(attack_name, data_loader, classifier, eps):
    """select one of attacks defined

    Args:
        attack_name (str): attack name, such as FGSM, BIM, DeepFool, CW, PGD, and UAP.
        data_loader (torch.utils.data.Dataloader): images and labels dataloader
        classifier (art.estimators.classification.PyTorchClassifier): ART classifier generate using model and features of attack
        eps (float): noise level

    Returns:
        images (np.ndarray): clean images 
        adv_images (np.ndarray) crafted image attacked by adversarial attacks 
        true_labels (np.ndarray): labels that indicate true values
    """    
    attack = None
    
    #load images and labels
    images, labels = zip(*[data_loader.dataset[i] for i in range(len(data_loader.dataset))])
    images = torch.stack(images).numpy() 
    true_labels = np.array(labels)
    
    #FGSM, DeepFool, C&W, UAP
    if attack_name == "FGSM":
        attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=32)
    elif attack_name == "BIM":
        attack = BasicIterativeMethod(estimator=classifier, eps=eps, batch_size=32)
    elif attack_name == "DeepFool":
        attack = DeepFool(classifier=classifier, epsilon=eps, batch_size=32, max_iter=10)
    elif attack_name == "CW":
         attack = CarliniL2Method(classifier=classifier, batch_size=32, max_iter=10)
        #true_labels = __get_one_hot(true_labels, nb_class)
    elif attack_name == "PGD":
        attack = ProjectedGradientDescent(estimator=classifier, eps=eps, batch_size=32)    
    elif attack_name == "UAP":
        attack = UniversalPerturbation(classifier=classifier, attacker="pgd", eps=eps, max_iter=10, batch_size=32)
    elif attack_name == "Wasserstein":
        attack = Wasserstein(estimator=classifier, eps=eps, max_iter=10, batch_size=32)
    
    adv_attack = attack.generate(x=images)
    
    return images, adv_attack, true_labels

def run_attack(val_attack_dataset, dataset_name, num_class, weights_path, model_name, input_size, attack_name, eps, batch_size, lr, save_metrics_path):    
    """running attack on dataset and target model

    Args:
        val_attack_dataset (torch.utils.data.Dataloader): dataloader in validation phase to attack dataset
        dataset_name (str): target dataset name
        num_class (int): number of class
        weights_path (str): models weights root path
        model_name (str): target model name
        input_size (tuple): image size (128, 128)
        attack_name (str): attack will be applied
        eps (float): noise level (e.g. 0.001)
        batch_size (int): batch of images size 
        lr (float): learning reate of the model
        save_metrics_path (str): path to save metrics generate by attacks

    Returns:
        _type_: _description_
        images (np.ndarray): clean images 
        adv_images (np.ndarray) crafted image attacked by adversarial attacks 
        true_labels (np.ndarray): labels that indicate true values
    """
    #2nd read models from checkpoints
    model_path = os.path.join(weights_path, "{}-{}-exp1.ckpt".format(model_name, dataset_name))
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=num_class)
            
    #3rd run attack
    time_start = time.time()
    images, adv_images, true_labels = generate_attack(
                                    model=model,
                                    input_shape=input_size,
                                    lr=lr,
                                    nb_class=num_class,
                                    attack_name=attack_name,
                                    data_loader=val_attack_dataset,
                                    eps=eps)
    
    final_time = time.time() - time_start
                
    #4th convert images and labels to dataloader
    loader_clean = utils.numpy_to_dataloader(images=images, labels=true_labels, batch_size=batch_size)
    loader_adv = utils.numpy_to_dataloader(images=adv_images, labels=true_labels, batch_size=batch_size)
                
    #5th evaluate accuracy of the models
    metrics_epochs = evaluate.evaluate_model(model=model,
                                            dataset=loader_adv, 
                                            nb_class=num_class,
                                            is_attacked=True)
    size = len(metrics_epochs["epochs"])
                
    #6th define metrics
    metrics_avg = pd.DataFrame([{"model": model_name,
                                 "dataset": dataset_name, 
                                 "attack": attack_name, 
                                 "eps": eps, 
                                 "val_acc_adv": metrics_epochs["val_acc"].mean(),
                                 "val_auc_adv": metrics_epochs["val_auc"].mean(), 
                                 "asr": metrics_epochs["asr"].mean()}])
    
    metrics_time = pd.DataFrame([{"attack": attack_name,
                                  "dataset": dataset_name, 
                                  "examples": len(images),
                                  "time": final_time}])
    #7th save metrics to dataframe
    metrics_epochs["model"] = np.repeat(model_name, size)
    metrics_epochs["attack"] = np.repeat(attack_name, size)
    metrics_epochs["dataset"] = np.repeat(dataset_name, size)
    metrics_epochs["eps"] = np.repeat(eps, size)
       
    #8th define path to save metrics
    avg_path = os.path.join(save_metrics_path, "attacks_avg.csv")
    epochs_path = os.path.join(save_metrics_path, "attacks_epochs.csv")
    time_path = os.path.join(save_metrics_path, "attacks_time.csv")
    
    #9th save metrics to CSV file
    metrics_avg.to_csv(avg_path, index=False, mode="a", header=False if os.path.exists(avg_path) else True)
    metrics_epochs.to_csv(epochs_path, index=False, mode="a", header=False if os.path.exists(epochs_path) else True)
    metrics_time.to_csv(time_path, index=False, mode="a", header=False if os.path.exists(time_path) else True)
    
    #10th save logits to numpy fil
            
    return images, adv_images, true_labels
    