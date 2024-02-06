import sys
sys.path.append("../utils")

from utils import utils
import generate_attacks
import numpy as np
import pandas as pd
import os
import time

def run_attack(root_path: str, dataset_name: str, csv_path: str, weights_path: str, model_name: str, attack_name: str, eps: float, batch_size: int, lr: float) -> None:
    """Run adversarial attack on selected dataset

    Args:
        root_path (str): root path of images
        dataset_name (str): target dataset name
        csv_path (str): csv that indicates dataset content
        weights_path (str): models weights root path
        model_name (str): target model name
        attack_name (str): attack will be applied
        eps (float): noise level (e.g. 0.001)
        batch_size (int): batch of images size 
        lr (float): learning reate of the model
    """
    input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
        
    print("Load validation database using {}...".format(dataset_name))
    #3rd read validation dataset to attack the model
    val_attack_dataset, num_class = utils.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=batch_size, image_size=input_size, percentage_attacked=0.2, test_size=0.3)
        
    #4th read models from checkpoints
    model_path = os.path.join(weights_path, "{}-{}-exp0.ckpt".format(model_name, dataset_name))
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=num_class)
        
    print("Generate attacked images using attack {}...".format(attack_name))
    print("The eps is {}".format(str(eps)))
    
    #5th run attack
    time_start = time.time()
    images, adv_images, true_labels = generate_attacks.generate_attack(
                            model=model,
                            input_shape=input_size,
                            lr=lr,
                            nb_class=num_class,
                            attack_name=attack_name,
                            data_loader=val_attack_dataset,
                            eps=eps
                        )
    final_time = time.time() - time_start
                
    #6th convert images and labels to dataloader
    loader_clean = utils.numpy_to_dataloader(images=images, labels=true_labels, batch_size=32)
    loader_adv = utils.numpy_to_dataloader(images=adv_images, labels=true_labels, batch_size=32)
                
    #7th evaluate accuracy of the models
    metrics_epochs = generate_attacks.evaluate_model(model=model, dataset_clean=loader_clean, dataset_adv=loader_adv)
    size = len(metrics_epochs["epochs"])
                
    #8th define metrics
    metrics_avg = pd.DataFrame([{"dataset": dataset_name, "model": model_name, "attack": attack_name, "eps": eps, "val_acc": metrics_epochs["val_acc"].mean(), "val_acc_adv": metrics_epochs["val_acc_adv"].mean()}])
    metrics_time = pd.DataFrame([{"dataset": dataset_name, "attack": attack_name, "examples": len(images),"time": final_time}])
    metrics_epochs["database"] = np.repeat(dataset_name, size)
    metrics_epochs["model"] = np.repeat(model_name, size)
    metrics_epochs["attack"] = np.repeat(attack_name, size)
    metrics_epochs["eps"] = np.repeat(eps, size)
                
    #9th save metrics to CSV
    metrics_avg.to_csv("../metrics/attacks_avg.csv", index=False, mode="a", header=False if os.path.exists("../metrics/attacks_avg.csv") else True)
    metrics_epochs.to_csv("../metrics/attacks_epochs.csv", index=False, mode="a", header=False if os.path.exists("../metrics/attacks_epochs.csv") else True)
    metrics_time.to_csv("../metrics/time_attack.csv", index=False, mode="a", header=False if os.path.exists("../metrics/time_attack.csv") else True)
    
    
    