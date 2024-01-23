import sys
sys.path.append("../utils")

from utils import utils
import generate_attacks
import numpy as np
import pandas as pd
import os
import time


# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-dm','--dataset_name', help='databaset name')
# parser.add_argument('-d','--dataset', help='databaset path', required=False)
# parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)

# # parser.add_argument('-mn', '--model_name', help="model to training name: resnet50 or resnet18", required=True)
# parser.add_argument('-wp', '--weights_path', help="root of model weigths path", required=True)

# # parser.add_argument('-an', '--attack_name', help="Attack name FGSM, PGD, CW or UAP", required=True)
# # parser.add_argument('-e', '--eps', help="Attack noise", required=True)
# #parser.add_argument('-pa', '--path_attack', help="Attack noise", required=True)

# args = vars(parser.parse_args())

def run_attack(root_path, dataset_name, csv_path, weights_path, model_name, attack_name, eps, batch_size, lr):    
   
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
    
    
    