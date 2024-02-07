import sys
sys.path.append('./evasion_attack')
sys.path.append("./ood_analysis")
sys.path.append("./utils")
sys.path.append("./explain_attacks")

from evasion_attack import generate_attacks, evaluate
from explain_attacks import explain_module
import utils
import torch
import numpy as np
import argparse
import pandas as pd
import os
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('-dm','--dataset_name', help='databaset name')
parser.add_argument('-d','--dataset', help='databaset path', required=False)
parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)

# parser.add_argument('-mn', '--model_name', help="model to training name: resnet50 or resnet18", required=True)
parser.add_argument('-wp', '--weights_path', help="root of model weigths path", required=True)

# parser.add_argument('-an', '--attack_name', help="Attack name FGSM, PGD, CW or UAP", required=True)
# parser.add_argument('-e', '--eps', help="Attack noise", required=True)
#parser.add_argument('-pa', '--path_attack', help="Attack noise", required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':

    torch.manual_seed(43)
    np.random.seed(43)
    
    #1st define de args
    dataset_name = args["dataset_name"]
    root_path = args["dataset"]
    csv_path = args["dataset_csv"]
    weights_path = args["weights_path"]
    
    #2nd define parameters
    batch_size = 32
    lr = 0.001
    model_name = "resnet50" #["resnet50", "vgg16","vgg19","inceptionv3", "efficientnet", "densenet"]
    attacks = ["FGSM", "BIM", "PGD", "DeepFool", "UAP", "CW"] 
    epsilons = [0.05, 0.5] #[0.001, 0.01, 0.05, 0.1, 0.5]
    class_names_path = "./dataset/MelanomaDB/class_name.json"
    save_metrics_path = "./metrics"
    
    #calculate eval metrics
    input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
    
    #1st read validation dataset to attack the model
    val_attack_dataset, num_class = utils.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=batch_size, image_size=input_size)
    
    print("Starting attack for model {}...".format(model_name))
    for attack_name in attacks:
        print("Generate attacked images using attack {}...".format(attack_name))
        for eps in epsilons: 
            print("The eps is {}".format(str(eps)))
            images, adv_images, true_labels = generate_attacks.run_attack(val_attack_dataset=val_attack_dataset, 
                                                                          dataset_name=dataset_name, 
                                                                          weights_path=weights_path,
                                                                          num_class=num_class, 
                                                                          model_name=model_name,
                                                                          input_size=input_size,
                                                                          attack_name=attack_name, 
                                                                          eps=eps, 
                                                                          batch_size=batch_size, 
                                                                          lr=lr,
                                                                          save_metrics_path="./metrics")
                
            
            path_to_save = f"./dataset/attacks/{dataset_name}/{attack_name}/{str(eps)}"
            os.makedirs(path_to_save, exist_ok=True)
            os.makedirs(os.path.join(path_to_save, "manifolds"), exist_ok=True)
            
            metrics  = explain_module.run_explainer(weights_path=weights_path, 
                                         model_name=model_name,
                                         dataset_name=dataset_name, 
                                         nb_class=7, 
                                         images_target=images,
                                         images_adv_target=adv_images, 
                                         labels_target=true_labels,
                                         class_names_path=class_names_path,
                                         root_save_path=path_to_save)
            
            metrics.insert(2, "Attack", attack_name)
            metrics.insert(3, "Eps", str(eps))
            
            metrics_path = os.path.join(save_metrics_path, "xai_metrics.csv")
            metrics.to_csv(metrics_path, index=False, mode="a", header=False if os.path.exists(metrics_path) else True)
            
            explain_module.manifold_visualization(weights_path=weights_path,
                                                  attack_title=attack_name,
                                                  dataset_name=dataset_name,
                                                  model_name=model_name,
                                                  nb_class=7, 
                                                  images_target=images,
                                                  images_adv_target=adv_images, 
                                                  labels_target=true_labels,
                                                  class_names_path=class_names_path,
                                                  root_save_path=os.path.join(path_to_save, "manifolds"))
                
                #utils.save_all_adv_image(path_to_save="./dataset/attacks", images_array=adv_images, labels=true_labels, db_name=dataset_name, attack_name=attack_name, model_name=model_name, eps=str(eps))
                #utils.save_all_adv_image(path_to_save="./dataset/attacks", images_array=adv_images, labels=true_labels, db_name=dataset_name, attack_name="None", model_name=model_name, eps=str(eps))
                
                #utils.show_random_adv_image(adv_images[:32], dataset_name, attack_name, eps,path_to_save="./metrics/figures/attacks")
    