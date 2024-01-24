import sys
sys.path.append('./evasion_attack')
sys.path.append("./ood_analysis")
sys.path.append("./utils")
sys.path.append("./explain_attacks")

from evasion_attack import generate_attacks
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
    models = ["resnet50"] #["resnet50", "vgg16","vgg19","inceptionv3", "efficientnet", "densenet"]
    attacks = ["FGSM"] #["FGSM", "BIM", "PGD", "DeepFool", "UAP", "CW"] 
    epsilons = [0.5] #[0.001, 0.01, 0.05, 0.1, 0.5]
    
    for model_name in models:
        print("Starting attack for model {}...".format(model_name))
        input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
        for attack_name in attacks:
            print("Generate attacked images using attack {}...".format(attack_name))
            for eps in epsilons: 
                print("The eps is {}".format(str(eps)))
                #5th run attack
                images, adv_images, true_labels = generate_attacks.run_attack(root_path=root_path, 
                                                                              dataset_name=dataset_name, 
                                                                              csv_path=csv_path, 
                                                                              weights_path=weights_path, 
                                                                              model_name=model_name,
                                                                              input_size=input_size,
                                                                              attack_name=attack_name, 
                                                                              eps=eps, 
                                                                              batch_size=batch_size, 
                                                                              lr=lr,
                                                                              save_metrics_path="./metrics",
                                                                              is_logits_save=False,
                                                                              is_features_save=False)
                
                print(len(images))
                model_path = os.path.join(weights_path, "{}-{}-exp1.ckpt".format(model_name, dataset_name))
                explain_module.shap_explainer(model_path=model_path, model_name=model_name, nb_class=7, images=images, labels=true_labels)
                
                #utils.save_all_adv_image(path_to_save="./dataset/attacks", images_array=adv_images, labels=true_labels, db_name=dataset_name, attack_name=attack_name, model_name=model_name, eps=str(eps))
                #utils.save_all_adv_image(path_to_save="./dataset/attacks", images_array=adv_images, labels=true_labels, db_name=dataset_name, attack_name="None", model_name=model_name, eps=str(eps))
                
                #utils.show_random_adv_image(adv_images[:32], dataset_name, attack_name, eps,path_to_save="./metrics/figures/attacks")
    