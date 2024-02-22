import sys
# sys.path.append('./evasion_attack')
# sys.path.append("./ood_analysis")
# sys.path.append("./utils")
# sys.path.append("./explain_attacks")

from evasion_attack import  evaluate
from utils import utils
#import utils
import torch
import numpy as np
import argparse
import pandas as pd
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(weights_path, model_name, dataset_name, nb_class):    
    model_path = os.path.join(weights_path, "{}-{}-exp1.ckpt".format(model_name, dataset_name))
    model = utils.read_model_from_checkpoint(model_path=model_path, model_name=model_name, nb_class=nb_class)
    model = model.to(device)
    model.eval() 
    
    return model

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
    models = ["resnet50", "vgg16","vgg19","inceptionv3", "efficientnet", "densenet"]
    
    for model_name in models:
        print(model_name)
        input_size = (299, 299) if model_name == "inceptionv3" else (224, 224)
        
        #1st read validation dataset to attack the model
        val_dataset, num_class = utils.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=batch_size, image_size=input_size)
        
        model = create_model(model_name=model_name, weights_path=weights_path, dataset_name=dataset_name, nb_class=num_class)
        metrics_epochs = evaluate.evaluate_model(model=model,
                                                dataset=val_dataset,
                                                nb_class=num_class)
        
        size = len(metrics_epochs["epochs"])
                    
        #6th define metrics
        metrics_avg = pd.DataFrame([{"model": model_name,
                                    "dataset": dataset_name, 
                                    "val_acc": metrics_epochs["val_acc"].mean(),
                                    "val_auc": metrics_epochs["val_auc"].mean()}])
        
        avg_path = os.path.join("./metrics", "model_eval_avg.csv")
        metrics_avg.to_csv(avg_path, index=False, mode="a", header=False if os.path.exists(avg_path) else True)