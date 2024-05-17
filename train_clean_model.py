import argparse
import pandas as pd
import torch
import numpy as np
import os
import sys
# sys.path.append("./utils")
# sys.path.append("./clean_train")


from utils import utils
from clean_train.train_model import PytorchTrainingAndTest
from clean_train.models import ModelsPretrained

#Update CUDA version
#pip install --user torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Device to run: {device}")

RANDOM_SEED = 43

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='')
parser.add_argument('-dv','--dataset_csv', help='databaset csv file', required=False)
parser.add_argument('-r', '--as_rgb', action="store_true", required=False)
parser.add_argument('-kf', '--kfold', help="number of folds",required=False)
parser.add_argument('-au', '--as_augmentation', action="store_true", required=False)
parser.add_argument("-t", "--test_size", required=False)
parser.add_argument("-pc", "--as_per_class", action="store_true", required=False)
parser.add_argument('-pre', '--as_pretrained', action="store_true", required=False)

parser.add_argument('-d','--dataset', help='databaset path')
parser.add_argument("-mn", "--model_name")
parser.add_argument("-ep", "--epochs")
parser.add_argument('-dm','--dataset_name', help='databaset name')
args = vars(parser.parse_args())
    
if __name__ == '__main__':

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    train_with_pytorch = PytorchTrainingAndTest()
    models_load = ModelsPretrained()
    
    batch_size = 32
    learning_rate = 0.0001
    
    #Variables to run experiments
    num_epochs = int(args["epochs"])
    test_size = None if args["test_size"] is None else float(args["test_size"])
    as_aug = args["as_augmentation"]
    kfold = int(args["kfold"]) if not args["kfold"] is None else None
    model_name = args["model_name"]
    as_rgb = args["as_rgb"]
    is_per_class = args["as_per_class"]
    image_size = (299, 299) if model_name == "inceptionv3" else (224, 224)

    base_path = args["dataset"]  
    csv_path = args["dataset_csv"]
    database_name = args["dataset_name"]
    is_pretrained = args["as_pretrained"]
      
    print("Database: {}".format(database_name))
    print("Loading database ...")
    
    if not kfold is None:
        train, test, num_class = utils.load_database_kf(root_path=base_path, 
                                                        batch_size=batch_size, 
                                                        image_size=image_size, 
                                                        csv_path=csv_path, 
                                                        is_agumentation=as_aug, 
                                                        n_folds=kfold, as_rgb=as_rgb, 
                                                        is_pretrained=is_pretrained)
        utils.show_images(train[0], database_name, "./metrics/figures")
    elif not csv_path is None:
        train, test, num_class = utils.load_database_df(root_path=base_path, 
                                                        batch_size=batch_size, 
                                                        image_size=image_size, 
                                                        csv_path=csv_path, 
                                                        is_agumentation=as_aug, 
                                                        test_size=test_size, 
                                                        as_rgb=as_rgb, 
                                                        is_pretrained=is_pretrained)
        utils.show_images(train, database_name, "./metrics/figures")
            #train, test, num_class = utils.load_database_kf(path_image=base_path, batch_size=batch_size, image_size=image_size,  n_folds=5, csv_path=csv_path)
    else:
        train, test, num_class = utils.load_database(path=base_path, 
                                                     batch_size=batch_size, 
                                                     image_size=image_size, 
                                                     is_agumentation=as_aug, 
                                                     is_pretrained=is_pretrained)
        utils.show_images(train, database_name, "./metrics/figures")

    print(f"Number of class: {str(num_class)}")
    

    model_config = models_load.make_model_pretrained(model_name, num_class, is_pretrained=is_pretrained)
    print("\nNetwork: "+ model_name + " is training...\n")
    if not kfold is None:
        results = train_with_pytorch.run_model_kfold(exp_num=1,
                                                    model=model_config, 
                                                    model_name=model_name, 
                                                    database_name=database_name, 
                                                    train=train,
                                                    test=test, 
                                                    learning_rate=learning_rate, 
                                                    num_epochs=num_epochs, 
                                                    num_class=num_class)
    else:
        results = train_with_pytorch.run_model(exp_num=1,
                                              model=model_config, 
                                              model_name=model_name, 
                                              database_name=database_name, 
                                              train=train, 
                                              test=test, 
                                              learning_rate=learning_rate, 
                                              num_epochs=num_epochs,
                                              metrics_save_path="./metrics", 
                                              num_class=num_class,
                                              is_per_class=is_per_class)
  

    if os.path.exists("./metrics/results_{}.csv".format(database_name)):
        results.to_csv("./metrics/results_{}.csv".format(database_name), mode="a", header=None)
    else:
        results.to_csv("./metrics/results_{}.csv".format(database_name))
        
        