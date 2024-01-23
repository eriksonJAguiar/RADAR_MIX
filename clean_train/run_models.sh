#!/bin/bash

#for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python main.py --dataset ../ --dataset_csv ../dataset/OCT/OCT_dataset.csv --model_name $models --as_augmentation --epochs 50 --dataset_name OCT
done