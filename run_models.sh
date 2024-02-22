#!/bin/bash

#for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
for models in "resnet50" "vgg16" "vgg19" "inceptionv3" "densenet" "efficientnet"
do
    echo "model $models"
    python train_clean_model.py --dataset ./dataset/ISIC2020 --dataset_csv dataset/ISIC2020/ISIC_2020_dataset.csv --model_name $models --as_augmentation --epochs 50 --dataset_name ISIC2020 --as_rgb
done