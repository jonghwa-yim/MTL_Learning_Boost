#!/usr/bin/env bash


python main_train_single_only.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_single --gpu=0 --batchSize=16 --nEpochs=60 --lr=0.01 --lr_decay_ep=12 > cityscapes_single_log.txt

python main_train_single_only.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_single_reverse --gpu=0 --batchSize=16 --nEpochs=60 --lr=0.01 --lr_decay_ep=12 --reverse > cityscapes_single_reverse_log.txt
