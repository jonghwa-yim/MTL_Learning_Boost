#!/usr/bin/env bash


#python main_train.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_normal_reverse --gpu=0 --batchSize=16 --nEpochs=18 --lr=0.01 --lr_decay_ep=8 --resume=./output/cityscapes_base/model_epoch_12.pth --reverse > cityscapes_normal_reverse_log2.txt
#python main_train.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_normal_reverse --gpu=0 --batchSize=16 --nEpochs=22 --lr=0.01 --lr_decay_ep=12 --resume=./output/cityscapes_base/model_epoch_8.pth --reverse > cityscapes_normal_reverse_log3.txt
#python main_train.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_normal_reverse --gpu=0 --batchSize=16 --nEpochs=26 --lr=0.01 --lr_decay_ep=16 --resume=./output/cityscapes_base/model_epoch_4.pth --reverse > cityscapes_normal_reverse_log4.txt


#python main_train.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_normal --gpu=0 --batchSize=16 --nEpochs=18 --lr=0.01 --lr_decay_ep=8 --resume=./output/cityscapes_base/model_epoch_12.pth > cityscapes_normal_log2.txt
#python main_train.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_normal --gpu=0 --batchSize=16 --nEpochs=22 --lr=0.01 --lr_decay_ep=12 --resume=./output/cityscapes_base/model_epoch_8.pth > cityscapes_normal_log3.txt
#python main_train.py --dataset=cityscapes --model=unet --outputDir=./output/cityscapes_normal --gpu=0 --batchSize=16 --nEpochs=26 --lr=0.01 --lr_decay_ep=16 --resume=./output/cityscapes_base/model_epoch_4.pth > cityscapes_normal_log4.txt





python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=40 --lr=0.001 --lr_decay_ep=0 --resume=./output/nyuv2_base/model_epoch_20.pth --reverse > nyuv2_normal_reverse_log0.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=42 --lr=0.01 --lr_decay_ep=2 --resume=./output/nyuv2_base/model_epoch_18.pth --reverse > nyuv2_normal_reverse_log1.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=44 --lr=0.01 --lr_decay_ep=4 --resume=./output/nyuv2_base/model_epoch_16.pth --reverse > nyuv2_normal_reverse_log2.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=46 --lr=0.01 --lr_decay_ep=6 --resume=./output/nyuv2_base/model_epoch_14.pth --reverse > nyuv2_normal_reverse_log3.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=48 --lr=0.01 --lr_decay_ep=8 --resume=./output/nyuv2_base/model_epoch_12.pth --reverse > nyuv2_normal_reverse_log4.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=50 --lr=0.01 --lr_decay_ep=10 --resume=./output/nyuv2_base/model_epoch_10.pth --reverse > nyuv2_normal_reverse_log5.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=52 --lr=0.01 --lr_decay_ep=12 --resume=./output/nyuv2_base/model_epoch_8.pth --reverse > nyuv2_normal_reverse_log6.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=54 --lr=0.01 --lr_decay_ep=14 --resume=./output/nyuv2_base/model_epoch_6.pth --reverse > nyuv2_normal_reverse_log7.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal_reverse --nEpochs=56 --lr=0.01 --lr_decay_ep=16 --resume=./output/nyuv2_base/model_epoch_4.pth --reverse > nyuv2_normal_reverse_log8.txt

python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=40 --lr=0.001 --lr_decay_ep=0 --resume=./output/nyuv2_base/model_epoch_20.pth > nyuv2_normal_log0.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=42 --lr=0.01 --lr_decay_ep=2 --resume=./output/nyuv2_base/model_epoch_18.pth > nyuv2_normal_log1.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=44 --lr=0.01 --lr_decay_ep=4 --resume=./output/nyuv2_base/model_epoch_16.pth > nyuv2_normal_log2.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=46 --lr=0.01 --lr_decay_ep=6 --resume=./output/nyuv2_base/model_epoch_14.pth > nyuv2_normal_log3.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=48 --lr=0.01 --lr_decay_ep=8 --resume=./output/nyuv2_base/model_epoch_12.pth > nyuv2_normal_log4.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=50 --lr=0.01 --lr_decay_ep=10 --resume=./output/nyuv2_base/model_epoch_10.pth > nyuv2_normal_log5.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=52 --lr=0.01 --lr_decay_ep=12 --resume=./output/nyuv2_base/model_epoch_8.pth > nyuv2_normal_log6.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=54 --lr=0.01 --lr_decay_ep=14 --resume=./output/nyuv2_base/model_epoch_6.pth > nyuv2_normal_log7.txt
python main_train.py --dataset=nyuv2 --model=unet --outputDir=./output/nyuv2_normal --nEpochs=56 --lr=0.01 --lr_decay_ep=16 --resume=./output/nyuv2_base/model_epoch_4.pth > nyuv2_normal_log8.txt



