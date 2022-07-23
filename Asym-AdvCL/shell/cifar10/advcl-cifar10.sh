#!/bin/bash

# follow the settings in [AdvCL](https://arxiv.org/pdf/2111.01124.pdf)

python pretraining_advCL.py --learning_rate 0.5 --dataset cifar10 --cosine --name $1 --epoch 400

python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type SLF
python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type ALF
python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type AFF_trades

# sh shell/cifar10/advcl-cifar10.sh advcl-cifar10 | tee logs/advcl-cifar10.out