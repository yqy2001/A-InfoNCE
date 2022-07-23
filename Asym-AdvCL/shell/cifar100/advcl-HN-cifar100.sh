#!/bin/bash

python pretraining_advCL.py --learning_rate 2. --dataset cifar100 --cosine --name $1 --epoch 400 --attack_ori --HN --beta 1. --tau_plus 0.01

python finetuning_advCL_SLF.py --dataset cifar100 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type SLF
python finetuning_advCL_SLF.py --dataset cifar100 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type ALF
python finetuning_advCL_SLF.py --dataset cifar100 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type AFF_trades

# sh shell/cifar100/advcl-HN-cifar100.sh advcl-HN-cifar100 | tee logs/advcl-HN-cifar100.out