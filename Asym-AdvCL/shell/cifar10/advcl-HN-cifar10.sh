#!/bin/bash

python pretraining_advCL.py --learning_rate 1.2 --dataset cifar10 --cosine --name $1 --epoch 400 --attack_ori --HN --beta 1.0 --tau_plus 0.12

python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.01 --finetune_type SLF
python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.01 --finetune_type ALF
python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type AFF_trades

# sh shell/cifar10/advcl-HN-cifar10.sh advcl-HN-cifar10 | tee logs/advcl-HN-cifar10.out