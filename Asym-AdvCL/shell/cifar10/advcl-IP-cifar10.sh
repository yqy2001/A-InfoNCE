#!/bin/bash

python pretraining_advCL.py --dataset cifar10 --cosine --name $1 --epoch 400 --stpg_degree 0.2 --stop_grad --adv_weight 1.2 --stop_grad_adaptive 30 --learning_rate 1.

python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.01 --finetune_type SLF
python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.01 --finetune_type ALF
python finetuning_advCL_SLF.py --dataset cifar10 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type AFF_trades

# sh shell/cifar10/advcl-IP-cifar10.sh advcl-IP-cifar10 | tee logs/advcl-IP-cifar10.out