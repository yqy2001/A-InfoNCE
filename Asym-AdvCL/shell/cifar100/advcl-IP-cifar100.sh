#!/bin/bash

python pretraining_advCL.py --learning_rate 2. --dataset cifar100 --cosine --name $1 --epoch 400 --stpg_degree 0.3 --stop_grad --adv_weight 1.5 --stop_grad_adaptive 20 --d_min 0.5

python finetuning_advCL_SLF.py --dataset cifar100 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type SLF
python finetuning_advCL_SLF.py --dataset cifar100 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type ALF
python finetuning_advCL_SLF.py --dataset cifar100 --ckpt checkpoint/$1/epoch_400.ckpt --name $1 --learning_rate 0.1 --finetune_type AFF_trades

# sh shell/cifar100/advcl-IP-cifar100.sh advcl-IP-cifar100 | tee logs/advcl-IP-cifar100.out