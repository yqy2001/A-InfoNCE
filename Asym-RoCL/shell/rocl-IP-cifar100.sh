#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12567 \
    rocl_train.py --ngpu 2 --batch-size=256 --model=$2 --k=7 --loss_type=sim --advtrain_type=Rep --attack_type=linf \
    --name=$1 --regularize_to=other --attack_to=other --train_type=contrastive --dataset=$3 --stop_grad --stpg_degree 0.2

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12567 \
    linear_eval.py --ngpu 1 --batch-size=1024 --train_type=linear_eval --model=$2 --epoch 150 --lr 0.1 --name $1 \
    --load_checkpoint=checkpoint/ckpt.t7rocl-IP-cifar100Rep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-100_b256_nGPU2_l256 \
    --clean=True --dataset=$3

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12567 \
    robustness_test.py --ngpu 1 --train_type=linear_eval --name=$1 --batch-size=1024 --model=$2 \
    --load_checkpoint='./checkpoint/ckpt.t7'$1'_Evaluate_linear_eval_ResNet18_'$3 --attack_type=linf --epsilon=0.0314 --alpha=0.00314 --k=20 --dataset=$3

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12357 \
    rocl_finetune.py --ngpu 1 --batch-size=1024 --finetune_type ALF --model=$2 --epoch 25 --lr 0.1 --name $1 \
    --load_checkpoint=checkpoint/ckpt.t7rocl-IP-cifar100Rep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-100_b256_nGPU2_l256 \
    --dataset=$3 --epsilon=0.0314 --alpha=0.00314 --k=20

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12357 \
    rocl_finetune.py --ngpu 1 --batch-size=1024 --finetune_type AFF_trades --model=$2 --epoch 25 --lr 0.1 --name $1 \
    --load_checkpoint=checkpoint/ckpt.t7rocl-IP-cifar100Rep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-100_b256_nGPU2_l256 \
    --dataset=$3 --epsilon=0.0314 --alpha=0.00314 --k=20

# sh shell/rocl-IP-cifar100.sh rocl-IP-cifar100 ResNet18 cifar-100 | tee logs/rocl-IP-cifar100.out
