#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
from logging import raiseExceptions
import os
import json
import copy

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import data_loader
import model_loader
import models
from models.projector import Projector

from argument import print_args
from utils import progress_bar, checkpoint
from collections import OrderedDict
from attack_lib import FastGradientSignUntargeted
from loss import pairwise_similarity, NT_xent

from trades import trades_loss

def finetune_parser():
    parser = argparse.ArgumentParser(description='RoCL finetune')

    ##### arguments for RoCL Linear eval (LE) or Robust Linear eval (r-LE)#####
    parser.add_argument('--train_type', type=str, default='linear_eval')
    parser.add_argument('--finetune_type', type=str, default='SLF', choices=['SLF', 'ALF', 'AFF', 'AFF_trades'])

    parser.add_argument('--epochwise', type=bool, default=False, help='epochwise saving...')
    parser.add_argument('--ss', default=False, type=bool, help='using self-supervised learning loss')

    parser.add_argument('--trans', default=False, type=bool, help='use transformed sample')
    parser.add_argument('--clean', default=False, type=bool, help='use clean sample')

    ##### arguments for training #####
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_multiplier', default=15.0, type=float, help='learning rate multiplier')

    parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/cifar-100')
    parser.add_argument('--load_checkpoint', default='./checkpoint/ckpt.t7one_task_0', type=str, help='PATH TO CHECKPOINT')
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type ResNet18/ResNet50')

    parser.add_argument('--name', default='', type=str, help='name of run')
    parser.add_argument('--seed', default=2342, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size / multi-gpu setting: batch per gpu')
    parser.add_argument('--epoch', default=25, type=int,
                        help='total epochs to run')

    # optimization
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    ##### arguments for data augmentation #####
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

    ##### arguments for distributted parallel #####
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=1)

    ##### arguments for PGD attack & Adversarial Training #####
    # parser.add_argument('--min', type=float, default=0.0,
    #     help='min for cliping image')
    # parser.add_argument('--max', type=float, default=1.0,
    #     help='max for cliping image')
    # parser.add_argument('--attack_type', type=str, default='linf',
    #     help='adversarial l_p')
    parser.add_argument('--epsilon', type=float, default=0.0314,
        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--alpha', type=float, default=0.007,
        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
    parser.add_argument('--k', type=int, default=10,
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--random_start', type=bool, default=True,
        help='True for PGD')
    args = parser.parse_args()

    return args

args = finetune_parser()
use_cuda = torch.cuda.is_available()
if use_cuda:
    ngpus_per_node = torch.cuda.device_count()

if args.local_rank % ngpus_per_node==0:
    print_args(args)

def print_status(string):
    if args.local_rank % ngpus_per_node == 0:
        print(string)

print_status(torch.cuda.device_count())
print_status('Using CUDA..')

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print_status('==> Preparing data..')
args.train_type = 'linear_eval'
trainloader, traindst, testloader, testdst = data_loader.get_dataset(args)

if args.dataset == 'cifar-10' or args.dataset=='mnist':
    num_outputs = 10
elif args.dataset == 'cifar-100':
    num_outputs = 100

if args.model == 'ResNet50':
    expansion = 4
else:
    expansion = 1

# Model
print_status('==> Building model..')
finetune_type  = args.finetune_type

# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, classifier, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.classifier = classifier
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, inputs, targets, train=True, finetune_type="AFF"):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        if train:
            num_step = 10
        else:
            num_step = 20
        for i in range(num_step):
            x.requires_grad_()
            with torch.enable_grad():
                features = self.model(x)
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        if finetune_type == "SLF" or finetune_type == "ALF":
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)
        return self.classifier(features), x

def load(args, epoch):
    model = model_loader.get_model(args)

    if epoch == 0:
        add = ''
    else:
        add = '_epoch_'+str(epoch)

    checkpoint_ = torch.load(args.load_checkpoint+add)

    new_state_dict = OrderedDict()
    for k, v in checkpoint_['model'].items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    
    if args.ss:
        projector = Projector(expansion=expansion)
        checkpoint_p = torch.load(args.load_checkpoint+'_projector'+add)
        new_state_dict = OrderedDict()
        for k, v in checkpoint_p['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        projector.load_state_dict(new_state_dict)
    
    if args.dataset=='cifar-10':
        Linear = nn.Sequential(nn.Linear(512*expansion, 10))
    elif args.dataset=='cifar-100':
        Linear = nn.Sequential(nn.Linear(512*expansion, 100))

    model_params = []
    if args.finetune_type == "AFF" or args.finetune_type == "AFF_trades":
        model_params += model.parameters()
        if args.ss:
            model_params += projector.parameters()
    model_params += Linear.parameters()
    loptim = torch.optim.SGD(model_params, lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        ngpus_per_node = torch.cuda.device_count()
        model.cuda()
        Linear.cuda()
        model = nn.DataParallel(model)
        Linear = nn.DataParallel(Linear)
        if args.ss:
            projector.cuda()
            projector = nn.DataParallel(projector)
    else:
        assert("Need to use GPU...")

    print_status('Using CUDA..')
    cudnn.benchmark = True

    if args.finetune_type == "ALF" or args.finetune_type == "AFF":
        config = {
                'epsilon': 8.0 / 255.,
                'num_steps': 10,
                'step_size': 2.0 / 255,
                'random_start': True,
                'loss_func': 'xent',
        }
        attack_info = 'Adv_train_epsilon_'+str(config['epsilon'])+'_num_steps_' \
            + str(config['num_steps']) + '_step_size_' + str(config['step_size']) \
            + '_type_' + str(args.finetune_type) + '_randomstart_' + str(config['random_start'])
        print_status("Adversarial training info...")
        print_status(attack_info)

        # attacker = FastGradientSignUntargeted(model, linear=Linear, epsilon=args.epsilon, alpha=args.alpha, min_val=args.min, max_val=args.max, max_iters=args.k, _type=args.attack_type)
        attacker = AttackPGD(model, Linear, config)
        attacker = attacker.cuda()
        cudnn.benchmark = True

    if args.finetune_type == "ALF" or args.finetune_type == "AFF":
        if args.ss:
            return model, Linear, projector, loptim, attacker
        return model, Linear, 'None', loptim, attacker
    if args.ss:
        return model, Linear, projector, loptim, 'None'
    return model, Linear, 'None', loptim, 'None'

criterion = nn.CrossEntropyLoss()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def linear_train(epoch, model, Linear, projector, loptim, net=None):
    Linear.train()
    
    # ss
    if args.finetune_type == "AFF" or args.finetune_type == "AFF_trades":
        model.train()
        if args.ss:
            projector.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (ori, inputs, inputs_2, target) in enumerate(trainloader):
        ori, inputs_1, inputs_2, target = ori.cuda(), inputs.cuda(), inputs_2.cuda(), target.cuda()

        if args.trans:
            inputs = inputs_1
        else:
            inputs = ori

        # if args.adv_img:
        #     advinputs      = attacker.perturb(original_images=inputs, labels=target, random_start=args.random_start)

        if args.finetune_type == "SLF":
            with torch.no_grad():
                features = model(inputs)
            output = Linear(features.detach())
            loss = criterion(output, target)
        elif args.finetune_type == "ALF":  # ALF, use PGD examples to train the classifier
            output, _ = net(inputs, target, train=False, finetune_type="ALF")
            loss = criterion(output, target)
        elif args.finetune_type == "AFF":
            _, x_adv = net(inputs, target, train=False, finetune_type="AFF")
            # calculate robust loss
            beta = 6.0
            trainmode = 'adv'

            # fixmode f1: fix nothing, f2: fix previous 3 stages, f3: fix all except fc')
            loss, output = trades_loss(model=model,
                                       classifier=Linear,
                                       x_natural=inputs,
                                       x_adv=x_adv,
                                       y=target,
                                       optimizer=loptim,
                                       beta=beta,
                                       trainmode=trainmode,
                                       fixmode='f1')
        elif args.finetune_type == "AFF_trades":
            # calculate robust loss
            step_size = 2. / 255.
            epsilon = 8. / 255.
            num_steps_train = 10
            beta = 6.0
            trainmode = 'adv'
            # fixmode f1: fix nothing, f2: fix previous 3 stages, f3: fix all except fc')
            loss, output = trades_loss(model=model,
                                       classifier=Linear,
                                       x_natural=inputs,
                                       x_adv="",
                                       y=target,
                                       optimizer=loptim,
                                       step_size=step_size,
                                       epsilon=epsilon,
                                       perturb_steps=num_steps_train,
                                       beta=beta,
                                       trainmode=trainmode,
                                       fixmode='f1',
                                       trades=True)
        else:
            raise NameError(args.finetune_type) from Exception

        if args.clean:
            total_inputs = inputs
            total_targets = target
        
        if args.ss:
            total_inputs = torch.cat((inputs, inputs_2))
            total_targets = torch.cat((target, target))

        if args.ss:
            feat   = model(total_inputs)
            output_p = projector(feat)
            B = ori.size(0)

            similarity, _ = pairwise_similarity(output_p[:2*B,:2*B], temperature=args.temperature, multi_gpu=False, adv_type = 'None')
            simloss  = NT_xent(similarity, 'None')
            loss += simloss

        # correct += predx.eq(total_targets.data).cpu().sum().item()
        # total += total_targets.size(0)
        # acc = 100.*correct/total

        acc, acc5 = accuracy(output, target, topk=(1, 5))
        acc = acc.cpu().item()
        total_loss += loss.data

        loptim.zero_grad()
        loss.backward()
        loptim.step()
        
        progress_bar(batch_idx, len(trainloader),
                    'Loss: {:.4f} | Acc: {:.2f}'.format(total_loss/(batch_idx+1), acc))

    print ("Epoch: {}, train accuracy: {}".format(epoch, acc))

    return acc, model, Linear, projector, loptim

def test(model, Linear, net):
    global best_acc, best_acc_adv
    args.train_type = 'test'
    _, _, testloader, _  = data_loader.get_dataset(args)

    model.eval()
    Linear.eval()

    test_loss = 0
    
    correct = 0
    correct_adv = 0

    total = 0

    for idx, (image, label) in enumerate(testloader):
        img = image.cuda()
        _, img_adv = net(img, label.cuda(), train=False)

        y = label.cuda()

        out = Linear(model(img))
        out_adv =  Linear(model(img_adv))

        _, predx = torch.max(out.data, 1)
        _, predx_adv = torch.max(out_adv.data, 1)

        loss = criterion(out, y)

        correct += predx.eq(y.data).cpu().sum().item()
        correct_adv += predx_adv.eq(y.data).cpu().sum().item()

        total += y.size(0)
        acc = 100.*correct/total
        acc_adv = 100.*correct_adv/total

        test_loss += loss.data

        if args.local_rank % ngpus_per_node == 0:
            progress_bar(idx, len(testloader),'Testing Loss {:.3f}, acc {:.3f}, acc_adv {:.3f}'.format(test_loss/(idx+1), acc, acc_adv))
        
    print ("Test accuracy: {0}, Test accuracy(adv): {1}".format(acc, acc_adv))

    return (acc, acc_adv, model, Linear)

def adjust_lr(epoch, optim):
    lr = args.lr
    if args.dataset=='cifar-10' or args.dataset=='cifar-100':
        lr_list = [15, 20]
    if epoch>=lr_list[0]:
        lr = lr/10
    if epoch>=lr_list[1]:
        lr = lr/10
    
    for param_group in optim.param_groups:
        param_group['lr'] = lr

##### Log file for training selected tasks #####
if not os.path.isdir('results'):
    os.mkdir('results')

args.name += ('_Evaluate_'+ args.finetune_type + '_' +args.model + '_' + args.dataset)
loginfo = 'results/log_generalization_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train acc','test acc'])

if args.epochwise:
    for k in range(100,1000,100):
        model, linear, projector, loptim, attacker = load(args, k)
        print('loading.......epoch ', str(k))
        ##### Linear evaluation #####
        for i in range(args.epoch):
            print('Epoch ', i)
            train_acc, model, linear, projector, loptim = linear_train(i, model, linear, projector, loptim, attacker)
            test_acc, model, linear = test(model, linear)
            adjust_lr(i, loptim)

        checkpoint(model, test_acc, args.epoch, args, loptim, save_name_add='epochwise'+str(k))
        checkpoint(linear, test_acc, args.epoch, args, loptim, save_name_add='epochwise'+str(k)+'_linear')
        if args.local_rank % ngpus_per_node == 0:
            with open(logname, 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([k, train_acc, test_acc])

model, linear, projector, loptim, attacker = load(args, 0)
config = {
        'epsilon': args.epsilon,
        'num_steps': args.k,
        'step_size': args.alpha,
        'random_start': args.random_start,
        'loss_func': 'xent',
}
attack_val = AttackPGD(model, linear, config)
attack_val = attack_val.cuda()

best_acc = 0
best_acc_adv = 0
##### Linear evaluation #####
for epoch in range(args.epoch):
    print('Epoch ', epoch)

    train_acc, model, linear, projector, loptim = linear_train(epoch, model=model, Linear=linear, projector=projector, loptim=loptim, net=attacker)
    test_acc, test_acc_adv, model, linear = test(model, linear, attack_val)
    adjust_lr(epoch, loptim)
    
    if test_acc_adv > best_acc_adv:
        best_acc = test_acc
        best_acc_adv = test_acc_adv
    
    if args.local_rank % ngpus_per_node == 0:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_acc, test_acc, test_acc_adv])

    print('best accuracy: {:.2f}'.format(best_acc_adv))
    print('best accuracy clean: {:.2f}'.format(best_acc))

checkpoint(model, args.finetune_type, test_acc, test_acc_adv, args.epoch, args, loptim)
checkpoint(linear, args.finetune_type, test_acc, test_acc_adv, args.epoch, args, loptim, save_name_add='_linear')

if args.local_rank % ngpus_per_node == 0:
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([1000, train_acc, test_acc])

