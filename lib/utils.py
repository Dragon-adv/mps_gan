#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from lib.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from lib.sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from lib.sampling import cifar_iid, cifar100_noniid, cifar10_noniid, cifar100_noniid_lt, cifar10_noniid_lt
from lib.sampling import eurosat_noniid,eurosat_noniid_lt
from lib.sampling import tiny_noniid,tiny_noniid_lt
from lib.sampling import fashion_noniid,fashion_noniid_lt
import femnist
import numpy as np
import os
import realwaste
import random
from torch.utils.data import Subset
import seaborn as sns
import matplotlib.pyplot as plt

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])
trans_tiny_val = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
def get_dataset(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = femnist_iid(train_dataset, args.num_users)
            # print("TBD")
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                # user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                user_groups = femnist_noniid_unequal(args, train_dataset, args.num_users)
                # print("TBD")
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users, n_list, k_list)
                user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list,classes_list_gt = cifar100_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar100_noniid_lt(test_dataset, args.num_users, classes_list)
    elif args.dataset=='eurosat' or args.dataset=='defungi':
        trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(root=data_dir, transform=trans)
        y=dataset.targets
        random.seed(42)
        dataset_len = len(dataset)
        train_len = int(0.8 * dataset_len)
        val_len = int(0.2 * dataset_len)
        indices = list(range(dataset_len))
        random.shuffle(indices)
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        train_dataset = Subset(dataset, train_indices)
        y_train=[y[i]for i in train_indices]
        test_dataset = Subset(dataset, val_indices)
        y_test=[y[i]for i in val_indices]

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = eurosat_noniid(args, train_dataset,y_train, args.num_users,n_list, k_list)
                user_groups_lt = eurosat_noniid_lt(args, test_dataset,y_test, args.num_users, classes_list)

    elif args.dataset == 'tinyimagenet':
        train_dataset = ImageFolder(data_dir+'/train/',  transform=trans_cifar100_train)
        test_dataset = ImageFolder(data_dir+'/val/', transform=trans_tiny_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list ,classes_list_gt= tiny_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = tiny_noniid_lt(args,test_dataset, args.num_users, classes_list)

    elif args.dataset=='realwaste':
        apply_transform_train =transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.59810066, 0.61901563, 0.63138026],
                                                                  std= [0.15483421, 0.1635018, 0.18204606])])
        apply_transform_test = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5975758, 0.619224, 0.63241196],
                                                                         std=[0.15579279, 0.16374466, 0.18181999])])
        train_dataset = realwaste.REALWASTE(args, data_dir, train=True, transform=apply_transform_train)
        test_dataset = realwaste.REALWASTE(args, data_dir, train=False, transform=apply_transform_test)


        user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users,n_list, k_list)
        user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset=='flowers':
        trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        dataset = ImageFolder(root=data_dir, transform=trans)
        y=dataset.targets
        random.seed(42)
        dataset_len = len(dataset)
        train_len = int(0.75 * dataset_len)
        val_len = int(0.25 * dataset_len)
        indices = list(range(dataset_len))
        random.shuffle(indices)
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        train_dataset = Subset(dataset, train_indices)
        y_train=[y[i]for i in train_indices]
        test_dataset = Subset(dataset, val_indices)
        y_test=[y[i]for i in val_indices]

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = eurosat_noniid(args, train_dataset,y_train, args.num_users,n_list, k_list)
                user_groups_lt = eurosat_noniid_lt(args, test_dataset,y_test, args.num_users, classes_list)

    elif args.dataset == 'fashion':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = fashion_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = fashion_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list
    '''
    # partition plot
    plt.rcParams['font.family'] = 'Times New Roman'
    values=np.zeros((args.num_classes,args.num_users))
    for user_id in range(len(classes_list)):
        for class_id in range(args.num_classes):
            if class_id in classes_list[user_id]:
                values[class_id,user_id]=k_list[user_id]

    x_ticks_pos = np.arange(args.num_users) + 0.5  
    y_ticks_pos = np.arange(args.num_classes) + 0.5  

    #  Generate a heatmap and disable automatic tick labels
    ax = sns.heatmap(values, cmap='Blues', xticklabels=False, yticklabels=False)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=18)#16

    # Set X-axis ticks and labels
    ax.set_xticks(x_ticks_pos[::2])
    ax.set_xticklabels(np.arange(args.num_users)[::2],fontsize=18)  
    ax.set_xlabel('Client ID',fontsize=22)

    # Set the y-axis scale and labels (example: keep an interval of 10)
    if args.dataset == 'femnist':
        ax.set_yticks(y_ticks_pos[::10])
        ax.set_yticklabels(np.arange(args.num_classes)[::10],fontsize=18)
    else:
        ax.set_yticks(y_ticks_pos)
        ax.set_yticklabels(np.arange(args.num_classes),fontsize=18)
    ax.set_ylabel('Class ID',fontsize=22)

    plt.xticks(rotation=0, ha='center')  
    plt.yticks(rotation=0, va='center') 

    figure = ax.get_figure()
    # figure.set_size_inches(2.3,1.91)
    figure.savefig(f'../plots/revise2/partition/{args.dataset}_n{args.ways}.pdf',bbox_inches='tight')
    plt.close(figure)
    '''


    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != '....':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return

def get_global_input(local_protos_dict):
    global_protos=[]
    global_labels=[]
    for id, protos_dict in local_protos_dict.items():
        for label,proto in protos_dict.items():
            global_protos.append(proto)
            global_labels.append(label)
    global_protos=torch.stack(global_protos)
    global_labels=torch.tensor(global_labels)
    return global_protos,global_labels

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass