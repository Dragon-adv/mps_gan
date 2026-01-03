#!/usr/bin/env python
# -*- coding: utf-8 -*-hello
# Python version: 3.6

import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path
from torch.utils.data import TensorDataset
import datetime
import logging
import pickle
import random
import numpy as np
import math
import json
import torch

# 将项目根目录添加到 sys.path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.options import *
from lib.update import *
from lib.models.models import *
from lib.utils import *
from lib.split_manager import load_split, save_split
from lib.checkpoint import (
    get_rng_state,
    set_rng_state,
    export_component_state_dicts,
    save_latest,
    save_best,
    load_checkpoint,
)
from lib.sfd_utils import RFF, aggregate_global_statistics
from lib.safs import MeanCovAligner, feature_synthesis, make_syn_nums
from lib.feature_generator import (
    DiversityLoss,
    StatsConditionedFeatureGenerator,
    gather_by_label,
    stack_low_global_stats,
)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Record console output
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
	    pass

def Fedavg(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, summary_writer,logger,logdir):

    idxs_users = np.arange(args.num_users)
    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w= local_model.update_weights_fedavg( idx=idx,model=copy.deepcopy(local_model_list[idx]))
            local_weights.append(copy.deepcopy(w))

        # aggregate local weights
        w_avg = copy.deepcopy(local_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[k] += local_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights))

        # Update each local model with the globally averaged parameters
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l= test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_lt,logger,summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | Test Acc: {:.5f}±{:.5f}, Test Loss: {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l), np.mean(loss_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
            net = copy.deepcopy(local_model_list[0])
            torch.save(net.state_dict(), logdir + '/localmodel0.pth')

    print('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    logger.info('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))


def Fedprox(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_prox(args,idx, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)
            local_weights.append(copy.deepcopy(w))

        # update global weights
        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights_list))

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l = test_inference_fedavg(args, round, local_model_list, test_dataset,user_groups_lt, logger, summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | Train Acc: {:.5f}, Test Acc: {:.5f}±{:.5f}, Train Loss: {:.5f}, Test Loss: {:.5f}'.format(
            round, np.mean(acc_list_train), np.mean(acc_list_l), np.std(acc_list_l), np.mean(loss_list_train), np.mean(loss_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Accuracy', {'train':np.mean(acc_list_train),'test':np.mean(acc_list_l)}, round)
        summary_writer.add_scalars('scalar/Total_Avg_Loss',{'train': np.mean(loss_list_train), 'test': np.mean(loss_list_l)}, round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
            net = copy.deepcopy(local_model_list[0])
            torch.save(net.state_dict(), logdir + '/localmodel0.pth')

def Moon(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,global_model,logger,summary_writer,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    old_nets_pool=[]#1

    if len(old_nets_pool) < 1:
        old_nets = copy.deepcopy(local_model_list)
        for net in old_nets:
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

    party_list_this_round = [i for i in range(args.num_users)]
    for round in tqdm(range(args.rounds)):

        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]

        for idx in idxs_users:
            prev_models = []
            for i in range(len(old_nets_pool)):
                prev_models.append(old_nets_pool[i][idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_moon(args,idx, model=copy.deepcopy(local_model_list[idx]),global_model=global_model,previous_models=prev_models, global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        total_data_points = sum([len(user_groups[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(user_groups[r]) / total_data_points for r in party_list_this_round]

        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for key, value in w_avg.items():
            w_avg[key] = value * fed_avg_freqs[0]
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]*fed_avg_freqs[i]

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        global_model.load_state_dict(w_avg)

        if len(old_nets_pool) < args.model_buffer_size:
            old_nets = copy.deepcopy(local_model_list)
            for  net in old_nets:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            old_nets_pool.append(old_nets)
        elif args.pool_option == 'FIFO':
            old_nets = copy.deepcopy(local_model_list)
            for net in old_nets:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            for i in range(args.model_buffer_size - 2, -1, -1):
                old_nets_pool[i] = old_nets_pool[i + 1]
            old_nets_pool[args.model_buffer_size - 1] = old_nets

        acc_list_l, loss_list_l,acc_list_g, loss_list,loss_total_list = test_inference_new_het_lt(args, local_model_list, test_dataset,classes_list, user_groups_lt)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | Test Acc (w/o protos): {:.5f}±{:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    logger.info('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))

def fedntd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w= local_model.update_weights_fedntd(args, idx=idx,model=copy.deepcopy(local_model_list[idx]))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        w_avg = copy.deepcopy(local_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[k] += local_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights))

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l= test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_lt,logger,summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    logger.info('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))

def fedgkd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, logdir):
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    models_buffer = []
    ensemble_model = None

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train = []
        loss_list_train = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_gkd(args, idx, model=copy.deepcopy(local_model_list[idx]), global_round=round, avg_teacher=ensemble_model)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)
            local_weights.append(copy.deepcopy(w))

        # update global avg weights for this round
        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights_list))

        # update global ensemble weights
        if len(models_buffer) >= args.buffer_length:
            models_buffer.pop(0)
        models_buffer.append(copy.deepcopy(w_avg))

        ensemble_w = copy.deepcopy(models_buffer[0])
        for k in ensemble_w.keys():
            for i in range(1, len(models_buffer)):
                ensemble_w[k] += models_buffer[i][k]
            ensemble_w[k] = torch.div(ensemble_w[k], len(models_buffer))

        if ensemble_model is None:
            ensemble_model = copy.deepcopy(local_model_list[0])
        ensemble_model.load_state_dict(ensemble_w, strict=True)

        # provide the client with the average model, not the ensemble model
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l = test_inference_fedavg(args, round, local_model_list, test_dataset, user_groups_lt, logger, summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        logger.info('| ROUND: {} | Train Acc: {:.5f}, Test Acc: {:.5f}±{:.5f}, Train Loss: {:.5f}, Test Loss: {:.5f}'.format(
            round, np.mean(acc_list_train), np.mean(acc_list_l), np.std(acc_list_l), np.mean(loss_list_train), np.mean(loss_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Accuracy',{'train': np.mean(acc_list_train), 'test': np.mean(acc_list_l)}, round)
        summary_writer.add_scalars('scalar/Total_Avg_Loss',{'train': np.mean(loss_list_train), 'test': np.mean(loss_list_l)}, round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
def Fedproc(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):

    idxs_users = np.arange(args.num_users)
    party_list_this_round = [i for i in range(args.num_users)]
    global_protos = []

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    for round in tqdm(range(args.rounds)):

        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]

        for idx in idxs_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos, idx_acc = local_model.update_weights_fedproc(args,idx, model=copy.deepcopy(local_model_list[idx]),global_protos=global_protos, global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss['1'])
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_protos[idx] = agg_protos

        # update global protos
        global_protos = proto_aggregation(local_protos)
        # update global weights
        total_data_points = sum([len(user_groups[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(user_groups[r]) / total_data_points for r in party_list_this_round]

        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for key, value in w_avg.items():
            w_avg[key] = value * fed_avg_freqs[0]
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]*fed_avg_freqs[i]

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model


        acc_list_l, loss_list_l,acc_list_g, loss_list,loss_total_list = test_inference_new_het_lt(args, local_model_list, test_dataset,classes_list, user_groups_lt)
        print('| ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
            round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    logger.info('| BEST ROUND: {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))


def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir):

    global_protos = []
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_acc_w=-float('inf')
    best_std_w=-float('inf')
    best_round = 0
    best_round_w=0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w,protos = local_model.update_weights_fedproto(args, idx, global_protos,model=copy.deepcopy(local_model_list[idx]),global_round=round)
            agg_protos = agg_func(protos)
            local_weights.append(copy.deepcopy(w))
            local_protos[idx] = agg_protos

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        # test
        acc_list_l, acc_list_g= test_inference_fedproto(args,logger, local_model_list, test_dataset,classes_list, user_groups_lt, global_protos)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | Test Acc (w/o protos): {:.5f}±{:.5f}, Test Acc (w/ protos): {:.5f}±{:.5f}'.format(
            round, np.mean(acc_list_l), np.std(acc_list_l), np.mean(acc_list_g), np.std(acc_list_g)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy_wp', np.mean(acc_list_g), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
        if np.mean(acc_list_g) > best_acc_w:
            best_acc_w = np.mean(acc_list_g)
            best_std_w = np.std(acc_list_g)
            best_round_w = round

    print('| BEST ROUND (w/o protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    logger.info('| BEST ROUND (w/o protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    print('| BEST ROUND (w/ protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round_w, best_acc_w, best_std_w))
    logger.info('| BEST ROUND (w/ protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round_w, best_acc_w, best_std_w))


def FedMPS(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,summary_writer,logger,logdir):
    """
    FedMPS 主训练函数（集成 ABBL）
    
    注意：确保 local_model_list 中的模型已经在主程序中正确移动到 GPU（通常在模型构建时完成）。
    """
    
    # ABBL: 初始化检查 - 确保所有本地模型已正确移动到设备
    for idx, model in enumerate(local_model_list):
        model_device = next(model.parameters()).device
        expected_device = torch.device(args.device)
        if model_device != expected_device:
            model.to(args.device)
    
    # global model: shares the same structure as the output layer of each local model
    global_model = GlobalFedmps(args)
    global_model.to(args.device)
    global_model.train()

    global_high_protos = {}
    global_low_protos = {}
    global_logits = {}
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf') # best results wo protos
    best_std = -float('inf')
    best_round = 0
    best_acc_w = -float('inf')  # best results w protos
    best_std_w = -float('inf')
    best_round_w = 0

    # Resume (Stage-1) state
    start_round = getattr(args, 'start_round', 0)
    resume_state = getattr(args, 'resume_state', None)
    if isinstance(resume_state, dict):
        global_high_protos = resume_state.get('global_high_protos', global_high_protos) or {}
        global_low_protos = resume_state.get('global_low_protos', global_low_protos) or {}
        global_logits = resume_state.get('global_logits', global_logits) or {}
        # Restore global classifier head weights if present (for consistent global_logits trajectory)
        global_model_sd = resume_state.get('global_model_state_dict', None)
        if isinstance(global_model_sd, dict) and len(global_model_sd) > 0:
            try:
                global_model.load_state_dict(global_model_sd, strict=True)
                print("Restored global_model (GlobalFedmps) weights from checkpoint.")
                if logger is not None:
                    logger.info("Restored global_model (GlobalFedmps) weights from checkpoint.")
            except Exception as e:
                print(f"Warning: failed to restore global_model weights from checkpoint ({e})")
                if logger is not None:
                    logger.info(f"Warning: failed to restore global_model weights from checkpoint ({e})")
        best_acc = resume_state.get('best_acc', best_acc)
        best_std = resume_state.get('best_std', best_std)
        best_round = resume_state.get('best_round', best_round)
        best_acc_w = resume_state.get('best_acc_w', best_acc_w)
        best_std_w = resume_state.get('best_std_w', best_std_w)
        best_round_w = resume_state.get('best_round_w', best_round_w)
    
    # ========== Initialize RFF models for SFD Statistics Aggregation ==========
    # This initialization only needs to be done once before the main loop
    # Step 1: Get feature dimensions by running a dummy forward pass
    # Use the first client's model to determine feature dimensions
    dummy_model = copy.deepcopy(local_model_list[0])
    dummy_model.eval()
    dummy_model = dummy_model.to(args.device)
    
    # Create a dummy input to get feature dimensions (adjust size based on dataset)
    if args.dataset == 'mnist' or args.dataset == 'femnist' or args.dataset == 'fashion':
        dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'realwaste' or args.dataset == 'flowers' or args.dataset == 'defungi':
        dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
    elif args.dataset == 'tinyimagenet':
        dummy_input = torch.randn(1, 3, 64, 64).to(args.device)
    elif args.dataset == 'imagenet':
        dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
    else:
        # Default to CIFAR-10 size
        dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
    
    with torch.no_grad():
        dummy_output = dummy_model(dummy_input)
        if isinstance(dummy_output, tuple) and len(dummy_output) >= 5:
            # 返回值顺序: logits, log_probs, high_level_features, low_level_features, projected_features
            high_feature_dim = dummy_output[2].shape[1]  # high-level feature dimension (索引2)
            low_feature_dim = dummy_output[3].shape[1]   # low-level feature dimension (索引3)
        else:
            raise ValueError(f"模型输出格式不符合预期，期望5个返回值，实际得到{len(dummy_output) if isinstance(dummy_output, tuple) else '非元组'}")
    
    # Step 2: Initialize RFF models for high and low levels
    # Set random seed for reproducibility
    backup_rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    
    # Set deterministic seed for RFF initialization
    random.seed(args.rf_seed)
    np.random.seed(args.rf_seed)
    torch.manual_seed(args.rf_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rf_seed)
    
    # Initialize RFF models
    rf_model_high = RFF(
        d=high_feature_dim,
        D=args.rf_dim_high,
        gamma=args.rbf_gamma_high,
        device=args.device,
        rf_type=args.rf_type
    )
    
    rf_model_low = RFF(
        d=low_feature_dim,
        D=args.rf_dim_low,
        gamma=args.rbf_gamma_low,
        device=args.device,
        rf_type=args.rf_type
    )
    
    rf_models = {
        'high': rf_model_high,
        'low': rf_model_low
    }
    
    # Restore random state
    random.setstate(backup_rng_state['python'])
    np.random.set_state(backup_rng_state['numpy'])
    torch.set_rng_state(backup_rng_state['torch'])
    
    logger.info(f'RFF Models: high(d={high_feature_dim}, D={args.rf_dim_high}, γ={args.rbf_gamma_high}), low(d={low_feature_dim}, D={args.rf_dim_low}, γ={args.rbf_gamma_low})')
    
    # Initialize global_stats to store the last round's statistics
    global_stats = None
    
    # ========== 保存数据分布元数据 (Metadata) ==========
    # 在训练开始前,收集并保存每个客户端的 pi_sample_per_class 和 classes_list
    # 收集每个客户端的 pi_sample_per_class
    client_pi_samples = {}
    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
        # 将 pi_sample_per_class 转换为 CPU 并脱离计算图
        pi_cpu = local_model.pi_sample_per_class.cpu().detach().clone()
        client_pi_samples[idx] = pi_cpu.numpy()  # 转换为 numpy 数组便于保存
    
    # 保存元数据到 pickle 文件
    metadata_dict = {
        'client_pi_sample_per_class': client_pi_samples,  # 每个客户端的平滑类别分布
        'classes_list': classes_list,  # 客户端 ID 与其拥有类别的映射表
        'num_users': args.num_users,
        'num_classes': args.num_classes,
        'beta_pi': getattr(args, 'beta_pi', 0.5)
    }
    
    metadata_path = os.path.join(logdir, 'data_distribution_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_dict, f)
    
    for round in tqdm(range(start_round, args.rounds)):
        local_weights, local_losses, local_high_protos, local_low_protos = [], [], {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train = []
        loss_list_train = []
        loss_ace_list = []  # ABBL: Loss_ACE (L_ACE)
        loss_scl_list = []  # ABBL: Loss_SCL (L_A-SCL)
        loss_proto_high_list = []  # FedMPS: Loss_proto_high (L_proto_high)
        loss_proto_low_list = []   # FedMPS: Loss_proto_low (L_proto_low)
        loss_soft_list = []        # FedMPS: Loss_soft (L_soft)
        
        # ABBL: 计算当前轮的 scl_weight（用于日志记录）
        scl_weight_start = getattr(args, 'scl_weight_start', 1.0)
        scl_weight_end = getattr(args, 'scl_weight_end', 0.0)
        if args.rounds > 0:
            scl_weight = 0.5 * (scl_weight_start - scl_weight_end) * (1 + math.cos(math.pi * round / args.rounds)) + scl_weight_end
        else:
            scl_weight = scl_weight_start
        for idx in idxs_users:
            # local model updating
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            # ABBL: 传递 total_rounds 参数用于计算余弦退火权重
            # 注意: update_weights_fedmps 返回的是按标签分类的特征字典，需要聚合后才成为原型
            w, loss, acc, high_features_by_label, low_features_by_label, idx_acc = local_model.update_weights_fedmps(
                args, idx, global_high_protos, global_low_protos, global_logits, 
                model=copy.deepcopy(local_model_list[idx]), global_round=round,
                total_rounds=args.rounds,  # ABBL: 传递总轮数用于余弦退火
                rf_models=rf_models, global_stats=global_stats
            )
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss['total'])
            # ABBL: 记录所有损失分量
            loss_ace_list.append(loss['ace'])
            loss_scl_list.append(loss['scl'])
            loss_proto_high_list.append(loss['proto_high'])
            loss_proto_low_list.append(loss['proto_low'])
            loss_soft_list.append(loss['soft'])
            # 将按标签分类的特征聚合为原型（每个类别的特征均值）
            agg_high_protos = agg_func(high_features_by_label)
            agg_low_protos = agg_func(low_features_by_label)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_high_protos[idx] = agg_high_protos
            local_low_protos[idx] = agg_low_protos

        # aggregate local multi-level prototypes instead of local weights
        local_weights_list = local_weights
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        global_high_protos = proto_aggregation(local_high_protos)
        global_low_protos = proto_aggregation(local_low_protos)

        # ========== SFD Statistics Aggregation Stage ==========
        # 检查是否需要计算全局统计量
        enable_stats_agg = getattr(args, 'enable_stats_agg', 0) == 1
        enable_safs = getattr(args, 'enable_safs', 0) == 1
        
        # 如果启用了 SAFS，必须启用统计量计算（SAFS 依赖全局统计量）
        if enable_safs and not enable_stats_agg:
            print(f'Warning: SAFS is enabled but statistics aggregation is disabled. Automatically enabling statistics aggregation.')
            logger.warning('SAFS is enabled but statistics aggregation is disabled. Automatically enabling statistics aggregation.')
            enable_stats_agg = True
        
        if enable_stats_agg:
            # Collect local statistics from all clients
            # 获取统计量计算层级（从 args 中获取，默认为 'high'）
            stats_level = getattr(args, 'stats_level', 'high')
            
            client_responses = []
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                local_stats = local_model.get_local_statistics(
                    model=copy.deepcopy(local_model_list[idx]),
                    rf_models=rf_models,
                    args=args,
                    stats_level=stats_level
                )
                client_responses.append(local_stats)
            
            # Aggregate global statistics
            global_stats = aggregate_global_statistics(
                client_responses=client_responses,
                class_num=args.num_classes,
                stats_level=stats_level
            )
        else:
            # 如果未启用统计量计算，设置为 None
            global_stats = None
            stats_level = None
        
        # ========== SFD SAFS Feature Synthesis Stage ==========
        # 如果启用了 SAFS，执行特征合成
        if getattr(args, 'enable_safs', 0) == 1:
            # SAFS 需要全局统计量，如果未启用统计量计算，应该已经在上面自动启用了
            if global_stats is None:
                raise ValueError('SAFS requires global statistics aggregation. Please enable --enable_stats_agg or it will be automatically enabled when SAFS is enabled.')
            
            # 确定使用的层级（与统计量聚合层级一致）
            level_to_use = stats_level if stats_level in ['high', 'low'] else 'high'
            
            # 获取对应层级的全局统计量和 RFF 模型
            if level_to_use == 'high':
                global_stats_level = global_stats['high']
                rf_model = rf_models['high']
                feature_dim = high_feature_dim
            else:  # level_to_use == 'low'
                global_stats_level = global_stats['low']
                rf_model = rf_models['low']
                feature_dim = low_feature_dim
            
            # 提取全局统计量
            class_means = global_stats_level['class_means']
            class_covs = global_stats_level['class_covs']
            class_rf_means = global_stats_level['class_rf_means']
            sample_per_class = global_stats['sample_per_class']
            
            # 计算每个类别的合成特征数量
            syn_nums = make_syn_nums(
                class_sizes=sample_per_class.tolist(),
                max_num=getattr(args, 'safs_max_syn_num', 2000),
                min_num=getattr(args, 'safs_min_syn_num', 600)
            )
            
            # 验证合成特征数量是否足够（必须大于特征维度）
            assert min(syn_nums) > feature_dim, \
                f'最小合成特征数量 {min(syn_nums)} 必须大于特征维度 {feature_dim}'
            
            # 为每个类别创建 MeanCov Aligner
            aligners = []
            for c in range(args.num_classes):
                aligner = MeanCovAligner(
                    target_mean=class_means[c],
                    target_cov=class_covs[c],
                    target_cov_eps=getattr(args, 'safs_target_cov_eps', 1e-5)
                )
                aligners.append(aligner)
            
            # 执行特征合成
            class_syn_datasets = feature_synthesis(
                feature_dim=feature_dim,
                class_num=args.num_classes,
                device=args.device,
                aligners=aligners,
                rf_model=rf_model,
                class_rf_means=class_rf_means,
                steps=getattr(args, 'safs_steps', 1000),
                lr=getattr(args, 'safs_lr', 0.1),
                syn_num_per_class=syn_nums,
                input_cov_eps=getattr(args, 'safs_input_cov_eps', 1e-5),
            )
        else:
            class_syn_datasets = None
        
        # ========== Global Model Training / Fine-tuning ==========
        # 根据是否启用 SAFS 选择不同的全局模型训练方式
        if getattr(args, 'enable_safs', 0) == 1 and class_syn_datasets is not None and len(class_syn_datasets) > 0:
            # 使用 SAFS 合成特征微调全局模型
            global_logits = fine_tune_global_model_safs(
                args,
                global_model,
                class_syn_datasets,
                global_high_protos,  # 注意：FedMPS主要使用全局高层原型进行分类层训练
                summary_writer=summary_writer,
                logger=logger,
                round=round
            )
        else:
            # 使用原来的方法：基于本地原型训练全局模型
            # create inputs: local high-level prototypes
            global_data, global_label = get_global_input(local_high_protos)
            dataset = TensorDataset(global_data, global_label)
            train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            # begin training and output global logits
            global_logits = train_global_proto_model(global_model, train_dataloader)

        # ABBL: 记录训练损失（包括所有损失分量）以及权重退火
        print('| ROUND: {} | Train Loss - Total: {:.5f}, L_ACE: {:.5f}, L_A-SCL: {:.5f}, L_proto_high: {:.5f}, L_proto_low: {:.5f}, L_soft: {:.5f}, SCL_Weight: {:.5f}'.format(
            round, np.mean(loss_list_train), np.mean(loss_ace_list), np.mean(loss_scl_list), 
            np.mean(loss_proto_high_list), np.mean(loss_proto_low_list), np.mean(loss_soft_list), scl_weight))
        logger.info('| ROUND: {} | Train Loss: Total={:.5f}, ACE={:.5f}, SCL={:.5f}, Proto_H={:.5f}, Proto_L={:.5f}, Soft={:.5f}, SCL_W={:.5f}'.format(
            round, np.mean(loss_list_train), np.mean(loss_ace_list), np.mean(loss_scl_list),
            np.mean(loss_proto_high_list), np.mean(loss_proto_low_list), np.mean(loss_soft_list), scl_weight))
        summary_writer.add_scalar('scalar/Train_Total_Loss', np.mean(loss_list_train), round)
        summary_writer.add_scalar('scalar/Train_Loss_ACE', np.mean(loss_ace_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_SCL', np.mean(loss_scl_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_Proto_High', np.mean(loss_proto_high_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_Proto_Low', np.mean(loss_proto_low_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_Soft', np.mean(loss_soft_list), round)
        summary_writer.add_scalar('scalar/SCL_Weight', scl_weight, round)  # ABBL: 记录权重退火变化

        # test
        acc_list_l, loss_list_l, acc_list_g, loss_list, loss_total_list = test_inference_new_het_lt(args,local_model_list,test_dataset,classes_list,user_groups_lt,global_high_protos)

        # 记录每个客户端的准确率（细粒度性能记录，可选）
        if getattr(args, 'log_client_acc', 0) == 1:
            for idx in range(args.num_users):
                summary_writer.add_scalar(f'scalar/Client_{idx}_Test_Acc_wo_Protos', acc_list_l[idx], round)
                if idx < len(acc_list_g):
                    summary_writer.add_scalar(f'scalar/Client_{idx}_Test_Acc_w_Protos', acc_list_g[idx], round)

        # 计算并记录标准差（用于评估公平性）
        std_acc_wo = np.std(acc_list_l)
        std_acc_w = np.std(acc_list_g) if len(acc_list_g) > 0 else 0.0
        summary_writer.add_scalar('scalar/Total_Test_Std_Accuracy_wo_Protos', std_acc_wo, round)
        summary_writer.add_scalar('scalar/Total_Test_Std_Accuracy_w_Protos', std_acc_w, round)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), std_acc_wo))
        logger.info('| ROUND: {} | Test Acc (w/o protos): {:.5f}±{:.5f}, Test Acc (w/ protos): {:.5f}±{:.5f}'.format(
            round, np.mean(acc_list_l), std_acc_wo, np.mean(acc_list_g), std_acc_w))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy_wp', np.mean(acc_list_g), round)

        # ========== 原型稳定性分析 (Prototype Data) ==========
        # 每隔 10 个 Round,保存当前的 global_high_protos 和 global_low_protos
        if (round + 1) % 10 == 0:
            # 将原型转换为 CPU 并脱离计算图
            proto_data = {
                'round': round + 1,
                'global_high_protos': {},
                'global_low_protos': {}
            }
            
            # 处理 global_high_protos
            # 注意: proto_aggregation 返回的格式是 {class_idx: [proto_tensor]}, 列表只包含一个张量
            for class_idx, proto_list in global_high_protos.items():
                if isinstance(proto_list, list) and len(proto_list) > 0:
                    # 列表通常只包含一个张量,直接取第一个元素
                    proto_tensor = proto_list[0] if isinstance(proto_list[0], torch.Tensor) else torch.tensor(proto_list[0])
                    proto_data['global_high_protos'][class_idx] = proto_tensor.cpu().detach().numpy()
                elif isinstance(proto_list, torch.Tensor):
                    proto_data['global_high_protos'][class_idx] = proto_list.cpu().detach().numpy()
            
            # 处理 global_low_protos
            for class_idx, proto_list in global_low_protos.items():
                if isinstance(proto_list, list) and len(proto_list) > 0:
                    proto_tensor = proto_list[0] if isinstance(proto_list[0], torch.Tensor) else torch.tensor(proto_list[0])
                    proto_data['global_low_protos'][class_idx] = proto_tensor.cpu().detach().numpy()
                elif isinstance(proto_list, torch.Tensor):
                    proto_data['global_low_protos'][class_idx] = proto_list.cpu().detach().numpy()
            
            # 保存到 pickle 文件
            proto_save_path = os.path.join(logdir, f'prototypes_round_{round+1}.pkl')
            with open(proto_save_path, 'wb') as f:
                pickle.dump(proto_data, f)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
        if np.mean(acc_list_g) > best_acc_w:
            best_acc_w = np.mean(acc_list_g)
            best_std_w = np.std(acc_list_g)
            best_round_w = round

        # ========== Stage-1 Checkpointing (latest overwrite + best-wo/wp no-overwrite) ==========
        ckpt_dir = getattr(args, 'stage1_ckpt_dir', None)
        if ckpt_dir is None:
            ckpt_dir = os.path.join(logdir, 'stage1_ckpts')
        mkdirs(ckpt_dir)

        mean_wo = float(np.mean(acc_list_l)) if len(acc_list_l) > 0 else 0.0
        mean_wp = float(np.mean(acc_list_g)) if len(acc_list_g) > 0 else 0.0

        # Always keep a latest checkpoint for resume (overwrite), optionally every N rounds.
        # Also keep non-overwriting historical snapshots (latest_rXXXX.pt) independently of latest_ckpt_interval.
        payload_latest = None
        do_save_latest = False
        if getattr(args, 'save_latest_ckpt', 1) == 1:
            interval = int(getattr(args, 'latest_ckpt_interval', 1))
            if interval <= 0:
                interval = 1
            do_save_latest = ((round + 1) % interval == 0)

        do_save_hist = False
        if int(getattr(args, 'save_latest_history', 1)) == 1:
            h_interval = int(getattr(args, 'latest_history_interval', 25))
            if h_interval <= 0:
                h_interval = 25
            do_save_hist = ((round + 1) % h_interval == 0)

        if do_save_latest or do_save_hist:
            # Export component state_dicts if requested
            comp_sd = None
            if getattr(args, 'save_stage1_components', 1) == 1:
                comp_sd = {}
                for cid, m in enumerate(local_model_list):
                    comp_sd[cid] = export_component_state_dicts(m)

            payload_latest = {
                'meta': {
                    'stage': 1,
                    'metric_type': 'latest',
                    'round': round,
                    'round_1based': round + 1,
                    'mean_acc_wo': mean_wo,
                    'mean_acc_wp': mean_wp,
                    'dataset': args.dataset,
                    'alg': args.alg,
                    'num_users': args.num_users,
                    'num_classes': args.num_classes,
                    'logdir': logdir,
                    'split_path': getattr(args, 'split_path', None),
                },
                'args': vars(args),
                'rng_state': get_rng_state(),
                'state': {
                    'local_models_full_state_dicts': {cid: m.state_dict() for cid, m in enumerate(local_model_list)},
                    'components_state_dicts': comp_sd,
                    # Global classifier head trained on (synthetic) prototypes to produce global_logits
                    'global_model_state_dict': global_model.state_dict() if global_model is not None else None,
                    'global_high_protos': global_high_protos,
                    'global_low_protos': global_low_protos,
                    'global_logits': global_logits,
                    'global_stats': global_stats,
                    'rf_models_state': {
                        'high': rf_models['high'].state_dict() if rf_models and 'high' in rf_models else None,
                        'low': rf_models['low'].state_dict() if rf_models and 'low' in rf_models else None,
                    },
                    'best': {
                        'best_acc': best_acc,
                        'best_std': best_std,
                        'best_round': best_round,
                        'best_acc_w': best_acc_w,
                        'best_std_w': best_std_w,
                        'best_round_w': best_round_w,
                    },
                },
            }

            if do_save_latest:
                save_latest(ckpt_dir=ckpt_dir, payload=payload_latest, filename='latest.pt')

            if do_save_hist:
                snap_name = f"latest_r{round + 1:04d}.pt"
                try:
                    save_latest(ckpt_dir=ckpt_dir, payload=payload_latest, filename=snap_name)
                except Exception as e:
                    # Do not crash training if snapshot saving fails (disk full, permission, etc.)
                    print(f"Warning: failed to save snapshot checkpoint {snap_name} ({e})")

        # Save best checkpoints without overwrite
        if getattr(args, 'save_best_ckpt', 1) == 1:
            best_overwrite = int(getattr(args, 'best_ckpt_overwrite', 1)) == 1
            # best-wo trigger
            if best_round == round:
                comp_sd = None
                if getattr(args, 'save_stage1_components', 1) == 1:
                    comp_sd = {}
                    for cid, m in enumerate(local_model_list):
                        comp_sd[cid] = export_component_state_dicts(m)
                payload = {
                    'meta': {
                        'stage': 1,
                        'metric_type': 'best-wo',
                        'round': round,
                        'round_1based': round + 1,
                        'mean_acc_wo': mean_wo,
                        'mean_acc_wp': mean_wp,
                        'dataset': args.dataset,
                        'alg': args.alg,
                        'num_users': args.num_users,
                        'num_classes': args.num_classes,
                        'logdir': logdir,
                        'split_path': getattr(args, 'split_path', None),
                    },
                    'args': vars(args),
                    'rng_state': get_rng_state(),
                    'state': {
                        'local_models_full_state_dicts': {cid: m.state_dict() for cid, m in enumerate(local_model_list)},
                        'components_state_dicts': comp_sd,
                        'global_model_state_dict': global_model.state_dict() if global_model is not None else None,
                        'global_high_protos': global_high_protos,
                        'global_low_protos': global_low_protos,
                        'global_logits': global_logits,
                        'global_stats': global_stats,
                        'rf_models_state': {
                            'high': rf_models['high'].state_dict() if rf_models and 'high' in rf_models else None,
                            'low': rf_models['low'].state_dict() if rf_models and 'low' in rf_models else None,
                        },
                        'best': {
                            'best_acc': best_acc,
                            'best_std': best_std,
                            'best_round': best_round,
                            'best_acc_w': best_acc_w,
                            'best_std_w': best_std_w,
                            'best_round_w': best_round_w,
                        },
                    },
                }
                save_best(
                    ckpt_dir=ckpt_dir,
                    metric_type='best-wo',
                    round_idx=round + 1,
                    mean_acc_wo=mean_wo,
                    mean_acc_wp=mean_wp,
                    payload=payload,
                    overwrite=best_overwrite,
                )

            # best-wp trigger
            if best_round_w == round:
                comp_sd = None
                if getattr(args, 'save_stage1_components', 1) == 1:
                    comp_sd = {}
                    for cid, m in enumerate(local_model_list):
                        comp_sd[cid] = export_component_state_dicts(m)
                payload = {
                    'meta': {
                        'stage': 1,
                        'metric_type': 'best-wp',
                        'round': round,
                        'round_1based': round + 1,
                        'mean_acc_wo': mean_wo,
                        'mean_acc_wp': mean_wp,
                        'dataset': args.dataset,
                        'alg': args.alg,
                        'num_users': args.num_users,
                        'num_classes': args.num_classes,
                        'logdir': logdir,
                        'split_path': getattr(args, 'split_path', None),
                    },
                    'args': vars(args),
                    'rng_state': get_rng_state(),
                    'state': {
                        'local_models_full_state_dicts': {cid: m.state_dict() for cid, m in enumerate(local_model_list)},
                        'components_state_dicts': comp_sd,
                        'global_model_state_dict': global_model.state_dict() if global_model is not None else None,
                        'global_high_protos': global_high_protos,
                        'global_low_protos': global_low_protos,
                        'global_logits': global_logits,
                        'global_stats': global_stats,
                        'rf_models_state': {
                            'high': rf_models['high'].state_dict() if rf_models and 'high' in rf_models else None,
                            'low': rf_models['low'].state_dict() if rf_models and 'low' in rf_models else None,
                        },
                        'best': {
                            'best_acc': best_acc,
                            'best_std': best_std,
                            'best_round': best_round,
                            'best_acc_w': best_acc_w,
                            'best_std_w': best_std_w,
                            'best_round_w': best_round_w,
                        },
                    },
                }
                save_best(
                    ckpt_dir=ckpt_dir,
                    metric_type='best-wp',
                    round_idx=round + 1,
                    mean_acc_wo=mean_wo,
                    mean_acc_wp=mean_wp,
                    payload=payload,
                    overwrite=best_overwrite,
                )

    print('| BEST ROUND (w/o protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    logger.info('| BEST ROUND (w/o protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round, best_acc, best_std))
    print('| BEST ROUND (w/ protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round_w, best_acc_w, best_std_w))
    logger.info('| BEST ROUND (w/ protos): {} | Test Acc: {:.5f}±{:.5f}'.format(best_round_w, best_acc_w, best_std_w))
    
    # Save final SFD statistics (from the last round)
    # Use the same logdir pattern as the main function
    save_dir = os.path.join('../newresults', args.alg, str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S"))+'_'+args.dataset+'_n'+str(args.ways)+'_sfd_stats')
    mkdirs(save_dir)
    
    # Save final global statistics from the last round
    if global_stats is not None:
        stats_save_path = os.path.join(save_dir, 'global_stats_final.pkl')
        with open(stats_save_path, 'wb') as f:
            pickle.dump(global_stats, f)
    
    # Save RFF models state dict
    rf_models_save_path = os.path.join(save_dir, 'rf_models.pkl')
    rf_models_state = {
        'high': rf_model_high.state_dict(),
        'low': rf_model_low.state_dict()
    }
    with open(rf_models_save_path, 'wb') as f:
        pickle.dump(rf_models_state, f)
    
    # Save metadata
    metadata = {
        'high_feature_dim': high_feature_dim,
        'low_feature_dim': low_feature_dim,
        'rf_dim_high': args.rf_dim_high,
        'rf_dim_low': args.rf_dim_low,
        'rbf_gamma_high': args.rbf_gamma_high,
        'rbf_gamma_low': args.rbf_gamma_low,
        'rf_type': args.rf_type,
        'rf_seed': args.rf_seed,
        'num_classes': args.num_classes,
        'num_clients': len(idxs_users)
    }
    metadata_save_path = os.path.join(save_dir, 'metadata.pkl')
    with open(metadata_save_path, 'wb') as f:
        pickle.dump(metadata, f)



if __name__ == '__main__':
    args = args_parser()

    import secrets
    # 如果种子为默认值，自动生成随机种子
    if args.seed == 1234:  # 默认值
        args.seed = secrets.randbelow(2**31)
    if args.rf_seed == 42:  # 默认值
        args.rf_seed = secrets.randbelow(2**31)
        
    exp_details(args)

    # ===================== Stage-2: Statistics Aggregation (low-only) =====================
    # Stage-2 loads a Stage-1 checkpoint (typically best-wo.pt) and computes global statistics once.
    stage = int(getattr(args, 'stage', 1))
    stage2_payload = None
    if stage == 2:
        if getattr(args, 'stage1_ckpt_path', None) is None:
            raise ValueError('Stage-2 requires --stage1_ckpt_path (e.g., <log_dir>/stage1_ckpts/best-wo.pt).')
        try:
            stage2_payload = load_checkpoint(args.stage1_ckpt_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f'Failed to load Stage-1 checkpoint from --stage1_ckpt_path: {args.stage1_ckpt_path} ({e})')

        if isinstance(stage2_payload, dict):
            ckpt_meta = stage2_payload.get('meta', {}) or {}
            ckpt_args = stage2_payload.get('args', {}) or {}
        else:
            raise ValueError(f'Invalid checkpoint payload type: {type(stage2_payload)}')

        # Force key dataset/model-shape arguments to match Stage-1 checkpoint for correct model reconstruction.
        for k in [
            'dataset', 'alg', 'num_users', 'num_classes',
            'ways', 'shots', 'train_shots_max', 'test_shots', 'stdev',
            'mode', 'model', 'num_channels',
        ]:
            if k in ckpt_args:
                try:
                    setattr(args, k, ckpt_args[k])
                except Exception:
                    pass

        # Infer log_dir from checkpoint meta if not provided
        if args.log_dir is None:
            meta_logdir = ckpt_meta.get('logdir', None)
            if meta_logdir:
                args.log_dir = meta_logdir

        # Force split_path reuse (Stage-2 should be consistent with Stage-1 split)
        if getattr(args, 'split_path', None) is None:
            meta_split = ckpt_meta.get('split_path', None)
            if meta_split:
                args.split_path = meta_split

    # If resuming and user didn't provide log_dir, infer it from resume_ckpt_path
    if getattr(args, 'resume_ckpt_path', None) and args.log_dir is None:
        ckpt_abs = os.path.abspath(args.resume_ckpt_path)
        ckpt_dir = os.path.dirname(ckpt_abs)
        # typical: <logdir>/stage1_ckpts/latest.pt
        if os.path.basename(ckpt_dir) == 'stage1_ckpts':
            args.log_dir = os.path.dirname(ckpt_dir)
        else:
            args.log_dir = ckpt_dir

    # 如果用户指定了自定义 logdir,使用它;否则自动生成
    if args.log_dir is not None:
        logdir = args.log_dir
    else:
        logdir = os.path.join('../newresults', args.alg, str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S"))+'_'+args.dataset+'_n'+str(args.ways))
    mkdirs(logdir)

    # ===================== Safety guard: prevent accidental overwrite =====================
    # If the user points to an existing logdir that already has a latest checkpoint, we refuse to
    # start from scratch unless allow_restart=1. This prevents "latest.pt got overwritten by a new run".
    ckpt_dir_default = os.path.join(logdir, 'stage1_ckpts')
    latest_default = os.path.join(ckpt_dir_default, 'latest.pt')
    # Only Stage-1 can overwrite latest.pt; Stage-2/3 are offline utilities.
    if stage == 1 and getattr(args, 'resume_ckpt_path', None) is None:
        if os.path.exists(latest_default) and int(getattr(args, 'allow_restart', 0)) != 1:
            raise RuntimeError(
                "检测到该 log_dir 下已存在 stage1_ckpts/latest.pt，但你没有指定 --resume_ckpt_path。\n"
                "为避免误覆盖断点文件，程序已停止。\n"
                "解决方案：\n"
                "  1) 断点续训：加上 --resume_ckpt_path \"<log_dir>\\stage1_ckpts\\latest.pt\"\n"
                "  2) 确认要从头重跑并覆盖 latest.pt：加上 --allow_restart 1\n"
            )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logdir, 'log.log'),
        format='[%(levelname)s](%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d/ %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    print("**Basic Setting...")
    print('  ', args)
    logging.info("="*60)
    logging.info("Experiment Settings:")
    logging.info("="*60)
    logging.info(args)

    # Put TensorBoard events into a subdir to avoid mixing with checkpoints/other artifacts.
    tb_logdir = os.path.join(logdir, 'tb')
    mkdirs(tb_logdir)
    summary_writer = SummaryWriter(tb_logdir)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Run Stage-2 early and exit (no training loop)
    if stage == 2:
        logger.info("="*60)
        logger.info("Stage-2: Statistics Aggregation (low-only)")
        logger.info("="*60)
        logger.info(f"stage1_ckpt_path = {args.stage1_ckpt_path}")

        ckpt_meta = stage2_payload.get('meta', {}) or {}
        ckpt_state = stage2_payload.get('state', {}) or {}

        # Resolve output dir
        stage2_out_dir = getattr(args, 'stage2_out_dir', None)
        if stage2_out_dir is None:
            stage2_out_dir = os.path.join(logdir, 'stage2_stats')
        mkdirs(stage2_out_dir)

        # Stage-2 requires a persisted split for consistency
        if getattr(args, 'split_path', None) is None:
            raise ValueError('Stage-2 requires split_path (can be inferred from checkpoint meta).')
        if not os.path.exists(args.split_path):
            raise FileNotFoundError(f'Stage-2 requires an existing split file, but not found: {args.split_path}')

        loaded_split = load_split(args.split_path)
        n_list = loaded_split['n_list']
        k_list = loaded_split['k_list']
        train_dataset, test_dataset, _, _, _, _ = get_dataset(args, n_list, k_list)
        user_groups = loaded_split['user_groups']
        user_groups_lt = loaded_split['user_groups_lt']
        classes_list = loaded_split['classes_list']
        classes_list_gt = loaded_split.get('classes_list_gt', classes_list)

        # Rebuild client models and load per-client weights from Stage-1 checkpoint
        local_model_list = []
        for i in range(args.num_users):
            if args.dataset == 'mnist':
                if args.mode == 'model_heter':
                    if i < 7:
                        args.out_channels = 18
                    elif i >= 7 and i < 14:
                        args.out_channels = 20
                    else:
                        args.out_channels = 22
                else:
                    args.out_channels = 20
                local_model = CNNMnist(args=args)
            elif args.dataset == 'femnist':
                if args.mode == 'model_heter':
                    if i < 7:
                        args.out_channels = 18
                    elif i >= 7 and i < 14:
                        args.out_channels = 20
                    else:
                        args.out_channels = 22
                else:
                    args.out_channels = 20
                local_model = CNNFemnist(args=args)
            elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'flowers' or args.dataset == 'defungi':
                local_model = CNNCifar(args=args)
            elif args.dataset == 'tinyimagenet':
                args.num_classes = 200
                local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
            elif args.dataset == 'realwaste':
                local_model = CNNCifar(args=args)
            elif args.dataset == 'fashion':
                local_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'imagenet':
                local_model = ResNetWithFeatures(base='resnet18', num_classes=args.num_classes)
            else:
                raise ValueError(f'Unsupported dataset for Stage-2: {args.dataset}')

            local_model.to(args.device)
            local_model.eval()
            local_model_list.append(local_model)

        local_sd = ckpt_state.get('local_models_full_state_dicts', None)
        if not isinstance(local_sd, dict):
            raise ValueError("Checkpoint missing state['local_models_full_state_dicts'] for Stage-2.")
        for cid, m in enumerate(local_model_list):
            key = cid
            if key not in local_sd and str(cid) in local_sd:
                key = str(cid)
            if key in local_sd:
                m.load_state_dict(local_sd[key], strict=True)
            else:
                raise KeyError(f'Checkpoint local_models_full_state_dicts missing client id={cid}')

        # Build low-level RFF model (d inferred from a dummy forward)
        dummy_model = copy.deepcopy(local_model_list[0]).to(args.device).eval()
        if args.dataset == 'mnist' or args.dataset == 'femnist' or args.dataset == 'fashion':
            dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
        elif args.dataset == 'tinyimagenet':
            dummy_input = torch.randn(1, 3, 64, 64).to(args.device)
        elif args.dataset == 'imagenet':
            dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
        else:
            dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
        with torch.no_grad():
            dummy_output = dummy_model(dummy_input)
            if isinstance(dummy_output, tuple) and len(dummy_output) >= 4:
                low_feature_dim = dummy_output[3].shape[1]
            else:
                raise ValueError('Dummy model output format unexpected for Stage-2.')

        backup_rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        }
        random.seed(args.rf_seed)
        np.random.seed(args.rf_seed)
        torch.manual_seed(args.rf_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.rf_seed)

        rf_model_low = RFF(
            d=low_feature_dim,
            D=args.rf_dim_low,
            gamma=args.rbf_gamma_low,
            device=args.device,
            rf_type=args.rf_type
        )
        rf_models = {'low': rf_model_low}

        random.setstate(backup_rng_state['python'])
        np.random.set_state(backup_rng_state['numpy'])
        torch.set_rng_state(backup_rng_state['torch'])

        # Collect local statistics (low-only) and aggregate
        client_responses = []
        for idx in range(args.num_users):
            local_upd = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            if hasattr(local_upd, 'get_local_statistics_streaming'):
                local_stats = local_upd.get_local_statistics_streaming(
                    model=copy.deepcopy(local_model_list[idx]),
                    rf_models=rf_models,
                    args=args,
                    stats_level='low'
                )
            else:
                local_stats = local_upd.get_local_statistics(
                    model=copy.deepcopy(local_model_list[idx]),
                    rf_models=rf_models,
                    args=args,
                    stats_level='low'
                )
            client_responses.append(local_stats)

        global_stats = aggregate_global_statistics(
            client_responses=client_responses,
            class_num=args.num_classes,
            stats_level='low'
        )

        stage2_meta = {
            'stage': 2,
            'metric_type': 'stats-agg',
            'dataset': args.dataset,
            'alg': args.alg,
            'num_users': args.num_users,
            'num_classes': args.num_classes,
            'logdir': logdir,
            'split_path': getattr(args, 'split_path', None),
            'stage1_ckpt_path': args.stage1_ckpt_path,
            'stage1_meta': ckpt_meta,
            'low_feature_dim': low_feature_dim,
            'rf_dim_low': args.rf_dim_low,
            'rbf_gamma_low': args.rbf_gamma_low,
            'rf_seed': args.rf_seed,
            'rf_type': args.rf_type,
        }
        out_payload = {
            'meta': stage2_meta,
            'args': vars(args),
            'state': {
                'global_stats': global_stats,
                'rf_models_state': {'low': rf_model_low.state_dict()},
            },
        }

        # Persist
        torch.save(out_payload, os.path.join(stage2_out_dir, 'global_stats.pt'))
        with open(os.path.join(stage2_out_dir, 'global_stats.pkl'), 'wb') as f:
            pickle.dump(out_payload, f)

        print(f"[Stage-2] Saved global statistics to: {stage2_out_dir}")
        raise SystemExit(0)

    # Run Stage-3 early and exit (train generator)
    if stage == 3:
        logger.info("="*60)
        logger.info("Stage-3: Train stats-conditioned low-level feature generator")
        logger.info("="*60)

        stage2_stats_path = getattr(args, "stage2_stats_path", None)
        if stage2_stats_path is None:
            stage2_stats_path = os.path.join(logdir, "stage2_stats", "global_stats.pt")
        if not os.path.exists(stage2_stats_path):
            raise FileNotFoundError(f"[Stage-3] stage2_stats_path not found: {stage2_stats_path}")

        payload = torch.load(stage2_stats_path, map_location="cpu")
        meta = payload.get("meta", {}) or {}
        state = payload.get("state", {}) or {}
        global_stats = state.get("global_stats", None)
        if global_stats is None:
            raise KeyError("[Stage-3] payload['state']['global_stats'] missing.")

        # Resolve dims and RFF model
        num_classes = int(meta.get("num_classes", getattr(args, "num_classes", 10)))
        low_feature_dim = int(meta.get("low_feature_dim", None) or stack_low_global_stats(global_stats).mu.shape[1])
        rf_dim_low = int(meta.get("rf_dim_low", getattr(args, "rf_dim_low", 3000)))
        rbf_gamma_low = float(meta.get("rbf_gamma_low", getattr(args, "rbf_gamma_low", 0.01)))
        rf_type = str(meta.get("rf_type", getattr(args, "rf_type", "orf")))

        rf_model_low = RFF(d=low_feature_dim, D=rf_dim_low, gamma=rbf_gamma_low, device=args.device, rf_type=rf_type)
        rf_state = (state.get("rf_models_state", {}) or {}).get("low", None)
        if rf_state is not None:
            rf_model_low.load_state_dict(rf_state, strict=True)
        rf_model_low = rf_model_low.to(args.device).eval()

        # Stack class-wise stats tensors
        stats = stack_low_global_stats(global_stats)
        stats = type(stats)(
            mu=stats.mu.to(args.device),
            cov_diag=stats.cov_diag.to(args.device),
            rf_mean=stats.rf_mean.to(args.device),
            sample_per_class=stats.sample_per_class.to(args.device),
        )

        qualified = torch.nonzero(stats.sample_per_class > 0, as_tuple=False).view(-1)
        if qualified.numel() == 0:
            raise ValueError("[Stage-3] No qualified classes with sample_per_class > 0 in global_stats.")

        # Seeds (separate from Stage-1 seed)
        gen_seed = int(getattr(args, "gen_seed", getattr(args, "seed", 1234)))
        random.seed(gen_seed)
        np.random.seed(gen_seed)
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)

        gen = StatsConditionedFeatureGenerator(
            num_classes=num_classes,
            feature_dim=low_feature_dim,
            noise_dim=int(getattr(args, "gen_noise_dim", 64)),
            y_emb_dim=int(getattr(args, "gen_y_emb_dim", 32)),
            stat_emb_dim=int(getattr(args, "gen_stat_emb_dim", 128)),
            hidden_dim=int(getattr(args, "gen_hidden_dim", 256)),
            n_hidden_layers=int(getattr(args, "gen_n_hidden_layers", 2)),
            relu_output=int(getattr(args, "gen_relu_output", 1)) == 1,
            use_cov_diag=int(getattr(args, "gen_use_cov_diag", 1)) == 1,
        ).to(args.device)

        diversity_loss_fn = DiversityLoss(metric="l1").to(args.device)
        opt = torch.optim.Adam(gen.parameters(), lr=float(getattr(args, "gen_lr", 1e-3)))

        w_mean = float(getattr(args, "gen_w_mean", 1.0))
        w_var = float(getattr(args, "gen_w_var", 0.1))
        w_rff = float(getattr(args, "gen_w_rff", 1.0))
        w_div = float(getattr(args, "gen_w_div", 0.01))
        w_arr = float(getattr(args, "gen_w_arr", 0.0))

        steps = int(getattr(args, "gen_steps", 2000))
        batch_size = int(getattr(args, "gen_batch_size", 256))
        print(f"[Stage-3] Training generator: steps={steps}, batch_size={batch_size}, d_low={low_feature_dim}, rf_dim={rf_dim_low}")
        logger.info(f"[Stage-3] stage2_stats_path={stage2_stats_path}")

        for step in range(steps):
            # sample labels from qualified classes (uniform)
            idx = torch.randint(low=0, high=qualified.numel(), size=(batch_size,), device=args.device)
            y = qualified[idx].long()

            mu_b, cov_diag_b, rf_b = gather_by_label(stats, y)
            gen_res = gen(y, mu=mu_b, cov_diag=cov_diag_b, verbose=True)
            x = gen_res["output"]
            eps = gen_res["eps"]

            uniq = torch.unique(y)
            mean_loss = torch.tensor(0.0, device=args.device)
            var_loss = torch.tensor(0.0, device=args.device)
            rff_loss = torch.tensor(0.0, device=args.device)
            n_used = 0
            for c in uniq.tolist():
                mask = (y == c)
                if not torch.any(mask):
                    continue
                xs = x[mask]
                n_used += 1

                target_mu = stats.mu[c]
                target_var = stats.cov_diag[c]
                target_rf = stats.rf_mean[c]

                mean_loss = mean_loss + F.mse_loss(xs.mean(dim=0), target_mu, reduction="mean")

                if xs.shape[0] >= 2:
                    v = xs.var(dim=0, unbiased=False)
                    var_loss = var_loss + F.l1_loss(v, target_var, reduction="mean")

                rf_mean_syn = rf_model_low(xs).mean(dim=0)
                rff_loss = rff_loss + F.l1_loss(rf_mean_syn, target_rf, reduction="mean")

            if n_used > 0:
                mean_loss = mean_loss / n_used
                var_loss = var_loss / n_used
                rff_loss = rff_loss / n_used

            div_loss = diversity_loss_fn(eps, x)

            # ARR (non-negativity) only meaningful if output isn't ReLU-ed
            if w_arr != 0.0:
                arr_loss = (-torch.minimum(x, torch.zeros_like(x))).sum(dim=1).mean()
            else:
                arr_loss = torch.tensor(0.0, device=args.device)

            loss = w_mean * mean_loss + w_var * var_loss + w_rff * rff_loss + w_div * div_loss + w_arr * arr_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 100 == 0 or step == steps - 1:
                msg = (
                    f"[Stage-3][{step:05d}/{steps}] "
                    f"loss={loss.item():.6f} "
                    f"mean={mean_loss.item():.6f} var={var_loss.item():.6f} "
                    f"rff={rff_loss.item():.6f} div={div_loss.item():.6f} arr={arr_loss.item():.6f}"
                )
                print(msg)
                logger.info(msg)

        # Save artifacts
        out_dir = getattr(args, "stage3_out_dir", None)
        if out_dir is None:
            out_dir = os.path.join(logdir, "stage3_gen")
        mkdirs(out_dir)

        gen_path = os.path.join(out_dir, "generator.pt")
        torch.save(gen.state_dict(), gen_path)

        meta_out = {
            "stage": 3,
            "dataset": meta.get("dataset", getattr(args, "dataset", None)),
            "alg": meta.get("alg", getattr(args, "alg", None)),
            "num_classes": num_classes,
            "low_feature_dim": low_feature_dim,
            "rf_dim_low": rf_dim_low,
            "rbf_gamma_low": rbf_gamma_low,
            "rf_type": rf_type,
            "stage2_stats_path": stage2_stats_path,
            "gen_seed": gen_seed,
            "gen_hparams": {
                "gen_noise_dim": int(getattr(args, "gen_noise_dim", 64)),
                "gen_y_emb_dim": int(getattr(args, "gen_y_emb_dim", 32)),
                "gen_stat_emb_dim": int(getattr(args, "gen_stat_emb_dim", 128)),
                "gen_hidden_dim": int(getattr(args, "gen_hidden_dim", 256)),
                "gen_n_hidden_layers": int(getattr(args, "gen_n_hidden_layers", 2)),
                "gen_relu_output": int(getattr(args, "gen_relu_output", 1)),
                "gen_use_cov_diag": int(getattr(args, "gen_use_cov_diag", 1)),
                "gen_steps": steps,
                "gen_batch_size": batch_size,
                "gen_lr": float(getattr(args, "gen_lr", 1e-3)),
                "weights": {
                    "gen_w_mean": w_mean,
                    "gen_w_var": w_var,
                    "gen_w_rff": w_rff,
                    "gen_w_div": w_div,
                    "gen_w_arr": w_arr,
                },
            },
        }
        with open(os.path.join(out_dir, "generator_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)

        print(f"[Stage-3] Saved generator to: {gen_path}")
        print(f"[Stage-3] Saved generator meta to: {os.path.join(out_dir, 'generator_meta.json')}")
        raise SystemExit(0)

    # ===================== Resume Stage-1 (optional) =====================
    resume_payload = None
    if getattr(args, 'resume_ckpt_path', None):
        try:
            resume_payload = load_checkpoint(args.resume_ckpt_path, map_location='cpu')
            meta = resume_payload.get('meta', {}) if isinstance(resume_payload, dict) else {}
            print(f"Loaded resume checkpoint: {args.resume_ckpt_path}")
            if isinstance(meta, dict) and len(meta) > 0:
                print(f"  ckpt.meta.round = {meta.get('round', None)} (0-based), round_1based = {meta.get('round_1based', None)}")
                print(f"  ckpt.meta.logdir = {meta.get('logdir', None)}")
        except Exception as e:
            print(f"Failed to load resume checkpoint: {args.resume_ckpt_path} ({e})")
            resume_payload = None

    # ===================== Dataset split (persist / reuse) =====================
    if getattr(args, 'split_path', None) is None:
        args.split_path = os.path.join(logdir, 'split.pkl')

    loaded_split = None
    if getattr(args, 'reuse_split', 1) == 1 and os.path.exists(args.split_path):
        loaded_split = load_split(args.split_path)

    # load dataset and user groups
    if loaded_split is None:
        n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)# Minimum 2 classes; cannot exceed the total number of classes
        if args.dataset == 'mnist':
            k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
        elif args.dataset == 'cifar10':
            k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
        elif args.dataset =='cifar100':
            k_list = np.random.randint(args.shots- args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
        elif args.dataset == 'femnist':
            k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
        elif args.dataset=='tinyimagenet':
            k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
        elif args.dataset == 'realwaste':
            k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
        elif args.dataset == 'flowers':
            k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
        elif args.dataset == 'defungi' or args.dataset == 'fashion':
            k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
        elif args.dataset == 'imagenet':
            k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)

        train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
        split_to_save = {
            'meta': {
                'dataset': args.dataset,
                'num_users': args.num_users,
                'num_classes': args.num_classes,
                'ways': args.ways,
                'shots': args.shots,
                'train_shots_max': args.train_shots_max,
                'test_shots': args.test_shots,
                'stdev': args.stdev,
                'seed': args.seed,
            },
            'n_list': n_list,
            'k_list': k_list,
            'user_groups': user_groups,
            'user_groups_lt': user_groups_lt,
            'classes_list': classes_list,
            'classes_list_gt': classes_list_gt,
        }
        save_split(args.split_path, split_to_save)
        print(f"Saved dataset split to: {args.split_path}")
    else:
        # Load datasets as usual, but override split with persisted split.
        n_list = loaded_split['n_list']
        k_list = loaded_split['k_list']
        train_dataset, test_dataset, _, _, _, _ = get_dataset(args, n_list, k_list)
        user_groups = loaded_split['user_groups']
        user_groups_lt = loaded_split['user_groups_lt']
        classes_list = loaded_split['classes_list']
        classes_list_gt = loaded_split.get('classes_list_gt', classes_list)
        print(f"Reused dataset split from: {args.split_path}")
    # user_groups: dictionary where
    #   - key = client ID
    #   - value = ndarray of selected sample IDs for the client’s chosen classes (class IDs sorted in ascending order)
    # user_groups_lt: test set sample ID dictionary
    # classes_list: list of lists representing the classes assigned to each client

    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar10' or args.dataset=='cifar100' or args.dataset == 'flowers'  or args.dataset == 'defungi' :
            local_model = CNNCifar(args=args)
        elif args.dataset=='tinyimagenet':
            args.num_classes = 200
            local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='realwaste':
            local_model = CNNCifar(args=args)
        elif args.dataset=='fashion':
            local_model=CNNFashion_Mnist(args=args)
        elif args.dataset == 'imagenet':
            local_model = ResNetWithFeatures(base='resnet18', num_classes=args.num_classes)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    # Restore model weights & training state if resuming
    if resume_payload is not None:
        try:
            st = resume_payload.get('state', {})
            local_sd = st.get('local_models_full_state_dicts', None)
            if isinstance(local_sd, dict):
                for cid, m in enumerate(local_model_list):
                    if cid in local_sd:
                        m.load_state_dict(local_sd[cid], strict=True)
                print("Restored local model weights from checkpoint.")
            # Restore RNG for more reproducible continuation
            rng_state = resume_payload.get('rng_state', None)
            if rng_state is not None:
                set_rng_state(rng_state)
                print("Restored RNG state from checkpoint.")

            # Inject resume state into args for FedMPS
            def _move_tensors_to_device(obj, device):
                """Recursively move tensors inside nested dict/list/tuple structures to the given device."""
                if torch.is_tensor(obj):
                    return obj.to(device)
                if isinstance(obj, dict):
                    return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_move_tensors_to_device(v, device) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_move_tensors_to_device(v, device) for v in obj)
                return obj

            best_state = st.get('best', {}) if isinstance(st, dict) else {}
            args.start_round = int(resume_payload.get('meta', {}).get('round', 0)) + 1
            args.resume_state = {
                # IMPORTANT: checkpoint is loaded with map_location='cpu', so move cached tensors back to args.device
                'global_high_protos': _move_tensors_to_device(st.get('global_high_protos', {}), args.device),
                'global_low_protos': _move_tensors_to_device(st.get('global_low_protos', {}), args.device),
                'global_logits': _move_tensors_to_device(st.get('global_logits', {}), args.device),
                # Global classifier head weights (GlobalFedmps) for consistent global_logits generation
                # Keep on CPU; load_state_dict will copy to the model device.
                'global_model_state_dict': st.get('global_model_state_dict', None),
                'best_acc': best_state.get('best_acc', -float('inf')),
                'best_std': best_state.get('best_std', -float('inf')),
                'best_round': best_state.get('best_round', 0),
                'best_acc_w': best_state.get('best_acc_w', -float('inf')),
                'best_std_w': best_state.get('best_std_w', -float('inf')),
                'best_round_w': best_state.get('best_round_w', 0),
            }
        except Exception as e:
            print(f"Warning: resume restoration failed, continuing without resume state. ({e})")


    if args.alg=='fedavg':
        Fedavg(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, summary_writer,logger,logdir)
    elif args.alg=='fedprox':
        Fedprox(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,logdir)
    elif args.alg=='moon':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'femnist':
            global_model = CNNFemnist(args=args)
        elif args.dataset == 'cifar10' or args.dataset=='realwaste' or args.dataset == 'flowers' or args.dataset == 'defungi':
            global_model = CNNCifar(args=args)
        elif args.dataset=='cifar100':
            args.num_classes = 100
            local_model = ModelCT( out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='tinyimagenet':
            args.num_classes = 200
            local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='fashion':
            global_model=CNNFashion_Mnist(args=args)
        elif args.dataset == 'imagenet':
            global_model = ResNetWithFeatures(base='resnet18')
        global_model.to(args.device)
        global_model.train()
        Moon(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,global_model,logger,summary_writer,logdir)
    elif args.alg == 'fedntd':
        fedntd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer, logger, logdir)
    elif args.alg == 'fedgkd':
        fedgkd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, logdir)
    elif args.alg=='fedproc':
        Fedproc(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    elif args.alg=='fedproto':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir)
    elif args.alg=='ours':#FedMPS
        FedMPS(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,summary_writer,logger,logdir)



