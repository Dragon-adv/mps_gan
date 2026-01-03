#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=8,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--log_dir', type=str, default=None, 
                        help="custom log directory path. If None, will auto-generate based on algorithm, dataset, and timestamp")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int, default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # Local arguments
    parser.add_argument('--ways', type=int, default=3, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=2, help="stdev of ways")
    parser.add_argument('--ld', type=float, default=1, help="fedproto: weight of proto loss")
    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--model_buffer_size', type=int, default=1,help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--T', type=float, default=5, help='the temperature parameter for soft label')
    parser.add_argument('--alph', type=float, default=1, help='weight of high proto contrastive loss')# Note that the weights of the various losses here differ from those in the article
    parser.add_argument('--beta', type=float, default=1, help='weight of low proto contrastive loss')
    parser.add_argument('--gama', type=float, default=0, help='weight of logits loss')

    parser.add_argument('--Dbeta', type=float, default=0.5, help="diri distribution parameter")

    # for fedgkd
    parser.add_argument("--gkd_temperature", default=1, type=float)
    parser.add_argument("--distillation_coefficient", default=0.1, type=float)
    parser.add_argument("--buffer_length", default=5, type=int)

    # for fedntd
    parser.add_argument('--ntd_tau', type=float, default=1, help='for fedntd, the temperature of distilling')
    parser.add_argument('--ntd_beta', type=float, default=1, help='for fedntd, the wight of ntdloss')

    # for SFD statistics aggregation (RFF parameters)
    parser.add_argument('--rf_seed', type=int, default=42, help='random seed for RFF model initialization')
    parser.add_argument('--rf_dim_high', type=int, default=3000, help='RFF dimension for high-level features')
    parser.add_argument('--rf_dim_low', type=int, default=3000, help='RFF dimension for low-level features')
    parser.add_argument('--rbf_gamma_high', type=float, default=0.01, help='RBF gamma parameter for high-level RFF model')
    parser.add_argument('--rbf_gamma_low', type=float, default=0.01, help='RBF gamma parameter for low-level RFF model')
    parser.add_argument('--rf_type', type=str, default='orf', help='RFF type: orf or iid')
    parser.add_argument('--stats_level', type=str, default='high', 
                        choices=['high', 'low', 'both'],
                        help='Statistics computation level: high (only high-level), low (only low-level), or both (default: high)')

    # for SFD SAFS (Synthetic Feature-based Decoupled training)
    # Note: For globally balanced datasets (e.g., CIFAR-10), synthetic feature numbers should be consistent
    parser.add_argument('--enable_safs', type=int, default=0, 
                        help='Enable SAFS feature synthesis (0: disabled, 1: enabled, default: 1)')
    parser.add_argument('--safs_steps', type=int, default=200, 
                        help='Number of optimization steps for SAFS feature synthesis')
    parser.add_argument('--safs_lr', type=float, default=0.1, 
                        help='Learning rate for SAFS feature synthesis')
    parser.add_argument('--safs_max_syn_num', type=int, default=600, 
                        help='Maximum synthetic features for the smallest class (default: 600, adjusted for globally balanced datasets)')
    parser.add_argument('--safs_min_syn_num', type=int, default=600, 
                        help='Minimum synthetic features for the largest class (default: 600, same as max for balanced datasets)')
    parser.add_argument('--safs_target_cov_eps', type=float, default=1e-5, 
                        help='Jitter for target covariance matrix in MeanCov Aligner (default: 1e-5)')
    parser.add_argument('--safs_input_cov_eps', type=float, default=1e-5, 
                        help='Jitter for input covariance matrix in MeanCov Aligner (default: 1e-5)')
    parser.add_argument('--safs_finetune_epochs', type=int, default=5, 
                        help='Number of epochs for SAFS global model fine-tuning (default: 5)')
    parser.add_argument('--safs_finetune_batch_size', type=int, default=32, 
                        help='Batch size for SAFS global model fine-tuning (default: 32)')

    # for ABBL (Adaptive Bi-Branch Learning) - SFD loss functions
    # Note: These parameters are designed for Non-IID scenarios with potential class imbalance
    parser.add_argument('--beta_pi', type=float, default=0.5,
                        help='SFD parameter: coefficient for computing smoothed class distribution π in L_ACE (default: 0.5, slightly reduced for Non-IID class-missing compensation)')
    parser.add_argument('--a_ce_gamma', type=float, default=0.1,
                        help='SFD parameter: logit adjustment strength in L_ACE loss (default: 0.1)')
    parser.add_argument('--scl_weight_start', type=float, default=1.0,
                        help='Initial weight for L_A-SCL contrastive loss (default: 1.0, for contrastive learning weight annealing)')
    parser.add_argument('--scl_weight_end', type=float, default=0.0,
                        help='Final weight for L_A-SCL contrastive loss (cosine annealing target, default: 0.0)')
    parser.add_argument('--scl_temperature', type=float, default=0.07,
                        help='Temperature coefficient for L_A-SCL contrastive learning (default: 0.07). Note: different from --temperature which is for FedMPS contrastive loss (default: 0.5)')

    # TensorBoard logging options
    parser.add_argument('--log_client_acc', type=int, default=0,
                        help='Whether to log each client\'s accuracy to TensorBoard (0: disabled, 1: enabled, default: 0)')

    # SFD Statistics Aggregation options
    parser.add_argument('--enable_stats_agg', type=int, default=0,
                        help='Whether to enable global statistics aggregation (0: disabled, 1: enabled, default: 0). Required if SAFS is enabled.')

    # ===================== Stage-1 persistence / checkpointing =====================
    # Dataset split persistence (for multi-stage consistency)
    parser.add_argument('--split_path', type=str, default=None,
                        help='Path to persist/reuse dataset split (pickle). If None, defaults to <log_dir>/split.pkl')
    parser.add_argument('--reuse_split', type=int, default=1,
                        help='If split_path exists, reuse it (1) or regenerate and overwrite (0). Default: 1')

    # Resume training
    parser.add_argument('--resume_ckpt_path', type=str, default=None,
                        help='Path to a checkpoint to resume Stage-1 training from (typically latest.pt)')

    # Safety guard: prevent accidental overwrite of an existing run
    parser.add_argument('--allow_restart', type=int, default=0,
                        help='If 0, refuse to start from scratch when <log_dir>/stage1_ckpts/latest.pt exists. '
                             'Set to 1 only when you intentionally want to restart and overwrite latest.pt. Default: 0')

    # Stage-1 checkpoints
    parser.add_argument('--stage', type=int, default=1,
                        help='Training stage selector (reserved for multi-stage pipeline). Default: 1')
    # ===================== Stage-2 statistics aggregation =====================
    # Stage-2 loads a Stage-1 checkpoint (typically best-wo.pt) and computes global statistics (low-only by default).
    parser.add_argument('--stage1_ckpt_path', type=str, default=None,
                        help='[Stage-2] Path to a Stage-1 checkpoint to load (e.g., <log_dir>/stage1_ckpts/best-wo.pt). '
                             'If provided, stage-2 will infer log_dir/split_path from the checkpoint meta when possible.')
    parser.add_argument('--stage2_out_dir', type=str, default=None,
                        help='[Stage-2] Output directory to save stage-2 aggregated statistics. '
                             'If None, defaults to <ckpt.meta.logdir>/stage2_stats (or <log_dir>/stage2_stats).')
    
    # ===================== Stage-3 generator training =====================
    # Stage-3 trains a stats-conditioned generator for low-level features using Stage-2 global_stats.pt.
    parser.add_argument('--stage2_stats_path', type=str, default=None,
                        help='[Stage-3] Path to Stage-2 global_stats.pt (e.g., <log_dir>/stage2_stats/global_stats.pt).')
    parser.add_argument('--stage3_out_dir', type=str, default=None,
                        help='[Stage-3] Output directory to save generator artifacts. If None, defaults to <log_dir>/stage3_gen')

    # Generator architecture
    parser.add_argument('--gen_noise_dim', type=int, default=64, help='[Stage-3] Generator noise dimension')
    parser.add_argument('--gen_y_emb_dim', type=int, default=32, help='[Stage-3] Generator label embedding dimension')
    parser.add_argument('--gen_stat_emb_dim', type=int, default=128, help='[Stage-3] Generator stats embedding dimension')
    parser.add_argument('--gen_hidden_dim', type=int, default=256, help='[Stage-3] Generator hidden dimension')
    parser.add_argument('--gen_n_hidden_layers', type=int, default=2, help='[Stage-3] Number of hidden layers in generator MLP')
    parser.add_argument('--gen_relu_output', type=int, default=1, help='[Stage-3] Use ReLU on generator output (0/1)')
    parser.add_argument('--gen_use_cov_diag', type=int, default=1, help='[Stage-3] Condition on diag(cov) in addition to mean (0/1)')

    # Generator training
    parser.add_argument('--gen_steps', type=int, default=2000, help='[Stage-3] Training steps for generator')
    parser.add_argument('--gen_batch_size', type=int, default=256, help='[Stage-3] Batch size for generator training')
    parser.add_argument('--gen_lr', type=float, default=1e-3, help='[Stage-3] Learning rate for generator training')
    parser.add_argument('--gen_seed', type=int, default=1234, help='[Stage-3] Random seed for generator training')

    # Loss weights
    parser.add_argument('--gen_w_mean', type=float, default=1.0, help='[Stage-3] Weight for mean matching loss')
    parser.add_argument('--gen_w_var', type=float, default=0.1, help='[Stage-3] Weight for diag-variance matching loss')
    parser.add_argument('--gen_w_rff', type=float, default=1.0, help='[Stage-3] Weight for RFF mean matching loss')
    parser.add_argument('--gen_w_div', type=float, default=0.01, help='[Stage-3] Weight for diversity regularizer')
    parser.add_argument('--gen_w_arr', type=float, default=0.0, help='[Stage-3] Weight for non-negativity range regularizer (ARR)')
    parser.add_argument('--stage1_ckpt_dir', type=str, default=None,
                        help='Directory to store Stage-1 checkpoints. If None, defaults to <log_dir>/stage1_ckpts')
    parser.add_argument('--save_latest_ckpt', type=int, default=1,
                        help='Whether to save latest checkpoint for resume (0/1). Default: 1')
    parser.add_argument('--latest_ckpt_interval', type=int, default=1,
                        help='Save latest checkpoint every N rounds (overwritten). Default: 1')
    parser.add_argument('--save_latest_history', type=int, default=1,
                        help='If 1, also save non-overwriting snapshots (latest_rXXXX.pt) to avoid losing progress. Default: 1')
    parser.add_argument('--latest_history_interval', type=int, default=25,
                        help='Save latest_rXXXX.pt every N rounds (non-overwrite). Default: 25')
    parser.add_argument('--save_best_ckpt', type=int, default=1,
                        help='Whether to save best checkpoints (best-wo/best-wp) without overwrite (0/1). Default: 1')
    parser.add_argument('--best_ckpt_overwrite', type=int, default=1,
                        help='If 1, keep only one best-wo.pt and one best-wp.pt by overwriting. If 0, keep historical best checkpoints with round/metrics in filename. Default: 1')
    parser.add_argument('--save_stage1_components', type=int, default=1,
                        help='Whether to export 4-component state_dicts (low/high/projector/head) into checkpoints (0/1). Default: 1')

    args = parser.parse_args()
    return args
