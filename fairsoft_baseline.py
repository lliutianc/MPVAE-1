import sys
import os
import pickle
import datetime
import types

import torch
from torch import optim

import numpy as np
import pandas as pd

from utils import build_path
from mpvae import VAE
from data import load_data, load_data_masked
from label_distance_obs import indication_similarity, constant_similarity

from main import THRESHOLDS, METRICS
from fairsoft_train import train_mpvae_softfair_one_epoch


def train_fair_through_regularize(args):
    train_fair_equalodds_through_regularize(args)
    train_fair_demparity_through_regularize(args)


def train_fair_demparity_through_regularize(args):
    hparams = f'constant'
    label_dist_path = os.path.join(
        args.model_dir, f'label_dist-{hparams}.npy')

    if args.train_new == 0 and os.path.exists(label_dist_path):
        label_dist = pickle.load(open(label_dist_path, 'rb'))
    else:
        label_dist = constant_similarity(args)
        pickle.dump(label_dist, open(label_dist_path, 'wb'))

    np.random.seed(args.seed)
    if args.mask_target_label:
        nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data_masked(
            args.dataset, args.mode, True, 'onehot')
    else:
        nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data(
            args.dataset, args.mode, True, 'onehot')

    # Test fairness on some labels
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_labels = label_type[idx: idx + 1].astype(int)

    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, sensitive_feat=sensitive_feat,
        train_idx=train_idx, valid_idx=valid_idx, batch_size=args.batch_size)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    fair_vae = VAE(args).to(args.device)
    fair_vae.train()

    if args.penalize_unfair:
        fair_vae_checkpoint_path = os.path.join(
            args.model_dir, f'fair_vae_prior-{hparams}-{args.label_z_fair_coeff:.2f}_{args.seed:04d}.pkl')
    else:
        fair_vae_checkpoint_path = os.path.join(
            args.model_dir, f'unfair_vae_prior-{args.label_z_fair_coeff:.2f}_{args.seed:04d}.pkl')

    if args.train_new == 0 and os.path.exists(fair_vae_checkpoint_path):
        print(f'find trained mpvae: {fair_vae_checkpoint_path}...')
    else:
        print(f'train a new mpvaeL: {fair_vae_checkpoint_path}...')

        optimizer = optim.Adam(fair_vae.parameters(),
                               lr=args.learning_rate, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        print('start training fair mpvae...')
        for _ in range(args.max_epoch):
            train_mpvae_softfair_one_epoch(
                data, fair_vae, optimizer, scheduler,
                penalize_unfair=args.penalize_unfair,
                target_fair_labels=target_fair_labels,
                label_distances=label_dist,
                eval_after_one_epoch=True,
                args=args)

        torch.save(fair_vae.cpu().state_dict(), fair_vae_checkpoint_path)


def train_fair_equalodds_through_regularize(args):
    hparams = f'indication'
    label_dist_path = os.path.join(
        args.model_dir, f'label_dist-{hparams}.npy')

    if args.train_new == 0 and os.path.exists(label_dist_path):
        label_dist = pickle.load(open(label_dist_path, 'rb'))
    else:
        label_dist = indication_similarity(args)
        pickle.dump(label_dist, open(label_dist_path, 'wb'))

    np.random.seed(args.seed)
    if args.mask_target_label:
        nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data_masked(
            args.dataset, args.mode, True, 'onehot')
    else:
        nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data(
            args.dataset, args.mode, True, 'onehot')

    # Test fairness on some labels
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_labels = label_type[idx: idx + 1].astype(int)

    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)
    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, sensitive_feat=sensitive_feat,
        train_idx=train_idx, valid_idx=valid_idx, batch_size=args.batch_size)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    fair_vae = VAE(args).to(args.device)
    fair_vae.train()

    if args.penalize_unfair:
        fair_vae_checkpoint_path = os.path.join(
            args.model_dir, f'fair_vae_prior-{hparams}-{args.label_z_fair_coeff:.2f}_{args.seed:04d}.pkl')
    else:
        fair_vae_checkpoint_path = os.path.join(
            args.model_dir, f'unfair_vae_prior-{args.label_z_fair_coeff:.2f}_{args.seed:04d}.pkl')

    if args.train_new == 0 and os.path.exists(fair_vae_checkpoint_path):
        print(f'find trained mpvae: {fair_vae_checkpoint_path}...')
    else:
        print(f'train a new mpvaeL: {fair_vae_checkpoint_path}...')

        optimizer = optim.Adam(fair_vae.parameters(),
                               lr=args.learning_rate, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        print('start training fair mpvae...')
        for _ in range(args.max_epoch):
            train_mpvae_softfair_one_epoch(
                data, fair_vae, optimizer, scheduler,
                penalize_unfair=args.penalize_unfair,
                target_fair_labels=target_fair_labels,
                label_distances=label_dist,
                eval_after_one_epoch=True,
                args=args)

        torch.save(fair_vae.cpu().state_dict(), fair_vae_checkpoint_path)


if __name__ == '__main__':
    from faircluster_train import parser
    parser.add_argument('-target_label_idx', type=int, default=0)
    parser.add_argument('-penalize_unfair', type=int, default=1)
    parser.add_argument('-mask_target_label', type=int, default=0)

    sys.path.append('./')

    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    param_setting = f"baseline_{args.target_label_idx}" if args.penalize_unfair else f"unfair"
    if args.mask_target_label:
        param_setting += 'masked'
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)

# python fairsoft_baseline.py -dataset adult -latent_dim 8 -epoch 20 -cuda 5 -target_label_idx 0 -penalize_unfair 0

# python fairsoft_baseline.py -dataset adult -latent_dim 8 -epoch 20 -cuda 5 -target_label_idx 0
