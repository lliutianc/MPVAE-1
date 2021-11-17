import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np

import sys
import os
import datetime
from copy import deepcopy
import types


import evals
from utils import build_path
from model import VAE, compute_loss
from data import load_data
from embed import CBOW
from label_cluster import construct_label_clusters
from train import train_mpvae_one_epoch, validate_mpvae
from main import parser, THRESHOLDS, METRICS

# cluster parameters
parser.add_argument('-labels_embed_method', type=str,
                    choices=['cbow', 'mpvae', 'none'])
parser.add_argument('-labels_cluster_method', type=str, default='apriori')
parser.add_argument('-labels_cluster_distance_threshold',
                    type=float, default=None)
parser.add_argument('-labels_cluster_num', type=int, default=None)
parser.add_argument('-labels_cluster_min_size', type=int, default=50)
# fair regularizer
parser.add_argument('-label_z_fair_coeff', type=float, default=1.0)
parser.add_argument('-feat_z_fair_coeff', type=float, default=1.0)
parser.add_argument('-min_support', type=float, default=None)
parser.add_argument('-min_confidence', type=float, default=None)

sys.path.append('./')


def train_fair_through_regularize():

    hparams = f'label_cluster-'\
            f'n_cluster={args.labels_cluster_num}-'\
            f'dist_cluster={args.labels_cluster_distance_threshold}-'\
            f'min_support={args.min_support}-'\
            f'min_confidence={args.min_confidence}'
    label_cluster_path = os.path.join(
        args.model_dir, hparams + '.npy')

    if args.resume and os.path.exists(label_cluster_path):
        label_clusters = np.load(open(label_cluster_path, 'rb'))
    else:
        label_clusters = construct_label_clusters(args)
        np.save(open(label_cluster_path, 'wb'), label_clusters)

    # retrain a new mpvae + fair regularization
    np.random.seed(4)
    nonsensitive_feat, sensitive_feat, labels = load_data(
        args.dataset, args.mode, True, 'onehot')

    train_cnt, valid_cnt = int(
        len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)
    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx, valid_idx=valid_idx,
        batch_size=args.batch_size, label_clusters=label_clusters, sensitive_feat=sensitive_feat)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    fair_vae = VAE(args).to(args.device)
    fair_vae.train()

    fair_vae_checkpoint_path = os.path.join(
        args.model_dir, f'fair_vae_prior_{hparams}.pkl')
    if args.resume and os.path.exists(fair_vae_checkpoint_path):
        print('use a trained fair mpvae...')
        fair_vae.load_state_dict(torch.load(fair_vae_checkpoint_path))
    else:
        print('train a new fair mpvae...')

    optimizer = optim.Adam(fair_vae.parameters(),
                           lr=args.learning_rate, weight_decay=1e-5)
    # optimizer_fair = optim.Adam(
    #     fair_vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

    print('start training fair mpvae...')
    for _ in range(args.max_epoch):
        train_mpvae_one_epoch(
            data, fair_vae, optimizer, scheduler,
            penalize_unfair=True, eval_after_one_epoch=True, args=args)
        # regularzie_mpvae_unfair(data, fair_vae, optimizer_fair, args, use_valid=True)

    torch.save(fair_vae.cpu().state_dict(), fair_vae_checkpoint_path)


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if args.labels_cluster_method == 'kmodes':
        if args.labels_embed_method != 'none':
            raise ValueError('Cannot run K-modes on embedded representations')

    # args.device = torch.args.device('cpu')
    param_setting = f"n_cluster={args.labels_cluster_num}-"\
                    f"cluster_distance_thre={args.labels_cluster_distance_threshold}"

    args.model_dir = f'fair_through_arule/model/{args.dataset}/{param_setting}'
    args.summary_dir = f'fair_through_arule/summary/{args.dataset}/{param_setting}'
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize()
