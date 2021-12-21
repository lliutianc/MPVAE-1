import sys
import os
import pickle
import types

import torch
from torch import optim

import numpy as np

from utils import build_path
from mpvae import VAE
from data import load_data
from label_distance import apriori_distance

from main import THRESHOLDS, METRICS
from fairsoft_train import train_mpvae_softfair_one_epoch


def train_fair_through_regularize(args):

    hparams = f'min_support={args.min_support}-'\
              f'min_confidence={args.min_confidence}-'\
              f'gamma={args.dist_gamma}'
    label_dist_path = os.path.join(
        args.model_dir, f'label_dist-{hparams}.npy')

    if args.resume and os.path.exists(label_dist_path):
        label_dist = pickle.load(open(label_dist_path, 'rb'))
    else:
        label_dist = apriori_distance(args, args.dist_gamma)
        pickle.dump(label_dist, open(label_dist_path, 'wb'))

    np.random.seed(4)
    nonsensitive_feat, sensitive_feat, labels = load_data(
        args.dataset, args.mode, True, 'onehot')

    # Test fairness on some labels
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_labels = label_type[idx: idx + 1].astype(int)
    # print(target_fair_labels)

    train_cnt, valid_cnt = int(
        len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)
    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, sensitive_feat=sensitive_feat,
        train_idx=train_idx, valid_idx=valid_idx, batch_size=args.batch_size)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    fair_vae = VAE(args).to(args.device)
    fair_vae.train()

    fair_vae_checkpoint_path = os.path.join(
        args.model_dir, f'fair_vae_prior-{hparams}.pkl')
    if args.resume and os.path.exists(fair_vae_checkpoint_path):
        print('use a trained fair mpvae...')
        fair_vae.load_state_dict(torch.load(fair_vae_checkpoint_path))
    else:
        print('train a new fair mpvae...')

    optimizer = optim.Adam(fair_vae.parameters(),
                           lr=args.learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

    print('start training fair mpvae...')
    for _ in range(args.max_epoch):
        train_mpvae_softfair_one_epoch(
            data, fair_vae, optimizer, scheduler,
            penalize_unfair=True, target_fair_labels=target_fair_labels, label_distances=label_dist,
            eval_after_one_epoch=True, args=args)
    torch.save(fair_vae.cpu().state_dict(), fair_vae_checkpoint_path)


if __name__ == '__main__':
    from faircluster_train import parser

    parser.add_argument('-min_support', type=float, default=None)
    parser.add_argument('-min_confidence', type=float, default=None)
    parser.add_argument('-dist_gamma', type=float, default=1.0)
    parser.add_argument('-target_label_idx', type=int, default=0)
    
    sys.path.append('./')

    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    param_setting = f"arule_{args.target_label_idx}"
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)

# python fairsoft_arule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -min_confidence 0.25  -cuda 5 -target_label_idx 0
# python fairsoft_arule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -min_confidence 0.25  -cuda 5 -target_label_idx 0 -dist_gamma 0.1
