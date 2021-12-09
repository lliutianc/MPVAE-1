import torch
from torch import optim

import numpy as np

import sys
import os
import datetime
from copy import deepcopy
import types
from tqdm import tqdm

from utils import build_path
from mpvae import VAE
from data import load_data
from train import train_mpvae_one_epoch

from main import parser


sys.path.append('./')
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]

METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC', 'meanAUPR', 'medianAUPR',
           'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']


def train_without_regularize():
    np.random.seed(4)

    nonsensitive_feat, sensitive_feat, labels = load_data(args.dataset, args.mode, True)
    # print(nonsensitive_feat.max(0), nonsensitive_feat.min(0))
    # print(labels.max(0), labels.min(0))
    train_cnt, valid_cnt = int(len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)

    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx, valid_idx=valid_idx,
        batch_size=args.batch_size, label_clusters=None, sensitive_feat=sensitive_feat)
    args.feature_dim = nonsensitive_feat.shape[1]
    args.label_dim = labels.shape[1]
    print(args.feature_dim, args.label_dim)

    base_vae = VAE(args).to(device)
    base_vae.train()

    base_vae_checkpoint_path = os.path.join(model_dir, f'baseline_vae')
    if args.resume and os.path.exists(base_vae_checkpoint_path):
        print('use a baseline fair mpvae...')
        base_vae.load_state_dict(torch.load(base_vae_checkpoint_path))
    else:
        print('train a new fair mpvae...')

    optimizer = optim.Adam(base_vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

    print('start training fair mpvae...')
    for _ in range(args.max_epoch):
        train_mpvae_one_epoch(
            data, base_vae, optimizer, scheduler,
            penalize_unfair=False, eval_after_one_epoch=True, args=args)

    torch.save(base_vae.cpu().state_dict(), base_vae_checkpoint_path)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    param_setting = f"lr-{args.learning_rate}_" \
                    f"lr-decay_{args.lr_decay_ratio}_" \
                    f"lr-times_{args.lr_decay_times}_" \
                    f"nll-{args.nll_coeff}_" \
                    f"l2-{args.l2_coeff}_" \
                    f"c-{args.c_coeff}"

    model_dir = f'fairreg/model/{args.dataset}/{param_setting}'
    summary_dir = f'fairreg/summary/{args.dataset}/{param_setting}'
    build_path(model_dir, summary_dir)

    train_without_regularize()