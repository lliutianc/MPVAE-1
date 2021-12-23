import sys
import os

from joblib.logger import Logger

import torch
import numpy as np

from utils import build_path
from logger import Logger

sys.path.append('./')


def train_fairsoft_arule(args):
    from fairsoft_arule import train_fair_through_regularize

    param_setting = f"arule_{args.target_label_idx}"
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def train_fairsoft_baseline(args):
    from fairsoft_baseline import train_fair_through_regularize

    param_setting = f"baseline_{args.target_label_idx}" if args.penalize_unfair else f"unfair"
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def eval_fairsoft_allmodels(args):
    from fairsoft_evaluate import evaluate_target_labels

    args.model_dir = f'fair_through_distance/model/{args.dataset}'
    logger = Logger(os.path.join(
        args.model_dir, f'evalution-{args.target_label_idx}.txt'))

    evaluate_target_labels(args, logger)


def retrieve_target_label_idx(args, target_label):
    from data import load_data

    if len(target_label) > 1 and isinstance(target_label, str) == False:
        raise NotImplementedError(
            'cannot handle multiple target labels yet...')
    _, _, labels = load_data(
        args.dataset, args.mode, True, 'onehot')

    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    for idx, lab in enumerate(label_type):
        if ''.join(lab.astype(int).astype(str)) == target_label:
            return idx 
    
    return None


if __name__ == '__main__':
    from faircluster_train import parser
    parser.add_argument('-min_support', type=float, default=None)
    parser.add_argument('-min_confidence', type=float, default=0.25)
    parser.add_argument('-dist_gamma', type=float, default=1.0)
    parser.add_argument('-target_label_idx', type=int, default=None)
    parser.add_argument('-target_label', type=str, default=None)
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # train unfair model
    args.penalize_unfair = 0
    train_fairsoft_baseline(args)

    args.penalize_unfair = 1

    # args.target_label = '0000000000000001000100000'
    if args.target_label is not None: 
        target_label_idx = retrieve_target_label_idx(args, args.target_label)
        args.target_label_idx = target_label_idx
        for dist_gamma in [.1, .5, 1., 1.5, 2.]:
            args.dist_gamma = dist_gamma
            train_fairsoft_arule(args)

        train_fairsoft_baseline(args)
        eval_fairsoft_allmodels(args)
    else:
        for target_label_idx in [0, 10, 20, 50]:
            args.target_label_idx = target_label_idx
            for dist_gamma in [.1, .5, 1., 1.5, 2.]:
                args.dist_gamma = dist_gamma
                train_fairsoft_arule(args)

            train_fairsoft_baseline(args)
            eval_fairsoft_allmodels(args)

            # TODO: remove this break after developement!
            break


# python fairsoft_trial.py -dataset adult -latent_dim 8 -cuda 5
