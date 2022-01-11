import sys
import os

from joblib.logger import Logger

import torch
import numpy as np

from utils import build_path
from logger import Logger
from fairsoft_utils import retrieve_target_label_idx

sys.path.append('./')


def train_fairsoft_arule(args):
    from fairsoft_arule import train_fair_through_regularize

    param_setting = f"arule_{args.target_label_idx}"
    if args.mask_target_label:
        param_setting += '_masked'
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def train_fairsoft_baseline(args):
    from fairsoft_baseline import train_fair_through_regularize

    param_setting = f"baseline_{args.target_label_idx}" if args.penalize_unfair else f"unfair"
    if args.mask_target_label:
        param_setting += '_masked'
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def train_fairsoft_hamming(args):
    from fairsoft_hamming import train_fair_through_regularize

    param_setting = f"hamming_{args.target_label_idx}"
    if args.mask_target_label:
        param_setting += '_masked'
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def train_fairsoft_jaccard(args):
    from fairsoft_jaccard import train_fair_through_regularize

    param_setting = f"jaccard_{args.target_label_idx}"
    if args.mask_target_label:
        param_setting += '_masked'
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize(args)


def eval_fairsoft_allmodels(args):
    from fairsoft_evaluate import evaluate_target_labels

    args.model_dir = f'fair_through_distance/model/{args.dataset}'
    if args.mask_target_label:
        logger = Logger(os.path.join(
            args.model_dir, f'evalution-{args.target_label_idx}_masked.txt'))
    else:
        logger = Logger(os.path.join(
            args.model_dir, f'evalution-{args.target_label_idx}.txt'))
    results = evaluate_target_labels(args, logger)

    fair_metrics = list(results.keys())
    fair_metrics.sort()
    colnames = ' & ' + ' & '.join(fair_metrics)
    logger.logging(colnames + '\\\\')
    logger.logging('\\midrule')
    for met in fair_metrics:
        result = []
        for mod in fair_metrics:
            result.append(results[met][mod])
        result.append(results[met]['unfair'])

        resultrow = met + ' & ' + ' & '.join(result)
        logger.logging(resultrow + '\\\\')
    logger.logging('\\buttomrule')


if __name__ == '__main__':
    from faircluster_train import parser
    parser.add_argument('-min_support', type=float, default=None)
    parser.add_argument('-min_confidence', type=float, default=0.25)
    parser.add_argument('-dist_gamma', type=float, default=1.0)
    parser.add_argument('-target_label_idx', type=int, default=None)
    parser.add_argument('-target_label', type=str, default=None)
    parser.add_argument('-mask_target_label', type=int, default=0)
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    if args.target_label is not None:
        args.target_label_idx = retrieve_target_label_idx(
            args, args.target_label)
        args.penalize_unfair = 0
        train_fairsoft_baseline(args)

        args.penalize_unfair = 1
        for dist_gamma in [1.]:
            args.dist_gamma = dist_gamma
            train_fairsoft_arule(args)

        train_fairsoft_hamming(args)
        train_fairsoft_jaccard(args)
        train_fairsoft_baseline(args)
        eval_fairsoft_allmodels(args)

    elif args.target_label_idx is not None:
        args.penalize_unfair = 0
        train_fairsoft_baseline(args)

        args.penalize_unfair = 1
        for dist_gamma in [1.]:
            args.dist_gamma = dist_gamma
            train_fairsoft_arule(args)  # 20308 samples

        train_fairsoft_hamming(args)  # 21587 samples
        train_fairsoft_jaccard(args)  # 19604 samples
        train_fairsoft_baseline(args)  # eo: 641 samples, dp: 21587 samples
        eval_fairsoft_allmodels(args)

    else:
        for target_label_idx in [0, 10, 20, 50]:
            args.target_label_idx = target_label_idx

            args.penalize_unfair = 0
            train_fairsoft_baseline(args)

            args.penalize_unfair = 1
            for dist_gamma in [1.]:
                args.dist_gamma = dist_gamma
                train_fairsoft_arule(args)

            train_fairsoft_hamming(args)
            train_fairsoft_jaccard(args)
            train_fairsoft_baseline(args)
            eval_fairsoft_allmodels(args)


# python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -cuda 5
