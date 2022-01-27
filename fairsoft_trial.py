import sys
import os
import pickle

import torch
import numpy as np

from utils import allexists, build_path
from logger import Logger
from fairsoft_utils import retrieve_target_label_idx

sys.path.append('./')

IMPLEMENTED_METHODS = ['baseline', 'unfair', 'jaccard']


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
    args.mode = 'test'
    from fairsoft_evaluate import evaluate_target_labels

    args.model_dir = f'fair_through_distance/model/{args.dataset}'
    if args.mask_target_label:
        logger = Logger(os.path.join(
            args.model_dir, f'evaluation-{args.target_label_idx}_masked.txt'))
        eval_results_path = os.path.join(
            args.model_dir, f'evaluation-{args.target_label_idx}_masked')
    else:
        logger = Logger(os.path.join(
            args.model_dir, f'evaluation-{args.target_label_idx}.txt'))
        eval_results_path = os.path.join(
            args.model_dir, f'evaluation-{args.target_label_idx}')
    build_path(eval_results_path)

    fair_results_path = os.path.join(
        eval_results_path, f'fair_eval_lambda={args.fair_coeff:.2f}_{args.seed:04d}.pkl')
    perform_results_path = os.path.join(
        eval_results_path, f'perform_eval_lambda={args.fair_coeff:.2f}_{args.seed:04d}.pkl')

    if allexists(fair_results_path, perform_results_path) and bool(args.train_new) is False:
        print(
            f'find evaluation results: {fair_results_path}, {perform_results_path}')
        fair_results = pickle.load(open(fair_results_path, 'rb'))
        perform_results = pickle.load(open(perform_results_path, 'rb'))
    else:
        print(
            f'create new evalution results: {fair_results_path}, {perform_results_path}')
        fair_results, perform_results = evaluate_target_labels(args, logger)
        pickle.dump(fair_results, open(fair_results_path, 'wb'))
        pickle.dump(perform_results, open(perform_results_path, 'wb'))

    fair_metrics = list(fair_results.keys())
    fair_metrics_nested = {}
    fair_metrics_sorted = []
    should_add_eo = False
    for met_hparam in fair_metrics:
        met = met_hparam.split('_')[0]
        if met not in fair_metrics_nested:
            fair_metrics_nested[met] = []
        fair_metrics_nested[met].append(met_hparam)

    # for met in ['constant_function', 'jaccard', 'hamming', 'arule', 'indication_function']:
    for met in ['constant', 'jaccard', 'indication']:
        if met in fair_metrics_nested:
            if len(fair_metrics_nested[met]) > 1:
                met_sorted = sorted(
                    fair_metrics_nested[met], key=lambda met: float(met.split('_')[-1]))
            else:
                met_sorted = fair_metrics_nested[met]
            fair_metrics_sorted += met_sorted
    fair_metrics = fair_metrics_sorted

    colnames = ' & ' + ' & '.join(fair_metrics)
    logger.logging(colnames + '\\\\')
    logger.logging('\\midrule')
    for met in fair_metrics:
        result = []
        for mod in fair_metrics + ['unfair']:
            train, valid = fair_results[met][mod]
            result.append(f"{round(train, 5)}~({round(valid, 5)})")

        resultrow = met + ' & ' + ' & '.join(result)
        logger.logging(resultrow + '\\\\')

    for perform_metric in args.perform_metric:
        result = []
        for mod in fair_metrics + ['unfair']:
            train, valid = perform_results[mod][perform_metric]
            result.append(f"{round(train, 5)}~({round(valid, 5)})")
        resultrow = perform_metric + ' & ' + ' & '.join(result)
        logger.logging(resultrow + '\\\\')
    logger.logging('\\bottomrule')


if __name__ == '__main__':
    from main import parser
    parser.add_argument('-min_support', type=float, default=None)
    parser.add_argument('-min_confidence', type=float, default=0.25)
    parser.add_argument('-dist_gamma', type=float, default=None)
    parser.add_argument('-target_label_idx', type=int, default=None)
    parser.add_argument('-target_label', type=str, default=None)
    parser.add_argument('-mask_target_label', type=int, default=0)
    parser.add_argument('-perform_metric', type=str, nargs='+',
                        default=['ACC', 'HA', 'ebF1', 'maF1', 'miF1'])
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    if args.target_label is not None:
        args.target_label_idx = retrieve_target_label_idx(
            args, args.target_label)
        args.penalize_unfair = 0
        train_fairsoft_baseline(args)

        args.penalize_unfair = 1
        for dist_gamma in [.01, 1., 5., 10.]:
            args.dist_gamma = dist_gamma
            train_fairsoft_jaccard(args)
            # train_fairsoft_arule(args)

        # train_fairsoft_hamming(args)
        train_fairsoft_baseline(args)
        eval_fairsoft_allmodels(args)

    elif args.target_label_idx is not None:
        args.penalize_unfair = 0
        train_fairsoft_baseline(args)

        args.penalize_unfair = 1
        for dist_gamma in [.01, 1., 5., 10.]:
            args.dist_gamma = dist_gamma
            train_fairsoft_jaccard(args)

            # train_fairsoft_arule(args)  # 20308 samples

        # train_fairsoft_hamming(args)  # 21587 samples
        # train_fairsoft_jaccard(args)  # 19604 samples
        train_fairsoft_baseline(args)  # eo: 641 samples, dp: 21587 samples
        eval_fairsoft_allmodels(args)

    else:
        args.penalize_unfair = 0
        train_fairsoft_baseline(args)

        for target_label_idx in [0, 10, 20, 50]:
            args.target_label_idx = target_label_idx

            args.penalize_unfair = 0
            train_fairsoft_baseline(args)

            args.penalize_unfair = 1
            for dist_gamma in [.01, 1., 5., 10.]:
                args.dist_gamma = dist_gamma
                train_fairsoft_jaccard(args)
                # train_fairsoft_arule(args)

            # train_fairsoft_hamming(args)
            train_fairsoft_baseline(args)
            eval_fairsoft_allmodels(args)


# python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -cuda 5


# python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed 1 -epoch 50 -bs 16 -label_z_fair_coeff 1.
