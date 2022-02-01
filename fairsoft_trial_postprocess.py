import sys
import os
import pickle

import torch
import numpy as np

from utils import allexists, build_path
from logger import Logger
from fairsoft_utils import retrieve_target_label_idx
from fairsoft_train_postprocess import train_fair_through_postprocess, evaluate_target_labels, IMPLEMENTED_METHODS

sys.path.append('./')


def train_fairsoft_postprocess(args):
    for label_dist in ['constant', 'indication']:
        param_setting = f"baseline_{args.target_label_idx}"
        if args.mask_target_label:
            param_setting += '_masked'

        args.model_dir = f"fair_through_postprocess/model/{args.dataset}/{param_setting}"
        build_path(args.model_dir)
        args.label_dist = label_dist
        train_fair_through_postprocess(args)

    param_setting = f"jaccard_{args.target_label_idx}"
    if args.mask_target_label:
        param_setting += '_masked'

    args.model_dir = f"fair_through_postprocess/model/{args.dataset}/{param_setting}"
    build_path(args.model_dir)
    args.label_dist = 'jaccard'
    for dist_gamma in [.01, 1., 5., 10.]:
        args.dist_gamma = dist_gamma
        train_fair_through_postprocess(args)


def eval_fairsoft_allmodels_postprocess(args):

    args.model_dir = f"fair_through_postprocess/model/{args.dataset}"

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
    for met_hparam in fair_metrics:
        met = met_hparam.split('_')[0]
        if met not in fair_metrics_nested:
            fair_metrics_nested[met] = []
        fair_metrics_nested[met].append(met_hparam)

    for met in ['constant', 'jaccard', 'indication']:
        if met in fair_metrics_nested:
            if len(fair_metrics_nested[met]) > 1:
                met_sorted = sorted(
                    fair_metrics_nested[met], key=lambda met: float(met.split('_')[-1]))
            else:
                met_sorted = fair_metrics_nested[met]
            fair_metrics_sorted += met_sorted
    fair_metrics = fair_metrics_sorted

    print(fair_metrics)
    colnames = ' & ' + ' & '.join(fair_metrics)
    logger.logging(colnames + '\\\\')
    logger.logging('\\midrule')
    for met in fair_metrics:
        result = []
        for mod in fair_metrics:
            train, valid, test = fair_results[met][mod]
            result.append(f"{train:.5f}({valid:.5f})({test:.5f})")

        resultrow = met + ' & ' + ' & '.join(result)
        logger.logging(resultrow + '\\\\')

    for perform_metric in args.perform_metric:
        result = []
        for mod in fair_metrics:
            train, valid, test = perform_results[mod][perform_metric]
            result.append(f"{train:.5f}({valid:.5f})({test:.5f})")
        resultrow = perform_metric + ' & ' + ' & '.join(result)
        logger.logging(resultrow + '\\\\')
    logger.logging('\\bottomrule')


if __name__ == '__main__':
    from main import parser
    parser.add_argument('-learn_logit', type=int, default=1)
    parser.add_argument('-dist_gamma', type=float, default=None)
    parser.add_argument('-target_label_idx', type=int, default=None)
    parser.add_argument('-target_label', type=str, default=None)
    parser.add_argument('-mask_target_label', type=int, default=0)
    parser.add_argument('-perform_metric', type=str, nargs='+',
                        default=['ACC', 'HA', 'ebF1', 'maF1', 'miF1'])
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    print(args.device)
    if args.target_label is not None:
        args.target_label_idx = retrieve_target_label_idx(
            args, args.target_label)
        train_fairsoft_postprocess(args)
        eval_fairsoft_allmodels_postprocess(args)

    elif args.target_label_idx is not None:
        train_fairsoft_postprocess(args)
        eval_fairsoft_allmodels_postprocess(args)

    else:

        for target_label_idx in [0, 10, 20, 50]:
            args.target_label_idx = target_label_idx
            train_fairsoft_postprocess(args)
            eval_fairsoft_allmodels_postprocess(args)


# python fairsoft_trial_postprocess.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -cuda 5
