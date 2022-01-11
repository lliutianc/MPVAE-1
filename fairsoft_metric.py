from math import log
import sys
import os
import pickle
import types

import torch

import numpy as np

from utils import search_files, build_path
from mpvae import VAE
from data import load_data
from logger import Logger

from label_distance import hamming_nonlinear_similarity, jaccard_nonlinear_similarity, apriori_similarity, indication_similarity, constant_similarity
from fairsoft_evaluate import IMPLEMENTED_METHODS, evaluate_mpvae
from fairsoft_utils import retrieve_target_label_idx

from main import THRESHOLDS, METRICS

sys.path.append('./')


def evaluate_models_over_label_distances(args):
    if args.eval_models == []:
        args.eval_models = IMPLEMENTED_METHODS

    if args.eval_distance in ['ham', 'hamming']:
        similarity = hamming_nonlinear_similarity
        hparam_distance = 'hamming'

    elif args.eval_distance in ['jac', 'jaccard']:
        similarity = jaccard_nonlinear_similarity
        hparam_distance = 'jaccard'

    elif args.eval_distance in ['apriori' 'arule']:
        similarity = apriori_similarity
        hparam_distance = 'arule'
    else:
        raise ValueError(
            f'unrecognized `args.eval_distance` value: {args.distance}')

    np.random.seed(4)
    _, _, labels, _, _ = load_data(args.dataset, args.mode, True)
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx
    target_fair_labels = label_type[idx: idx + 1].astype(int)

    np.random.seed(4)
    nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data(
        args.dataset, args.mode, True, 'onehot')

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx,
        valid_idx=valid_idx, batch_size=args.batch_size, label_clusters=None,
        sensitive_feat=sensitive_feat)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    if args.mask_target_label:
        logger = Logger(os.path.join(
            args.model_dir, f'sim_evaluation-{args.target_label_idx}_masked.txt'))
    else:
        logger = Logger(os.path.join(
            args.model_dir, f'sim_evaluation-{args.target_label_idx}.txt'))

    model_paths = []
    for model in args.eval_models:
        if model != 'unfair':
            model += f'_{args.target_label_idx}'
            if args.mask_target_label:
                model += '_masked'
        model_files = search_files(os.path.join(
            args.model_dir, model), postfix='.pkl')
        if len(model_files):
            model_paths += [os.path.join(
                args.model_dir, model, mod) for mod in model_files]
    logger.logging('\n' * 5)
    logger.logging(f"""Fair Models to evaluate: {model_paths}""")

    results = {}
    for model_stat in model_paths:
        print(f'Fair model: {model_stat}')
        model = VAE(args).to(args.device)
        model.load_state_dict(torch.load(model_stat))

        model_trained = model_stat.replace(
            '.pkl', '').split('/')[-1]
        if 'unfair' in model_trained:
            model_trained = 'unfair'
        else:
            model_trained = '-'.join(model_trained.split('-')[1:])

        for gamma in [.05, .1, .2, .5, 1., 1.5, 2., 5.]:
            dist_metric = f'{hparam_distance}_{gamma}'
            label_dist_path = os.path.join(
                args.model_dir, 'sim_evaluation',
                f'label_dist-{dist_metric}.npy')

            if args.train_new == 0 and os.path.exists(label_dist_path):
                label_dist = pickle.load(open(label_dist_path, 'rb'))
            else:
                label_dist = similarity(args, gamma)
                pickle.dump(label_dist, open(label_dist_path, 'wb'))

            train, valid = evaluate_mpvae(
                model, data, target_fair_labels, label_dist, args, logger=logger)
            results[model_trained][dist_metric] = f"{round(train['fair_mean_diff'], 5)}~({round(valid['fair_mean_diff'], 5)})"

        # run two baseline methods: EO and DP.
        dist_metric = f'indication_function'
        label_dist_path = os.path.join(
            args.model_dir, 'sim_evaluation',
            f'label_dist-{dist_metric}.npy')
        if args.train_new == 0 and os.path.exists(label_dist_path):
            label_dist = pickle.load(open(label_dist_path, 'rb'))
        else:
            label_dist = indication_similarity(args)
            pickle.dump(label_dist, open(label_dist_path, 'wb'))

        train, valid = evaluate_mpvae(
            model, data, target_fair_labels, label_dist, args, logger=logger)
        results[model_trained][dist_metric] = f"{round(train['fair_mean_diff'], 5)}~({round(valid['fair_mean_diff'], 5)})"

        dist_metric = f'constant_function'
        label_dist_path = os.path.join(
            args.model_dir, 'sim_evaluation',
            f'label_dist-{dist_metric}.npy')
        if args.train_new == 0 and os.path.exists(label_dist_path):
            label_dist = pickle.load(open(label_dist_path, 'rb'))
        else:
            label_dist = constant_similarity(args)
            pickle.dump(label_dist, open(label_dist_path, 'wb'))

        train, valid = evaluate_mpvae(
            model, data, target_fair_labels, label_dist, args, logger=logger)
        results[model_trained][dist_metric] = f"{round(train['fair_mean_diff'], 5)}~({round(valid['fair_mean_diff'], 5)})"

        # TODO: remove this break after developement.
        break

    models = list(results.keys())
    fair_metrics = [k for k in results[models[0]].keys()
                    if 'function' not in k]
    fair_metrics.sort()
    fair_metrics = ['constant_function'] + \
        fair_metrics + ['indication_function']
    colnames = ' & ' + ' & '.join(fair_metrics)
    logger.logging(colnames + '\\\\')
    logger.logging('\\midrule')
    for mod in models:
        result = []
        for met in fair_metrics:
            result.append(results[mod][met])
        resultrow = mod + ' & ' + ' & '.join(result)
        logger.logging(resultrow)
    logger.logging('\\buttomrule')


if __name__ == '__main__':
    from faircluster_train import parser
    parser.add_argument('-eval_models', type=str,
                        nargs='+', default=['unfair'])
    parser.add_argument('-eval_distance', type=str, default='jac')
    parser.add_argument('-min_support', type=float, default=None)
    parser.add_argument('-min_confidence', type=float, default=0.25)
    parser.add_argument('-dist_gamma', type=float, default=1.0)
    parser.add_argument('-target_label_idx', type=int, default=None)
    parser.add_argument('-target_label', type=str, default=None)
    parser.add_argument('-mask_target_label', type=int, default=0)
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    args.model_dir = f'fair_through_distance/model/{args.dataset}'
    build_path(args.model_dir, args.summary_dir,
               os.path.join(args.model_dir, 'sim_evaluation'))

    if args.target_label is not None:
        args.target_label_idx = retrieve_target_label_idx(
            args, args.target_label)
        evaluate_models_over_label_distances(args)

    elif args.target_label_idx is not None:
        evaluate_models_over_label_distances(args)

    else:
        for target_label_idx in [0, 10, 20, 50]:
            args.target_label_idx = target_label_idx
            evaluate_models_over_label_distances(args)


# python fairsoft_metric.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -cuda 5
