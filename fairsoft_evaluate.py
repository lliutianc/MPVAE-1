from joblib.logger import Logger
from faircluster_train import parser
from copy import deepcopy
import os
import types
import pickle

import torch

import numpy as np
from tqdm import tqdm

import evals
from mpvae import VAE, compute_loss
from data import load_data, load_data_masked
from faircluster_train import THRESHOLDS, METRICS
from utils import search_files
from logger import Logger

IMPLEMENTED_METHODS = ['arule', 'baseline', 'unfair', 'hamming', 'jaccard']


def evaluate_mpvae(model, data, target_fair_labels, label_distances, args, eval_fairness=True, eval_train=True, eval_valid=True, logger=Logger()):
    if eval_fairness and target_fair_labels is None:
        target_fair_labels = list(label_distances.keys())
        raise NotImplementedError('Have not supported smooth-OD yet.')

    target_fair_labels_str = []
    for target_fair_label in target_fair_labels:
        if isinstance(target_fair_label, np.ndarray):
            target_fair_label = ''.join(target_fair_label.astype(str))
        target_fair_labels_str.append(target_fair_label)
    target_fair_labels = target_fair_labels_str

    with torch.no_grad():
        model.eval()

        if eval_train:
            train_nll_loss = 0
            train_c_loss = 0
            train_total_loss = 0

            train_indiv_prob = []
            train_label = []

            if eval_fairness:
                train_feat_z = []
                train_sensitive = data.sensitive_feat[data.train_idx]
            with tqdm(
                    range(int(len(data.train_idx) / float(data.batch_size)) + 1),
                    desc='Evaluate on training set') as t:

                for i in t:
                    start = i * data.batch_size
                    end = min(data.batch_size * (i + 1), len(data.train_idx))
                    idx = data.train_idx[start:end]

                    input_feat = torch.from_numpy(
                        data.input_feat[idx]).to(args.device)

                    input_label = torch.from_numpy(data.labels[idx])
                    input_label = deepcopy(input_label).float().to(args.device)

                    label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                        input_label, input_feat)
                    total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
                        input_label, label_out, label_mu, label_logvar, feat_out, feat_mu,
                        feat_logvar, model.r_sqrt_sigma, args)

                    train_nll_loss += nll_loss.item() * (end - start)
                    train_c_loss += c_loss.item() * (end - start)
                    train_total_loss += total_loss.item() * (end - start)

                    for j in deepcopy(indiv_prob).cpu().data.numpy():
                        train_indiv_prob.append(j)
                    for j in deepcopy(input_label).cpu().data.numpy():
                        train_label.append(j)

                    if eval_fairness:
                        feat_z = model.feat_reparameterize(
                            feat_mu, feat_logvar)
                        train_feat_z.append(feat_z.cpu().data.numpy())

                train_indiv_prob = np.array(train_indiv_prob)
                train_label = np.array(train_label)

                nll_loss = train_nll_loss / len(data.train_idx)
                c_loss = train_c_loss / len(data.train_idx)
                total_loss = train_total_loss / len(data.train_idx)

                best_val_metrics = None
                for threshold in THRESHOLDS:
                    val_metrics = evals.compute_metrics(
                        train_indiv_prob, train_label, threshold, all_metrics=True)

                    if best_val_metrics is None:
                        best_val_metrics = {}
                        for metric in METRICS:
                            best_val_metrics[metric] = val_metrics[metric]
                    else:
                        for metric in METRICS:
                            if 'FDR' in metric:
                                best_val_metrics[metric] = min(
                                    best_val_metrics[metric], val_metrics[metric])
                            else:
                                best_val_metrics[metric] = max(
                                    best_val_metrics[metric], val_metrics[metric])

                acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], \
                    best_val_metrics['ebF1'], best_val_metrics['maF1'], \
                    best_val_metrics['miF1']

                if eval_fairness:
                    train_feat_z = np.concatenate(train_feat_z)
                    assert train_feat_z.shape[0] == len(data.train_idx) and \
                        train_feat_z.shape[1] == args.latent_dim

                    sensitive_centroid = np.unique(train_sensitive, axis=0)
                    idxs = np.arange(len(data.train_idx))

                    mean_diffs = []
                    for target_fair_label in target_fair_labels:
                        target_label_dist = label_distances[target_fair_label]
                        weights = []
                        for label in data.labels[idxs]:
                            label = label.astype(int)
                            distance = target_label_dist.get(
                                ''.join(label.astype(str)), 0.)
                            weights.append(distance)
                        weights = np.array(weights).reshape(-1, 1)
                        if weights.sum() > 0:
                            feat_z_weighted = np.sum(
                                train_feat_z * weights, axis=0) / weights.sum()

                            for sensitive in sensitive_centroid:
                                target_sensitive = np.all(
                                    np.equal(train_sensitive, sensitive), axis=1)
                                feat_z_sensitive = train_feat_z[idxs[target_sensitive]]
                                weights_sensitive = weights[idxs[target_sensitive]]
                                if weights_sensitive.sum() > 0:
                                    unfair_feat_z_sen = np.sum(
                                        feat_z_sensitive * weights_sensitive, 0) / weights_sensitive.sum()
                                    mean_diffs.append(
                                        np.mean(np.power(unfair_feat_z_sen - feat_z_weighted, 2)))

                    mean_diffs = np.mean(mean_diffs)

                    logger.logging(
                        "********************train********************")
                    logger.logging(
                        ' & '.join([
                            str(round(m, 4)) for m in [
                                acc, ha, ebf1, maf1, mif1, mean_diffs]]))
                    best_val_metrics['fair_mean_diff'] = mean_diffs
                else:
                    logger.logging(
                        "********************train********************")
                    logger.logging(
                        ' & '.join(
                            [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1]]))

            train_best_metrics = best_val_metrics
        else:
            train_best_metrics = None

        if eval_valid:
            valid_nll_loss = 0
            valid_c_loss = 0
            valid_total_loss = 0

            valid_indiv_prob = []
            valid_label = []

            if eval_fairness:
                valid_feat_z = []
                valid_sensitive = data.sensitive_feat[data.valid_idx]
            with tqdm(
                    range(int(len(data.valid_idx) / float(data.batch_size)) + 1),
                    desc='Evaluate on validation set') as t:

                for i in t:
                    start = i * data.batch_size
                    end = min(data.batch_size * (i + 1), len(data.valid_idx))
                    idx = data.valid_idx[start:end]

                    input_feat = torch.from_numpy(
                        data.input_feat[idx]).to(args.device)

                    input_label = torch.from_numpy(data.labels[idx])
                    input_label = deepcopy(input_label).float().to(args.device)

                    label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                        input_label, input_feat)
                    total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
                        input_label, label_out, label_mu, label_logvar, feat_out, feat_mu,
                        feat_logvar, model.r_sqrt_sigma, args)

                    valid_nll_loss += nll_loss.item() * (end - start)
                    valid_c_loss += c_loss.item() * (end - start)
                    valid_total_loss += total_loss.item() * (end - start)

                    for j in deepcopy(indiv_prob).cpu().data.numpy():
                        valid_indiv_prob.append(j)
                    for j in deepcopy(input_label).cpu().data.numpy():
                        valid_label.append(j)

                    if eval_fairness:
                        feat_z = model.feat_reparameterize(
                            feat_mu, feat_logvar)
                        valid_feat_z.append(feat_z.cpu().data.numpy())

                valid_indiv_prob = np.array(valid_indiv_prob)
                valid_label = np.array(valid_label)

                nll_loss = valid_nll_loss / len(data.valid_idx)
                c_loss = valid_c_loss / len(data.valid_idx)
                total_loss = valid_total_loss / len(data.valid_idx)

                best_val_metrics = None
                for threshold in THRESHOLDS:
                    val_metrics = evals.compute_metrics(
                        valid_indiv_prob, valid_label, threshold, all_metrics=True)

                    if best_val_metrics is None:
                        best_val_metrics = {}
                        for metric in METRICS:
                            best_val_metrics[metric] = val_metrics[metric]
                    else:
                        for metric in METRICS:
                            if 'FDR' in metric:
                                best_val_metrics[metric] = min(
                                    best_val_metrics[metric], val_metrics[metric])
                            else:
                                best_val_metrics[metric] = max(
                                    best_val_metrics[metric], val_metrics[metric])

                acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], \
                    best_val_metrics['ebF1'], best_val_metrics['maF1'], \
                    best_val_metrics['miF1']

                if eval_fairness:
                    valid_feat_z = np.concatenate(valid_feat_z)
                    assert valid_feat_z.shape[0] == len(data.valid_idx) and \
                        valid_feat_z.shape[1] == args.latent_dim

                    sensitive_centroid = np.unique(valid_sensitive, axis=0)
                    idxs = np.arange(len(data.valid_idx))

                    mean_diffs = []
                    for target_fair_label in target_fair_labels:
                        target_label_dist = label_distances[target_fair_label]
                        weights = []
                        for label in data.labels[idxs]:
                            label = label.astype(int)
                            distance = target_label_dist.get(
                                ''.join(label.astype(str)), 0.)
                            weights.append(distance)
                        weights = np.array(weights).reshape(-1, 1)

                        if weights.sum() > 0:
                            feat_z_weighted = np.sum(
                                valid_feat_z * weights, axis=0) / weights.sum()

                            for sensitive in sensitive_centroid:
                                target_sensitive = np.all(
                                    np.equal(valid_sensitive, sensitive), axis=1)
                                feat_z_sensitive = valid_feat_z[idxs[target_sensitive]]
                                weights_sensitive = weights[idxs[target_sensitive]]
                                if weights_sensitive.sum() > 0:
                                    unfair_feat_z_sen = np.sum(
                                        feat_z_sensitive * weights_sensitive, 0) / weights_sensitive.sum()
                                    mean_diffs.append(
                                        np.mean(np.power(unfair_feat_z_sen - feat_z_weighted, 2)))

                    mean_diffs = np.mean(mean_diffs)

                    logger.logging(
                        "********************valid********************")
                    logger.logging(
                        ' & '.join(
                            [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1, mean_diffs]]))
                    best_val_metrics['fair_mean_diff'] = mean_diffs
                else:
                    logger.logging(
                        "********************valid********************")
                    logger.logging(
                        ' & '.join(
                            [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1]]))

            valid_best_metrics = best_val_metrics
        else:
            valid_best_metrics = None

    return train_best_metrics, valid_best_metrics


def evaluate_over_labels(target_fair_labels, args, logger=Logger()):

    np.random.seed(4)
    nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data(
        args.dataset, args.mode, True, 'onehot')
    # if args.mask_target_label:
    #     nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data_masked(
    #         args.dataset, args.mode, True, 'onehot')
    # else:
    #     nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx = load_data(
    #         args.dataset, args.mode, True, 'onehot')

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx,
        valid_idx=valid_idx, batch_size=args.batch_size, label_clusters=None,
        sensitive_feat=sensitive_feat)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    label_dist_metric_paths = []
    for label_dist_metric in [meth for meth in IMPLEMENTED_METHODS if meth != 'unfair']:
        label_dist_metric = label_dist_metric + f'_{args.target_label_idx}'
        if args.mask_target_label:
            label_dist_metric += '_masked'

        label_dist_files = search_files(
            os.path.join(args.model_dir, label_dist_metric), postfix='.npy')

        if len(label_dist_files):
            label_dist_metric_paths += [os.path.join(
                args.model_dir, label_dist_metric, label_dist_file) for
                label_dist_file in label_dist_files]
    logger.logging('\n' * 5)
    logger.logging(f"""Fairness definitions: {label_dist_metric_paths}""")

    model_paths = []
    for model_prior in IMPLEMENTED_METHODS:
        if model_prior != 'unfair':
            model_prior += f'_{args.target_label_idx}'
            if args.mask_target_label:
                model_prior += '_masked'
        model_files = search_files(os.path.join(
            args.model_dir,  model_prior), postfix='.pkl')
        if len(model_files):
            model_paths += [os.path.join(
                args.model_dir, model_prior, model_file) for
                model_file in model_files]
    logger.logging('\n' * 5)
    logger.logging(f"""Fair Models: {model_paths}""")

    fair_results = {}
    perform_result = {}
    for dist_metric in label_dist_metric_paths:
        logger.logging(f'Evaluate fairness definition: {dist_metric}...')
        logger.logging('\n' * 3)
        label_dist = pickle.load(open(dist_metric, 'rb'))

        dist_metric = dist_metric.replace(
            '.npy', '').split('/')[-1].split('-')[1:]
        dist_metric = '-'.join(dist_metric)
        fair_results[dist_metric] = {}
        met_perform = []
        for model_stat in model_paths:
            print(f'Fair model: {model_stat}')
            model = VAE(args).to(args.device)
            model.load_state_dict(torch.load(model_stat))
            train, valid = evaluate_mpvae(
                model, data, target_fair_labels, label_dist, args, logger=logger)

            model_trained = model_stat.replace(
                '.pkl', '').split('/')[-1]
            if 'unfair' in model_trained:
                model_trained = 'unfair'
            else:
                model_trained = '-'.join(model_trained.split('-')[1:])

            fair_results[dist_metric][
                model_trained] = f"{round(train['fair_mean_diff'], 5)}~({round(valid['fair_mean_diff'], 5)})"

            if model_trained not in perform_result:
                perform_result[model_trained] = []
            perform_result[model_trained].append(
                [train[args.perform_metric], valid[args.perform_metric]])

    for model_trained in perform_result:
        met_perform = np.mean(perform_result[model_trained], axis=0)
        perform_result[model_trained] = f"{round(met_perform[0], 5)}~({round(met_perform[1], 5)})"

    return fair_results, perform_result


def retrieve_nearest_neighbor_labels(target_label, num_neighbor, label_distances):
    target_label = ''.join(target_label.astype(str))
    if target_label not in label_distances:
        return []
    label_dist = label_distances[target_label]
    label_dist = sorted(
        label_dist.items(), reverse=True, key=lambda item: item[1])
    neighbors = [item[0]
                 for item in label_dist[:num_neighbor + 1] if item[1] < 1.]
    assert target_label not in neighbors and len(neighbors) == num_neighbor

    return neighbors


def evaluate_nearest_neighbor_labels(args, logger=Logger()):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(args.dataset, args.mode, True)
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_label = label_type[idx].astype(int)

    #  TODO: use args to handle
    label_dist_files = search_files(
        os.path.join(args.model_dir, 'arule'), postfix='.npy')

    if len(label_dist_files):
        label_dist_file = label_dist_files[0]
        label_dist = pickle.load(open(os.path.join(
            args.model_dir, 'arule', label_dist_file), 'rb'))
        target_fair_labels = retrieve_nearest_neighbor_labels(
            target_fair_label, 5, label_dist)
        if target_fair_labels == []:
            logger.logging(f'Fail to retrieve nearest neighbors...')
        else:
            logger.logging(
                f'Evaluate on nearest neibors: {target_fair_labels}')
            evaluate_over_labels(target_fair_labels, args, logger)


def evaluate_target_labels(args, logger=Logger()):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(args.dataset, args.mode, True)
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_labels = label_type[idx: idx + 1].astype(int)

    return evaluate_over_labels(target_fair_labels, args, logger)


if __name__ == '__main__':
    from faircluster_train import parser

    parser.add_argument('-target_label_idx', type=int, default=0)
    parser.add_argument('-perform_metric', type=str, default='HA',
                        choices=['ACC', 'HA', 'ebF1', 'maF1', 'miF1'])
    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    args.model_dir = f'fair_through_distance/model/{args.dataset}'
    logger = Logger(os.path.join(
        args.model_dir, f'evalution-{args.target_label_idx}.txt'))
    evaluate_target_labels(args, logger)

    logger = Logger(os.path.join(
        args.model_dir, f'evalution-{args.target_label_idx}-nn.txt'))
    # evaluate_nearest_neighbor_labels(args, logger)

# python fairsoft_evaluate.py -dataset adult -latent_dim 8 -cuda 6 -target_label_idx 0
