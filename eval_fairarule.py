import pickle

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from copy import deepcopy
import os
import sys
import types

import evals
from model import VAE, compute_loss
from data import load_data
from fairreg import THRESHOLDS, METRICS
from fairarule import parser
from label_cluster import construct_label_clusters

# fair regularizer
# parser.add_argument('-min_support', type=float, default=None)
# parser.add_argument('-min_confidence', type=float, default=None)

parser.add_argument('-fairness_strate_embed', type=str, default='', choices=[
                    'mpvae', 'cbow', 'none', ''])
parser.add_argument('-fairness_strate_cluster', type=str, default='kmeans', choices=[
                    'kmeans', 'kmodes', 'apriori', 'kprototype', 'hierarchical'])

sys.path.append('./')


def evaluate_mpvae(model, data, eval_fairness=True, eval_train=True, eval_valid=True):
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
                    mean_diffs = []
                    idxs = np.arange(len(data.train_idx))

                    sensitive_centroid = np.unique(train_sensitive, axis=0)
                    for label_centroid in np.unique(data.label_clusters[idxs]):
                        target_centroid = np.equal(
                            data.label_clusters[idxs], label_centroid)

                        cluster_feat_z = train_feat_z[idxs[target_centroid]]
                        if len(cluster_feat_z):
                            for sensitive in sensitive_centroid:
                                target_sensitive = np.all(
                                    np.equal(train_sensitive, sensitive), axis=1)
                                cluster_sensitve = np.all(
                                    np.stack((target_sensitive, target_centroid), axis=1), axis=1
                                )
                                cluster_feat_z_sensitive = train_feat_z[idxs[cluster_sensitve]]
                                if len(cluster_feat_z_sensitive):
                                    mean_diffs.append(np.mean(
                                        np.power(cluster_feat_z_sensitive.mean(0) - cluster_feat_z.mean(0), 2)))

                    best_val_metrics['fair'] = np.mean(mean_diffs)

                    # nll_coeff: BCE coeff, lambda_1
                    # c_coeff: Ranking loss coeff, lambda_2
                    print("********************train********************")
                    print(round(best_val_metrics['fair'], 4))
                    # print(
                    #     ' & '.join(
                    #         [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1, mean_diffs]]))
                else:
                    # nll_coeff: BCE coeff, lambda_1
                    # c_coeff: Ranking loss coeff, lambda_2
                    print("********************train********************")
                    # print(
                    #     ' & '.join(
                    #         [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1]]))

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
                    mean_diffs = []
                    idxs = np.arange(len(data.valid_idx))

                    sensitive_centroid = np.unique(valid_sensitive, axis=0)
                    for label_centroid in np.unique(data.label_clusters[idxs]):
                        target_centroid = np.equal(
                            data.label_clusters[idxs], label_centroid)

                        cluster_feat_z = valid_feat_z[idxs[target_centroid]]
                        if len(cluster_feat_z):
                            for sensitive in sensitive_centroid:
                                target_sensitive = np.all(
                                    np.equal(valid_sensitive, sensitive), axis=1)
                                cluster_sensitve = np.all(
                                    np.stack((target_sensitive, target_centroid), axis=1), axis=1
                                )
                                cluster_feat_z_sensitive = valid_feat_z[idxs[cluster_sensitve]]
                                if len(cluster_feat_z_sensitive):
                                    mean_diffs.append(np.mean(
                                        np.power(cluster_feat_z_sensitive.mean(0) - cluster_feat_z.mean(0), 2)))

                    best_val_metrics['fair'] = np.mean(mean_diffs)

                    # nll_coeff: BCE coeff, lambda_1
                    # c_coeff: Ranking loss coeff, lambda_2
                    print("********************valid********************")
                    print(round(best_val_metrics['fair'], 4))
                    # print(
                    #     ' & '.join(
                    #         [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1, mean_diffs]]))
                else:
                    # nll_coeff: BCE coeff, lambda_1
                    # c_coeff: Ranking loss coeff, lambda_2
                    print("********************valid********************")
                    # print(
                    #     ' & '.join(
                    #         [str(round(m, 4)) for m in [acc, ha, ebf1, maf1, mif1]]))

            valid_best_metrics = best_val_metrics
        else:
            valid_best_metrics = None

    return train_best_metrics, valid_best_metrics


if __name__ == '__main__':

    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    if args.labels_cluster_num:
        args.labels_cluster_distance_threshold = None

    for min_support in [0.001, 0.005, 0.01, 0.05]:
        param_setting = f"n_cluster={args.labels_cluster_num}-"\
                        f"cluster_distance_thre={args.labels_cluster_distance_threshold}"

        args.model_dir = f'fair_through_arule/model/{args.dataset}/{param_setting}'
        args.summary_dir = f'fair_through_arule/summary/{args.dataset}/{param_setting}'

        np.random.seed(4)
        # prepare label_clusters
        nonsensitive_feat, sensitive_feat, labels = load_data(
            args.dataset, args.mode, True)
        train_cnt, valid_cnt = int(
            len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
        train_idx = np.arange(train_cnt)
        valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)
        data = types.SimpleNamespace(
            input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx,
            valid_idx=valid_idx, batch_size=args.batch_size, label_clusters=None,
            sensitive_feat=sensitive_feat)
        args.feature_dim = data.input_feat.shape[1]
        args.label_dim = data.labels.shape[1]

        label_cluster_path = os.path.join(
            args.model_dir,
            f'label_cluster-'
            f'n_cluster={args.labels_cluster_num}-'
            f'dist_cluster={args.labels_cluster_distance_threshold}-'
            f'min_support={min_support}-'
            f'min_confidence={args.min_confidence}' + '.npy')

        if os.path.exists(label_cluster_path):
            label_clusters = np.load(open(label_cluster_path, 'rb'))
        else:
            label_clusters = construct_label_clusters(args)
            np.save(open(label_cluster_path, 'wb'), label_clusters)

        data.label_clusters = label_clusters
        print(
            f'min_support={min_support}, cluster numbers: {len(np.unique(label_clusters))}')

        for min_support_ in [0.001, 0.005, 0.01, 0.05]:
            model_file = os.path.join(
                args.model_dir,
                f'fair_vae_prior_label_cluster-'
                f'min_support={min_support_}-'
                f'min_confidence={args.min_confidence}' + '.pkl')
            print(f'try loading model from: {model_file}')

            if os.path.exists(model_file):
                model = VAE(args).to(args.device)
                model.load_state_dict(torch.load(model_file))
                print(f'start evaluating {model_file}...')
                train, valid = evaluate_mpvae(model, data, True)

    for min_confidence in [0.01, 0.05, 0.1, 0.25]:
        param_setting = f"n_cluster={args.labels_cluster_num}-"\
                        f"cluster_distance_thre={args.labels_cluster_distance_threshold}"

        args.model_dir = f'fair_through_arule/model/{args.dataset}/{param_setting}'
        args.summary_dir = f'fair_through_arule/summary/{args.dataset}/{param_setting}'

        np.random.seed(4)
        # prepare label_clusters
        nonsensitive_feat, sensitive_feat, labels = load_data(
            args.dataset, args.mode, True)
        train_cnt, valid_cnt = int(
            len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
        train_idx = np.arange(train_cnt)
        valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)
        data = types.SimpleNamespace(
            input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx,
            valid_idx=valid_idx, batch_size=args.batch_size, label_clusters=None,
            sensitive_feat=sensitive_feat)
        args.feature_dim = data.input_feat.shape[1]
        args.label_dim = data.labels.shape[1]

        label_cluster_path = os.path.join(
            args.model_dir,
            f'label_cluster-'
            f'min_support={args.min_support}-'
            f'min_confidence={min_confidence}' + '.npy')

        if os.path.exists(label_cluster_path):
            label_clusters = np.load(open(label_cluster_path, 'rb'))
        else:
            label_clusters = construct_label_clusters(args)
            np.save(open(label_cluster_path, 'wb'), label_clusters)

        data.label_clusters = label_clusters
        print(
            f'min_support={min_support}, cluster numbers: {len(np.unique(label_clusters))}')

        for min_confidence_ in [0.001, 0.005, 0.01, 0.05]:
            model_file = os.path.join(
                args.model_dir,
                f'fair_vae_prior_label_cluster-'
                f'min_support={args.min_support}-'
                f'min_confidence={min_confidence_}' + '.pkl')
            print(f'try loading model from: {model_file}')

            if os.path.exists(model_file):
                model = VAE(args).to(args.device)
                model.load_state_dict(torch.load(model_file))
                print(f'start evaluating {model_file}...')
                train, valid = evaluate_mpvae(model, data, True)


# python eval_fairarule.py - dataset adult - latent_dim 8 - fairness_strate_embed none - labels_embed_method none - cuda 6 - labels_cluster_num 8
