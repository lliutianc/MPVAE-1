import sys
import os
import datetime
from copy import deepcopy
import types

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import evals
from utils import build_path
from model import VAE, compute_loss
from data import load_data, preprocess
from main import THRESHOLDS, METRICS
from fairreg import parser

parser.add_argument('-min_support', type=float, default=None)
parser.add_argument('-min_confidence', type=float, default=None)

sys.path.append('./')


def apriori_pair_dist(labelset1, labelset2, apriori_rules):
    if labelset1 == labelset2:
        return 0

    samelabels = labelset1.intersection(labelset2)
    diff1 = labelset1.difference(samelabels)
    diff2 = labelset2.difference(samelabels)

    if len(samelabels) == 0:
        return np.inf

    score = np.inf
    candidates = apriori_rules.loc[apriori_rules['antecedents'] == frozenset(
        samelabels)]
    if len(candidates):
        score1 = candidates.loc[
            candidates['consequents'] == frozenset(diff1), 'confidence']
        score2 = candidates.loc[
            candidates['consequents'] == frozenset(diff2), 'confidence']

        if len(score1) and len(score2):
            score = np.abs(score1.to_numpy() - score2.to_numpy()).item()

    return score


def apriori_distance(args):
    np.random.seed(4)

    _, _, labels = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot')
    print(labels_oh.shape)

    labels = labels.astype(str)

    encoder = TransactionEncoder()
    labels_df = encoder.fit_transform(labels)
    cols = encoder.columns_
    cols[0] = 'high_income'
    labels_df = pd.DataFrame(labels_df, columns=cols)

    min_support = args.min_support or 1 / len(labels_df)
    min_confidence = args.min_confidence or 0.
    labels_apri = apriori(
        labels_df, min_support=min_support, use_colnames=True, verbose=1)
    labels_rules = association_rules(labels_apri, min_threshold=min_confidence)

    income_level = np.unique(labels[:, 0])
    occupation = np.unique(labels[:, 1])
    workclass = np.unique(labels[:, 2])
    labelsets = []
    for income in income_level:
        for occu in occupation[1:]:
            for work in workclass[1:]:
                labelsets.append(set([income, occu, work]))

    labels_oh_str = np.concatenate([labels_oh.astype(str), labels], axis=1)
    labels_oh_str = np.unique(labels_oh_str, axis=0)
    labels_express = {}
    for label in labels_oh_str:
        label_oh = label[:-3]
        label_str = label[-3:]
        print(len(label), len(label_oh), len(label_str))
        labels_express[frozenset(label_str)] = label_oh.astype(int)

    dist_dict = {}
    for p1 in labelsets:
        p1_oh = labels_express.get(frozenset(p1), None)
        if p1_oh is not None:
            lab1 = ''.join(p1_oh.astype(str))
            dist_dict[lab1] = {}
            for p2 in labelsets:
                p2_oh = labels_express.get(frozenset(p2), None)
                if p2_oh is not None:
                    lab2 = ''.join(p2_oh.astype(str))
                    dist_dict[lab1][lab2] = apriori_pair_dist(
                        p1, p2, labels_rules)

    return dist_dict


def train_mpvae_softfair_one_epoch(
        data, model, optimizer, scheduler, penalize_unfair, target_fair_labels, label_distances, eval_after_one_epoch, args):

    if penalize_unfair and target_fair_labels is None:
        target_fair_labels = list(label_distances.keys())
        raise NotImplementedError('Have not supported smooth-OD yet.')

    target_fair_labels_str = []
    for target_fair_label in target_fair_labels:
        target_fair_label = ''.join(target_fair_label.astype(str))
        target_fair_labels_str.append(target_fair_label)
    target_fair_labels = target_fair_labels_str

    np.random.shuffle(data.train_idx)
    args.device = next(model.parameters()).device

    smooth_nll_loss = 0.0  # label encoder decoder cross entropy loss
    smooth_nll_loss_x = 0.0  # feature encoder decoder cross entropy loss
    smooth_c_loss = 0.0  # label encoder decoder ranking loss
    smooth_c_loss_x = 0.0  # feature encoder decoder ranking loss
    smooth_kl_loss = 0.0  # kl divergence
    smooth_total_loss = 0.0  # total loss
    smooth_macro_f1 = 0.0  # macro_f1 score
    smooth_micro_f1 = 0.0  # micro_f1 score
    smooth_reg_fair = 0.

    temp_label = []
    temp_indiv_prob = []

    with tqdm(range(int(len(data.train_idx) / float(data.batch_size)) + 1), desc='Train VAE') as t:
        for i in t:
            optimizer.zero_grad()
            start = i * data.batch_size
            end = min(data.batch_size * (i + 1), len(data.train_idx))
            idx = data.train_idx[start:end]

            input_feat = torch.from_numpy(
                data.input_feat[idx]).float().to(args.device)

            input_label = torch.from_numpy(
                data.labels[idx]).float().to(args.device)
            label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                input_label, input_feat)

            if args.residue_sigma == "random":
                r_sqrt_sigma = torch.from_numpy(
                    np.random.uniform(
                        -np.sqrt(6.0 / (args.label_dim + args.z_dim)),
                        np.sqrt(6.0 / (args.label_dim + args.z_dim)),
                        (args.label_dim, args.z_dim))).to(
                    args.device)
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    r_sqrt_sigma, args)
            else:
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    model.r_sqrt_sigma, args)

            if penalize_unfair:
                label_z = model.label_reparameterize(label_mu, label_logvar)
                feat_z = model.feat_reparameterize(feat_mu, feat_logvar)

                sensitive_feat = torch.from_numpy(
                    data.sensitive_feat[idx]).to(args.device)
                sensitive_centroids = torch.unique(sensitive_feat, dim=0)
                idx_tensor = torch.arange(sensitive_feat.shape[0])

                reg_label_z_unfair = 0.
                reg_feat_z_unfair = 0.

                for target_fair_label in target_fair_labels:
                    target_label_dist = label_distances[target_fair_label]
                    batch_distance = []
                    for label in data.labels[idx]:
                        distance = target_label_dist.get(
                            ''.join(label.astype(str)), 0.)
                        batch_distance.append(distance)
                    batch_distance = torch.tensor(
                        batch_distance).to(args.device)
                    gamma = 1.
                    weights = torch.exp(-batch_distance * gamma)
                    weights = torch.clamp(weights, min=1e-6)

                    label_z_weighted = torch.sum(
                        label_z * weights, axis=0) / weights.sum()
                    feat_z_weighted = torch.sum(
                        feat_z * weights, axis=0) / weights.sum()

                    for sensitive in sensitive_centroids:
                        target_sensitive = torch.all(
                            torch.eq(sensitive_feat, sensitive), dim=1)
                        label_z_sensitive = label_z[idx_tensor[target_sensitive]]
                        feat_z_sensitive = feat_z[idx_tensor[target_sensitive]]
                        weight_sensitive = weights[idx_tensor[target_sensitive]]

                        reg_label_z_sen = torch.sum(
                            label_z_sensitive * weight_sensitive, 0) / weight_sensitive.sum()
                        reg_feat_z_sen = torch.sum(
                            feat_z_sensitive * weight_sensitive, 0) / weight_sensitive.sum()
                        reg_label_z_unfair += torch.mean(
                            torch.pow(reg_label_z_sen - label_z_weighted, 2))
                        reg_feat_z_unfair += torch.mean(
                            torch.pow(reg_feat_z_sen - feat_z_weighted, 2))

                fairloss = args.label_z_fair_coeff * reg_label_z_unfair + \
                    args.feat_z_fair_coeff * reg_feat_z_unfair

                if isinstance(fairloss, float):
                    raise UserWarning('Fail to construct fairness regualizers')
                else:
                    total_loss += fairloss
                    smooth_reg_fair += fairloss.item()

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if scheduler:
                scheduler.step()

            # evaluation
            train_metrics = evals.compute_metrics(
                indiv_prob.cpu().data.numpy(), input_label.cpu().data.numpy(), 0.5,
                all_metrics=False)
            macro_f1, micro_f1 = train_metrics['maF1'], train_metrics['miF1']

            smooth_nll_loss += nll_loss.item()
            smooth_nll_loss_x += nll_loss_x.item()
            # smooth_l2_loss += l2_loss
            smooth_c_loss += c_loss.item()
            smooth_c_loss_x += c_loss_x.item()
            smooth_kl_loss += kl_loss.item()
            smooth_total_loss += total_loss.item()
            smooth_macro_f1 += macro_f1.item()
            smooth_micro_f1 += micro_f1.item()

            # log the labels

            running_postfix = {'total_loss': smooth_total_loss / float(i + 1),
                               'nll_loss_label': smooth_nll_loss / float(i + 1),
                               'nll_loss_feat': smooth_nll_loss_x / float(i + 1),
                               }
            if penalize_unfair:
                running_postfix['fair_loss'] = smooth_reg_fair / float(i + 1)
            t.set_postfix(running_postfix)

    if eval_after_one_epoch:
        nll_loss = smooth_nll_loss / float(i + 1)
        nll_loss_x = smooth_nll_loss_x / float(i + 1)
        c_loss = smooth_c_loss / float(i + 1)
        c_loss_x = smooth_c_loss_x / float(i + 1)
        kl_loss = smooth_kl_loss / float(i + 1)
        total_loss = smooth_total_loss / float(i + 1)
        macro_f1 = smooth_macro_f1 / float(i + 1)
        micro_f1 = smooth_micro_f1 / float(i + 1)

        # temp_indiv_prob = np.array(temp_indiv_prob).reshape(-1)
        # temp_label = np.array(temp_label).reshape(-1)

        time_str = datetime.datetime.now().isoformat()
        print(
            "macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\ntotal_loss=%.6f\n" % (
                macro_f1, micro_f1, nll_loss * args.nll_coeff, nll_loss_x * args.nll_coeff,
                c_loss * args.c_coeff, c_loss_x * args.c_coeff, kl_loss, total_loss))

        current_loss, val_metrics = validate_mpvae(
            model, data.input_feat, data.labels, data.valid_idx, args)


def validate_mpvae(model, feat, labels, valid_idx, args):
    args.device = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        print("performing validation...")

        all_nll_loss = 0
        all_l2_loss = 0
        all_c_loss = 0
        all_total_loss = 0

        all_indiv_prob = []
        all_label = []

        real_batch_size = min(args.batch_size, len(valid_idx))
        with tqdm(range(int((len(valid_idx) - 1) / real_batch_size) + 1), desc='Validate VAE') as t:
            for i in t:
                start = real_batch_size * i
                end = min(real_batch_size * (i + 1), len(valid_idx))
                input_feat = feat[valid_idx[start:end]]
                input_label = labels[valid_idx[start:end]]

                input_feat = torch.from_numpy(
                    input_feat).float().to(args.device)
                input_label = torch.from_numpy(
                    input_label).float().to(args.device)

                label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                    input_label, input_feat)
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    model.r_sqrt_sigma, args)

                all_nll_loss += nll_loss * (end - start)
                # all_l2_loss += l2_loss*(end-start)
                all_c_loss += c_loss * (end - start)
                all_total_loss += total_loss * (end - start)

                for j in deepcopy(indiv_prob).cpu().data.numpy():
                    all_indiv_prob.append(j)
                for j in deepcopy(input_label).cpu().data.numpy():
                    all_label.append(j)

        # collect all predictions and ground-truths
        all_indiv_prob = np.array(all_indiv_prob)
        all_label = np.array(all_label)

        nll_loss = all_nll_loss / len(valid_idx)
        l2_loss = all_l2_loss / len(valid_idx)
        c_loss = all_c_loss / len(valid_idx)
        total_loss = all_total_loss / len(valid_idx)

        best_val_metrics = None
        for threshold in THRESHOLDS:
            val_metrics = evals.compute_metrics(
                all_indiv_prob, all_label, threshold, all_metrics=True)

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

        time_str = datetime.datetime.now().isoformat()
        acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], best_val_metrics[
            'ebF1'], best_val_metrics['maF1'], best_val_metrics['miF1']

        # nll_coeff: BCE coeff, lambda_1
        # c_coeff: Ranking loss coeff, lambda_2
        print("**********************************************")
        print(
            "valid results: %s\nacc=%.6f\tha=%.6f\texam_f1=%.6f, macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tc_loss=%.6f\ttotal_loss=%.6f" % (
                time_str, acc, ha, ebf1, maf1, mif1, nll_loss *
                args.nll_coeff, c_loss * args.c_coeff,
                total_loss))
        print("**********************************************")

    model.train()

    return nll_loss, best_val_metrics


def train_fair_through_regularize():

    hparams = f'min_support={args.min_support}-'\
              f'min_confidence={args.min_confidence}'
    label_dist_path = os.path.join(
        args.model_dir, f'label_dist-{hparams}.npy')

    if args.resume and os.path.exists(label_dist_path):
        label_dist = np.load(open(label_dist_path, 'rb'))
    else:
        label_dist = apriori_distance(args)
        np.save(open(label_dist_path, 'wb'), label_dist)

    np.random.seed(4)
    nonsensitive_feat, sensitive_feat, labels = load_data(
        args.dataset, args.mode, True, 'onehot')

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

    # test fairness on some labels
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    print(labels.shape)
    label_type = label_type[count_sort_idx]
    target_fair_labels = label_type[:1].astype(int)
    print(list(label_dist.keys()))
    print(label_type.shape)
    print(target_fair_labels)
    exit(1)
    print('start training fair mpvae...')
    for _ in range(args.max_epoch):
        train_mpvae_softfair_one_epoch(
            data, fair_vae, optimizer, scheduler,
            penalize_unfair=True, target_fair_labels=target_fair_labels, label_distances=label_dist,
            eval_after_one_epoch=True, args=args)
    torch.save(fair_vae.cpu().state_dict(), fair_vae_checkpoint_path)


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    args.labels_cluster_method = 'apriori'

    if args.labels_cluster_num:
        args.labels_cluster_distance_threshold = None

    param_setting = f"arule"
    args.model_dir = f"fair_through_distance/model/{args.dataset}/{param_setting}"
    args.summary_dir = f"fair_through_distance/summary/{args.dataset}/{param_setting}"
    build_path(args.model_dir, args.summary_dir)

    train_fair_through_regularize()
