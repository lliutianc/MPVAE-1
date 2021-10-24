import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np

import sys
import os
import datetime
from copy import deepcopy
import types
from tqdm import tqdm

import evals
from utils import build_path, get_label, get_feat
from model import VAE, compute_loss
from fairmodel import FairCritic, compute_fair_loss
from data import load_data
from embed import CBOW, CBOWData

from main import parser, THRESHOLDS, METRICS

# cluster parameters
parser.add_argument('-labels_embed_method', type=str, choices=['cbow', 'mpvae', 'mlp'])
parser.add_argument('-labels_cluster_distance_threshold', type=float, default=.1)
parser.add_argument('-labels_cluster_min_size', type=int, default=4)
# fair regularizer
parser.add_argument('-label_z_fair_coeff', type=float, default=1.0)
parser.add_argument('-feat_z_fair_coeff', type=float, default=1.0)

parser.add_argument('-cuda', type=int, default=0)

sys.path.append('./')


def construct_labels_embed(data):
    one_epoch_iter = np.ceil(len(data.train_idx) / args.batch_size)

    if args.labels_embed_method == 'mpvae':
        # train a prior mpvae
        prior_vae = VAE(args).to(device)
        prior_vae.train()

        optimizer = optim.Adam(prior_vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        prior_vae_checkpoint_path = os.path.join(model_dir, f'prior_vae')
        if args.resume and os.path.exists(prior_vae_checkpoint_path):
            print('load trained prior mpvae...')
            prior_vae.load_state_dict(torch.load(prior_vae_checkpoint_path))
        else:
            print('train a new prior mpvae...')
            for _ in range(args.max_epoch // 3):
                train_mpvae_one_epoch(
                    data, prior_vae, optimizer, scheduler, penalize_unfair=False)
            torch.save(prior_vae.cpu().state_dict(), prior_vae_checkpoint_path)

        prior_vae = prior_vae.to(device)
        with torch.no_grad():
            prior_vae.eval()
            idxs = np.arange(int(len(data.input_feat) * .9))
            labels_mu, labels_logvar = [], []
            for i in range(int(len(idxs) / float(data.batch_size)) + 1):
                start = i * data.batch_size
                end = min(data.batch_size * (i + 1), len(idxs))

                input_feat = data.input_feat[idxs[start:end]]
                input_feat = torch.from_numpy(input_feat).to(device)
                input_label = data.labels[idxs[start:end]]
                input_label = torch.from_numpy(input_label).to(device)
                label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = prior_vae(
                    input_label, input_feat)
                labels_mu.append(label_mu.cpu().data.numpy())
                labels_logvar.append(label_logvar.cpu().data.numpy())

        labels_mu = np.concatenate(labels_mu)
        labels_logvar = np.concatenate(labels_logvar)
        labels_embed = labels_mu

    if args.labels_embed_method == 'cbow':
        cbow_data = CBOWData(data.labels, device)
        print(len(cbow_data))
        print(cbow_data[0])
        # cbow_dataloader = DataLoader(cbow_data, batch_size=data.batch_size)
        prior_cbow = CBOW(data.labels.shape[1], args.latent_dim, 64).to(device)
        prior_cbow.train()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(prior_cbow.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        prior_cbow_checkpoint_path = os.path.join(model_dir, f'prior_cbow')
        if args.resume and os.path.exists(prior_cbow_checkpoint_path):
            print('load trained prior cbow...')
            prior_cbow.load_state_dict(torch.load(prior_cbow_checkpoint_path))
        else:
            print('train a new prior cbow...')
            for _ in range(args.max_epoch // 3):
                smooth_loss = 0.
                with tqdm(cbow_data, desc='Train CBOW') as t:
                    for i, (context, target) in enumerate(t):
                        optimizer.zero_grad()
                        logprob = prior_cbow(context)
                        loss = criterion(logprob, target)
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(prior_cbow.parameters(), 100)
                        optimizer.step()
                        scheduler.step()
                        smooth_loss += loss.item()
                        t.set_postfix({'running loss': smooth_loss / (i + 1)})
            torch.save(prior_cbow.cpu().state_dict(), prior_cbow_checkpoint_path)

        prior_cbow = prior_cbow.to(device)
        with torch.no_grad():
            prior_cbow.eval()
            labels_embed = []

            for idx in np.arange(int(len(data.input_feat) * .9)):
                input_label = data.labels[1]
                input_label = torch.from_numpy(input_label).to(device)

                idx = torch.arange(data.labels.shape[1],device=device)
                label_embed = prior_cbow.get_embedding(idx[input_label == 1])
                labels_embed.append(label_embed.cpu().data.numpy())

            labels_embed = np.vstack(labels_embed)

    return labels_embed


def hard_cluster(labels_embed):
    # todo: do we have soft clustering algorithm? For example, can we use gumble softmax?

    # todo: how to properly cluster labels based on JSD or KL average distance?
    #  Take KL for instance, afer merging two points, the new cluster is a Gaussian mixture,
    #  do we still have closed form formula to update new distance?

    from sklearn.cluster import AgglomerativeClustering
    distance_threshold = args.labels_cluster_distance_threshold
    succ_cluster = False
    for cluster_try in range(10):
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold).fit(labels_embed)
        labels_cluster = cluster.labels_
        _, counts = np.unique(labels_cluster, return_counts=True)
        if counts.min() < args.labels_cluster_min_size:
            distance_threshold *= 2
        else:
            succ_cluster = True
            break

    if succ_cluster is False:
        raise UserWarning('Labels clustering not converged')

    assert labels_cluster.shape[0] == labels_embed.shape[0], \
        f'{labels_embed.shape}, {labels_cluster.shape}'

    return labels_cluster


def train_mpvae_one_epoch(
    data, model, optimizer, scheduler, penalize_unfair, eval_after_one_epoch=True):

    np.random.shuffle(data.train_idx)

    smooth_nll_loss = 0.0  # label encoder decoder cross entropy loss
    smooth_nll_loss_x = 0.0  # feature encoder decoder cross entropy loss
    smooth_c_loss = 0.0  # label encoder decoder ranking loss
    smooth_c_loss_x = 0.0  # feature encoder decoder ranking loss
    smooth_kl_loss = 0.0  # kl divergence
    smooth_total_loss = 0.0  # total loss
    smooth_macro_f1 = 0.0  # macro_f1 score
    smooth_micro_f1 = 0.0  # micro_f1 score
    smooth_reg_fair = 0.
    # smooth_l2_loss = 0.0

    temp_label = []
    temp_indiv_prob = []

    with tqdm(range(int(len(data.train_idx) / float(data.batch_size)) + 1), desc='Train VAE') as t:
        for i in t:
            optimizer.zero_grad()
            start = i * data.batch_size
            end = min(data.batch_size * (i + 1), len(data.train_idx))
            idx = data.train_idx[start:end]

            input_feat = torch.from_numpy(data.input_feat[idx]).to(device)

            input_label = torch.from_numpy(data.labels[idx])
            input_label = deepcopy(input_label).float().to(device)

            label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                input_label, input_feat)

            if args.residue_sigma == "random":
                r_sqrt_sigma = torch.from_numpy(
                    np.random.uniform(
                        -np.sqrt(6.0 / (args.label_dim + args.z_dim)),
                        np.sqrt(6.0 / (args.label_dim + args.z_dim)),
                        (args.label_dim, args.z_dim))).to(
                    device)
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

                clusters = torch.from_numpy(data.label_clusters[idx]).to(device)
                sensitive_feat = torch.from_numpy(data.sensitive_feat[idx]).to(device)

                reg_labels_z_unfair = 0.
                reg_feats_z_unfair = 0.
                label_centroids = torch.unique(clusters)
                sensitive_centroids = torch.unique(sensitive_feat, dim=0)
                idx_tensor = torch.arange(clusters.shape[0])

                for label_centroid in torch.unique(clusters):
                    target_centroid = torch.eq(clusters, label_centroid)

                    # z_y penalty: E(z_y | cluster, a) = E( z_y | cluster)
                    cluster_label_z = label_z[idx_tensor[target_centroid]]
                    if len(cluster_label_z):
                        for sensitive in sensitive_centroids:
                            target_sensitive = torch.all(torch.eq(sensitive_feat, sensitive), dim=1)
                            sensitive_centroid = torch.all(
                                torch.stack((target_sensitive, target_centroid), dim=1), dim=1)
                            cluster_labels_z_sensitive = label_z[idx_tensor[sensitive_centroid]]
                            if len(cluster_labels_z_sensitive):
                                reg_labels_z_unfair += torch.mean(torch.pow(
                                    cluster_labels_z_sensitive.mean(0) - cluster_label_z.mean(0), 2))

                    # z_x penalty: E(z_x | cluster, a) = E( z_x | cluster)
                    cluster_feat_z = feat_z[idx_tensor[target_centroid]]
                    if len(cluster_feat_z):
                        for sensitive in sensitive_centroids:
                            target_sensitive = torch.all(torch.eq(sensitive_feat, sensitive), dim=1)
                            sensitive_centroid = torch.all(
                                torch.stack((target_sensitive, target_centroid), dim=1), dim=1)
                            cluster_feats_z_sensitive = feat_z[idx_tensor[sensitive_centroid]]
                            if len(cluster_feats_z_sensitive):
                                reg_feats_z_unfair += torch.mean(torch.pow(
                                    cluster_feats_z_sensitive.mean(0) - cluster_feat_z.mean(0), 2))

                fairloss = args.label_z_fair_coeff * reg_labels_z_unfair + \
                           args.feat_z_fair_coeff * reg_feats_z_unfair
                if isinstance(fairloss, float):
                    raise UserWarning('Fail to construct fairness regualizers')
                else:
                    total_loss += fairloss
                    smooth_reg_fair += fairloss.item()

            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100)
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
            temp_label.append(input_label.cpu().data.numpy())
            # log the individual prediction of the probability on each label
            temp_indiv_prob.append(indiv_prob.detach().data.cpu().numpy())

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

        temp_indiv_prob = np.array(temp_indiv_prob).reshape(-1)
        temp_label = np.array(temp_label).reshape(-1)

        time_str = datetime.datetime.now().isoformat()
        print(
            "macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\ntotal_loss=%.6f\n" % (
                macro_f1, micro_f1, nll_loss * args.nll_coeff, nll_loss_x * args.nll_coeff,
                c_loss * args.c_coeff, c_loss_x * args.c_coeff, kl_loss, total_loss))

        current_loss, val_metrics = validate_mpvae(
            model, data.input_feat, data.labels, data.valid_idx)


def validate_mpvae(model, feat, labels, valid_idx):
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
                input_feat, input_label = torch.from_numpy(input_feat).to(device), torch.from_numpy(
                    input_label)
                input_label = deepcopy(input_label).float().to(device)

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
            val_metrics = evals.compute_metrics(all_indiv_prob, all_label, threshold, all_metrics=True)

            if best_val_metrics is None:
                best_val_metrics = {}
                for metric in METRICS:
                    best_val_metrics[metric] = val_metrics[metric]
            else:
                for metric in METRICS:
                    if 'FDR' in metric:
                        best_val_metrics[metric] = min(best_val_metrics[metric], val_metrics[metric])
                    else:
                        best_val_metrics[metric] = max(best_val_metrics[metric], val_metrics[metric])

        time_str = datetime.datetime.now().isoformat()
        acc, ha, ebf1, maf1, mif1 = best_val_metrics['ACC'], best_val_metrics['HA'], best_val_metrics[
            'ebF1'], best_val_metrics['maF1'], best_val_metrics['miF1']

        # nll_coeff: BCE coeff, lambda_1
        # c_coeff: Ranking loss coeff, lambda_2
        print("**********************************************")
        print(
            "valid results: %s\nacc=%.6f\tha=%.6f\texam_f1=%.6f, macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tc_loss=%.6f\ttotal_loss=%.6f" % (
                time_str, acc, ha, ebf1, maf1, mif1, nll_loss * args.nll_coeff, c_loss * args.c_coeff,
                total_loss))
        print("**********************************************")

    model.train()

    return nll_loss, best_val_metrics


def train_fair_through_regularize():

    np.random.seed(4)
    input_feat, labels = load_data(args.dataset, args.mode)
    train_cnt, valid_cnt = int(len(input_feat) * 0.7), int(len(input_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)

    data = types.SimpleNamespace(
        input_feat=input_feat, labels=labels, train_idx=train_idx, valid_idx=valid_idx,
        batch_size=args.batch_size)
    args.feature_dim = input_feat.shape[1]
    args.label_dim = labels.shape[1]
    print(args.feature_dim, args.label_dim)

    one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)
    n_iter = one_epoch_iter * args.max_epoch

    labels_embed = construct_labels_embed(data)
    label_clusters = hard_cluster(labels_embed)
    # retrain a new mpvae + fair regularization
    np.random.seed(4)
    nonsensitive_feat, sensitive_feat, labels = load_data(args.dataset, args.mode, True)
    train_cnt, valid_cnt = int(len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx, valid_idx=valid_idx,
        batch_size=args.batch_size, label_clusters=label_clusters, sensitive_feat=sensitive_feat)
    args.feature_dim = nonsensitive_feat.shape[1]
    args.label_dim = labels.shape[1]

    fair_vae = VAE(args).to(device)
    fair_vae.train()

    fair_vae_checkpoint_path = os.path.join(model_dir, f'fair_vae_prior={args.labels_embed_method}')
    if args.resume and os.path.exists(fair_vae_checkpoint_path):
        print('use a trained fair mpvae...')
        fair_vae.load_state_dict(torch.load(fair_vae_checkpoint_path))
    else:
        print('train a new fair mpvae...')

    optimizer = optim.Adam(fair_vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_fair = optim.Adam(fair_vae.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

    print('start training fair mpvae...')
    for _ in range(args.max_epoch):
        train_mpvae_one_epoch(
            data, fair_vae, optimizer, scheduler, penalize_unfair=True)
        # regularzie_mpvae_unfair(data, fair_vae, optimizer_fair, args, use_valid=True)

    torch.save(fair_vae.cpu().state_dict(), fair_vae_checkpoint_path)


# def train_mpvae_one_epoch(data, model, optimizer, scheduler, args, eval_after_one_epoch=True):
#     np.random.shuffle(data.train_idx)
#
#     smooth_nll_loss = 0.0 # label encoder decoder cross entropy loss
#     smooth_nll_loss_x = 0.0 # feature encoder decoder cross entropy loss
#     smooth_c_loss = 0.0 # label encoder decoder ranking loss
#     smooth_c_loss_x = 0.0 # feature encoder decoder ranking loss
#     smooth_kl_loss = 0.0 # kl divergence
#     smooth_total_loss = 0.0 # total loss
#     smooth_macro_f1 = 0.0 # macro_f1 score
#     smooth_micro_f1 = 0.0 # micro_f1 score
#     #smooth_l2_loss = 0.0
#
#     temp_label = []
#     temp_indiv_prob = []
#
#     with tqdm(range(int(len(data.train_idx) / float(data.batch_size)) + 1), desc='VAE') as t:
#         for i in t:
#             optimizer.zero_grad()
#             start = i * data.batch_size
#             end = min(data.batch_size * (i + 1), len(data.train_idx))
#
#             input_feat = data.input_feat[data.train_idx[start:end]]
#             input_feat = torch.from_numpy(input_feat).to(device)
#
#             input_label = data.labels[data.train_idx[start:end]]
#             input_label = torch.from_numpy(input_label)
#             input_label = deepcopy(input_label).float().to(device)
#
#             label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
#                 input_label, input_feat)
#
#             if args.residue_sigma == "random":
#                 r_sqrt_sigma = torch.from_numpy(
#                     np.random.uniform(
#                         -np.sqrt(6.0 / (args.label_dim + args.z_dim)),
#                         np.sqrt(6.0 / (args.label_dim + args.z_dim)), (args.label_dim, args.z_dim))).to(
#                     device)
#                 total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
#                     input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
#                     r_sqrt_sigma, args)
#             else:
#                 total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob = compute_loss(
#                     input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
#                     model.r_sqrt_sigma, args)
#
#             total_loss.backward()
#             grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100)
#             optimizer.step()
#             if scheduler:
#                 scheduler.step()
#
#             # evaluation
#             train_metrics = evals.compute_metrics(
#                 indiv_prob.cpu().data.numpy(), input_label.cpu().data.numpy(), 0.5,
#                 all_metrics=False)
#             macro_f1, micro_f1 = train_metrics['maF1'], train_metrics['miF1']
#
#             smooth_nll_loss += nll_loss.item()
#             smooth_nll_loss_x += nll_loss_x.item()
#             # smooth_l2_loss += l2_loss
#             smooth_c_loss += c_loss.item()
#             smooth_c_loss_x += c_loss_x.item()
#             smooth_kl_loss += kl_loss.item()
#             smooth_total_loss += total_loss.item()
#             smooth_macro_f1 += macro_f1.item()
#             smooth_micro_f1 += micro_f1.item()
#
#             # log the labels
#             temp_label.append(input_label.cpu().data.numpy())
#             # log the individual prediction of the probability on each label
#             temp_indiv_prob.append(indiv_prob.detach().data.cpu().numpy())
#
#             t.set_postfix({'total_loss': smooth_total_loss / float(i+1),
#                            'nll_loss_label': smooth_nll_loss / float(i+1),
#                            'nll_loss_feat': smooth_nll_loss_x / float(i+1),
#                            })
#
#     if eval_after_one_epoch:
#
#         nll_loss = smooth_nll_loss / float(i+1)
#         nll_loss_x = smooth_nll_loss_x / float(i+1)
#         c_loss = smooth_c_loss / float(i+1)
#         c_loss_x = smooth_c_loss_x / float(i+1)
#         kl_loss = smooth_kl_loss / float(i+1)
#         total_loss = smooth_total_loss / float(i+1)
#         macro_f1 = smooth_macro_f1 / float(i+1)
#         micro_f1 = smooth_micro_f1 / float(i+1)
#
#         temp_indiv_prob = np.array(temp_indiv_prob).reshape(-1)
#         temp_label = np.array(temp_label).reshape(-1)
#
#         time_str = datetime.datetime.now().isoformat()
#         print(
#             "macro_f1=%.6f, micro_f1=%.6f\nnll_loss=%.6f\tnll_loss_x=%.6f\nc_loss=%.6f\tc_loss_x=%.6f\tkl_loss=%.6f\ntotal_loss=%.6f\n" % (
#             macro_f1, micro_f1, nll_loss * args.nll_coeff,
#             nll_loss_x * args.nll_coeff, c_loss * args.c_coeff, c_loss_x * args.c_coeff, kl_loss,
#             total_loss))
#
#         current_loss, val_metrics = validate_mpvae(
#             model, data.input_feat, data.labels, data.valid_idx, args)


# def regularzie_mpvae_unfair(data, model, optimizer, args, use_valid=True):
#     if use_valid:
#         idxs = data.valid_idx
#     else:
#         idxs = data.train_idx
#
#     np.random.shuffle(idxs)
#
#     optimizer.zero_grad()
#
#     labels_z, feats_z = [], []
#     for i in range(int(len(idxs) / float(data.batch_size)) + 1):
#         start = i * data.batch_size
#         end = min(data.batch_size * (i + 1), len(idxs))
#
#         input_feat = data.input_feat[idxs[start:end]]
#         input_feat = torch.from_numpy(input_feat).to(device)
#
#         input_label = data.labels[idxs[start:end]]
#         input_label = torch.from_numpy(input_label).to(device)
#         label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
#             input_label, input_feat)
#
#         label_z = model.label_reparameterize(label_mu, label_logvar)
#         feat_z = model.feat_reparameterize(feat_mu, feat_logvar)
#         labels_z.append(label_z)
#         feats_z.append(feat_z)
#
#     labels_z = torch.cat(labels_z)
#     feats_z = torch.cat(feats_z)
#     clusters = torch.from_numpy(data.label_clusters[idxs]).to(device)
#     sensitive_feat = torch.from_numpy(data.sensitive_feat[idxs]).to(device)
#
#     labels_z_unfair = 0.
#     feats_z_unfair = 0.
#     label_centroids = torch.unique(clusters)
#     sensitive_centroids = torch.unique(sensitive_feat, dim=0)
#
#     idx = torch.arange(clusters.shape[0])
#     for centroid in label_centroids:
#         target_centroid = torch.eq(clusters, centroid)
#         cluster_labels_z = labels_z[idx[target_centroid]]
#         if len(cluster_labels_z):
#             for sensitive in sensitive_centroids:
#                 target_sensitive = torch.all(torch.eq(sensitive_feat, sensitive), dim=1)
#                 sensitive_centroid = torch.all(
#                     torch.stack((target_sensitive, target_centroid), dim=1), dim=1)
#                 cluster_labels_z_sensitive = labels_z[idx[sensitive_centroid]]
#                 if len(cluster_labels_z_sensitive):
#                     labels_z_unfair += torch.pow(
#                         cluster_labels_z_sensitive.mean() - cluster_labels_z.mean(), 2)
#
#         cluster_feats_z = feats_z[idx[target_centroid]]
#         if len(cluster_feats_z):
#             for sensitive in sensitive_centroids:
#                 target_sensitive = torch.all(torch.eq(sensitive_feat, sensitive), dim=1)
#                 sensitive_centroid = torch.all(
#                     torch.stack((target_sensitive, target_centroid), dim=1), dim=1)
#                 cluster_feats_z_sensitive = feats_z[idx[sensitive_centroid]]
#                 if len(cluster_feats_z_sensitive):
#                     feats_z_unfair += torch.pow(
#                         cluster_feats_z_sensitive.mean() - cluster_feats_z.mean(), 2)
#
#     fairloss = args.label_z_fair_coeff * labels_z_unfair + args.feat_z_fair_coeff * feats_z_unfair
#     if isinstance(fairloss, float):
#         raise UserWarning('Fail to construct fairness regualizers')
#     else:
#         fairloss.backward()
#         optimizer.step()


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

    model_dir = f'fairreg/model/model_{args.dataset}/{param_setting}'
    summary_dir = f'fairreg/summary/{args.dataset}/{param_setting}'
    build_path(model_dir, summary_dir)

    train_fair_through_regularize()