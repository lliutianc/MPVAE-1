from mpvae import VAE, compute_loss
import evals
import numpy as np
import torch.nn as nn
import torch
from main import THRESHOLDS, METRICS
from data import load_data
from tqdm import tqdm
from torch import optim

import sys
import datetime
from copy import deepcopy
sys.path.append('./')


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

    contributed_reg_fair_sample = 0
    succses_updates = 0
    print(data.batch_size)
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
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, indiv_prob_label = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    r_sqrt_sigma, args)
            else:
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, indiv_prob_label = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    model.r_sqrt_sigma, args)

            if penalize_unfair:
                # label_z = model.label_reparameterize(label_mu, label_logvar)
                # feat_z = model.feat_reparameterize(feat_mu, feat_logvar)
                # label_z = label_out
                # feat_z = feat_out
                label_z = indiv_prob_label
                feat_z = indiv_prob

                sensitive_feat = torch.from_numpy(
                    data.sensitive_feat[idx]).to(args.device)
                sensitive_centroids = torch.unique(sensitive_feat, dim=0)
                idx_tensor = torch.arange(sensitive_feat.shape[0])

                reg_label_z_unfair = 0.
                reg_feat_z_unfair = 0.

                for target_fair_label in target_fair_labels:
                    target_label_dist = label_distances[target_fair_label]
                    weights = []
                    for label in data.labels[idx]:
                        label = label.astype(int)
                        distance = target_label_dist.get(
                            ''.join(label.astype(str)), 0.)
                        weights.append(distance)
                        if distance > 0:
                            contributed_reg_fair_sample += 1
                    weights = torch.tensor(weights).to(
                        args.device).reshape(-1, 1)

                    if weights.sum() > 0:
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

                            if weight_sensitive.sum() > 0:
                                reg_label_z_sen = torch.sum(
                                    label_z_sensitive * weight_sensitive, 0) / weight_sensitive.sum()
                                reg_feat_z_sen = torch.sum(
                                    feat_z_sensitive * weight_sensitive, 0) / weight_sensitive.sum()
                                reg_label_z_unfair += torch.sum(
                                    torch.pow(reg_label_z_sen - label_z_weighted, 2))
                                reg_feat_z_unfair += torch.sum(
                                    torch.pow(reg_feat_z_sen - feat_z_weighted, 2))

                fairloss = args.fair_coeff * (reg_label_z_unfair + reg_feat_z_unfair)

                # fairloss = args.label_z_fair_coeff * reg_label_z_unfair

                if not isinstance(fairloss, float):
                    total_loss += fairloss
                    smooth_reg_fair += fairloss.item()

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if has_finite_grad(model):
                optimizer.step()
                if scheduler:
                    scheduler.step()
                succses_updates += 1

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
                               'success_updates': succses_updates,
                               }
            if penalize_unfair:
                running_postfix['fair_loss'] = smooth_reg_fair / float(i + 1)
                running_postfix['contributed_sample'] = contributed_reg_fair_sample
            t.set_postfix(running_postfix)

    print(f'contributed samples: {contributed_reg_fair_sample}')

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


def has_finite_grad(model):
    finite_grad = True
    for param in model.parameters():
        if param.grad is not None:
            valid_gradients = not (torch.isnan(
                param.grad).any() or torch.isinf(param.grad).any())
            finite_grad = finite_grad and valid_gradients

    return finite_grad


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
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, indiv_prob_label = compute_loss(
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
