import sys
import os
import pickle
import types
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from torch.autograd import Variable

from logger import Logger
from utils import allexists, build_path, search_files
from fairsoft_utils import has_finite_grad
from mpvae import compute_loss, VAE
import evals
from main import THRESHOLDS, METRICS
from data import load_data, load_data_masked
from label_distance_obs import jaccard_similarity, constant_similarity, indication_similarity

IMPLEMENTED_METHODS = ['baseline', 'unfair', 'jaccard']

sys.path.append('./')


def load_trained_mpvae_unfair(args):
    trained_mpvae_path = os.path.join(
        f'fair_through_distance/model/{args.dataset}/unfair',
        f'unfair_vae_prior-500.00_{args.seed:04d}.pkl')
    trained_mpvae = VAE(args).to(args.device)
    trained_mpvae.load_state_dict(torch.load(trained_mpvae_path))

    trained_mpvae.eval()
    return trained_mpvae


def logit(p):
    return torch.log(p / (1 - p + 1e-6))


def sigmoid(logit):
    return 1 / (torch.exp(-logit) + 1)


def calibrate_p(p, threshold):
    cal_logit = logit(p) - logit(threshold)
    return sigmoid(cal_logit)


def postprocess_threshold_one_epoch(
        data, model, threshold_, optimizer, scheduler, target_fair_labels, label_distances, args):
    model.eval()

    if target_fair_labels is None:
        raise NotImplementedError('Have not supported smooth-OD yet.')

    target_fair_labels_str = []
    for target_fair_label in target_fair_labels:
        target_fair_label = ''.join(target_fair_label.astype(str))
        target_fair_labels_str.append(target_fair_label)
    target_fair_labels = target_fair_labels_str

    np.random.shuffle(data.train_idx)
    args.device = threshold_.device

    smooth_total_loss = 0.
    smooth_bce_loss = 0.
    smooth_fair_loss = 0.

    smooth_macro_f1 = 0.  # macro_f1 score
    smooth_micro_f1 = 0.  # micro_f1 score

    contributed_reg_fair_sample = 0
    succses_updates = 0
    print(data.batch_size)

    sen_centroids = np.unique(data.sensitive_feat, axis=0)
    sen_centroids = torch.from_numpy(sen_centroids).to(args.device)

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
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, _ = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    r_sqrt_sigma, args)
            else:
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, _ = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar,
                    model.r_sqrt_sigma, args)

            sensitive_feat = torch.from_numpy(
                data.sensitive_feat[idx]).to(args.device)
            sen_belong = torch.all(
                torch.eq(sensitive_feat.unsqueeze(1), sen_centroids), dim=2)

            threshold = threshold_
            if args.learn_logit:
                threshold = sigmoid(threshold)

            cal_prob = calibrate_p(indiv_prob.unsqueeze(-1), threshold)
            cal_prob = cal_prob.transpose(1, 2)[sen_belong]

            bce_loss = -(input_label * torch.log(cal_prob + 1e-6) +
                         (1 - input_label) * torch.log(1 - cal_prob + 1e-6))
            bce_loss = bce_loss.sum(1).mean()

            sensitive_centroids = torch.unique(sensitive_feat, dim=0)
            idx_tensor = torch.arange(sensitive_feat.shape[0])

            fair_loss = 0.
            for target_fair_label in target_fair_labels:
                target_label_dist = label_distances[target_fair_label]
                # compute weights of each sample
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

                # current batch has contributed samples
                if weights.sum() > 0:
                    cal_prob_weighted = torch.sum(
                        cal_prob * weights, axis=0) / weights.sum()

                    for sensitive in sensitive_centroids:
                        target_sensitive = torch.all(
                            torch.eq(sensitive_feat, sensitive), dim=1)
                        cal_prob_sensitive = cal_prob[idx_tensor[target_sensitive]]
                        weight_sensitive = weights[idx_tensor[target_sensitive]]

                        if weight_sensitive.sum() > 0:
                            reg_cal_prob_sen = torch.sum(
                                cal_prob_sensitive * weight_sensitive, 0) / weight_sensitive.sum()
                            fair_loss += torch.sum(
                                torch.pow(reg_cal_prob_sen - cal_prob_weighted, 2))

            total_loss = bce_loss + fair_loss * args.fair_coeff
            smooth_fair_loss += fair_loss.item()

            total_loss.backward()
            nn.utils.clip_grad_norm_(threshold_, 10.)
            if has_finite_grad(threshold_):
                optimizer.step()
                if scheduler:
                    scheduler.step()
                if args.learn_logit:
                    torch.clip(threshold_, 1e-6, 1. - 1e-6)
                succses_updates += 1

            # evaluation
            train_metrics = evals.compute_metrics(
                cal_prob.cpu().data.numpy(), input_label.cpu().data.numpy(), 0.5,
                all_metrics=False)
            macro_f1, micro_f1 = train_metrics['maF1'], train_metrics['miF1']

            smooth_bce_loss += bce_loss.item()
            smooth_macro_f1 += macro_f1.item()
            smooth_micro_f1 += micro_f1.item()
            smooth_total_loss += total_loss.item()
            # log the labels

            running_postfix = {'total_loss': smooth_total_loss / float(i + 1),
                               'smooth_bce_loss': smooth_bce_loss / float(i + 1),
                               'smooth_fair_loss': smooth_fair_loss / float(i + 1),
                               'maF1': smooth_macro_f1 / float(i + 1),
                               'miF1': smooth_micro_f1 / float(i + 1),
                               'success_updates': succses_updates,
                               'contributed samples': contributed_reg_fair_sample
                               }

            t.set_postfix(running_postfix)


def train_fair_through_postprocess(args):

    # load label distance
    if args.label_dist == 'jaccard':
        hparams = f'jaccard_{args.dist_gamma}'
        label_dist_path = os.path.join(
            args.model_dir, f'label_dist-{hparams}.npy')
        if args.train_new == 0 and os.path.exists(label_dist_path):
            label_dist = pickle.load(open(label_dist_path, 'rb'))
        else:
            label_dist = jaccard_similarity(args)
            pickle.dump(label_dist, open(label_dist_path, 'wb'))

    elif args.label_dist == 'constant':
        hparams = f'constant'
        label_dist_path = os.path.join(
            args.model_dir, f'label_dist-{hparams}.npy')
        if args.train_new == 0 and os.path.exists(label_dist_path):
            label_dist = pickle.load(open(label_dist_path, 'rb'))
        else:
            label_dist = constant_similarity(args)
            pickle.dump(label_dist, open(label_dist_path, 'wb'))

    elif args.label_dist == 'indication':
        hparams = f'indication'
        label_dist_path = os.path.join(
            args.model_dir, f'label_dist-{hparams}.npy')

        if args.train_new == 0 and os.path.exists(label_dist_path):
            label_dist = pickle.load(open(label_dist_path, 'rb'))
        else:
            label_dist = indication_similarity(args)
            pickle.dump(label_dist, open(label_dist_path, 'wb'))
    else:
        raise NotImplementedError()

    fair_threshold_path = os.path.join(
        args.model_dir, f'thresold-{hparams}-{args.fair_coeff:.2f}_{args.seed:04d}.pkl'
    )

    if allexists(fair_threshold_path) and bool(args.train_new) is False:
        print(
            f'find fair threshold: {fair_threshold_path}')
        threshold = pickle.load(open(fair_threshold_path, 'rb'))
        threshold = torch.from_numpy(threshold).to(args.device)
    else:
        print(
            f'train new fair threshold: {fair_threshold_path}')

        # prepare data
        np.random.seed(args.seed)
        if args.mask_target_label:
            nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx, test_idx = load_data_masked(
                args.dataset, args.mode, True, 'onehot')
        else:
            nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx, test_idx = load_data(
                args.dataset, args.mode, True, 'onehot')

        # Test fairness on some labels
        label_type, count = np.unique(labels, axis=0, return_counts=True)
        count_sort_idx = np.argsort(-count)
        label_type = label_type[count_sort_idx]
        idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
        target_fair_labels = label_type[idx: idx + 1].astype(int)
        one_epoch_iter = np.ceil(len(train_idx) / args.batch_size)

        data = types.SimpleNamespace(
            input_feat=nonsensitive_feat, labels=labels, sensitive_feat=sensitive_feat,
            train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, batch_size=args.batch_size)

        if args.learn_logit:
            threshold_ = logit(torch.rand(1, labels.shape[1], len(
                np.unique(sensitive_feat, axis=0)), device=args.device) * .1 + .45)
        else:
            threshold_ = torch.rand(1, labels.shape[1], len(
                np.unique(sensitive_feat, axis=0)), device=args.device) * .1 + .45
        threshold_.requires_grad = True

        optimizer = optim.Adam([threshold_],
                               lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        args.feature_dim = data.input_feat.shape[1]
        args.label_dim = data.labels.shape[1]

        trained_mpvae = load_trained_mpvae_unfair(args)

        print('start calibrating probablity...')
        for _ in range(args.max_epoch):
            postprocess_threshold_one_epoch(
                data, trained_mpvae, threshold_, optimizer, scheduler,
                target_fair_labels=target_fair_labels,
                label_distances=label_dist,
                args=args)
            # print(threshold_[0][0].cpu().data.numpy())
            # print(threshold_.shape)
            # exit(1)

        threshold_np = threshold_.cpu().data.numpy()
        pickle.dump(threshold_np, open(fair_threshold_path, 'wb'))


def evaluate_fair_through_postprocess(
        model, data, target_fair_labels, label_distances, threshold_, args,
        subset='train', eval_fairness=True, eval_train=True, eval_valid=True, logger=Logger()):
    if subset == 'train':
        subset_idx = data.train_idx
    elif subset == 'valid':
        subset_idx = data.valid_idx
    else:
        subset_idx = data.test_idx

    if eval_fairness and target_fair_labels is None:
        target_fair_labels = list(label_distances.keys())
        raise NotImplementedError('Have not supported smooth-OD yet.')

    target_fair_labels_str = []
    for target_fair_label in target_fair_labels:
        if isinstance(target_fair_label, np.ndarray):
            target_fair_label = ''.join(target_fair_label.astype(str))
        target_fair_labels_str.append(target_fair_label)
    target_fair_labels = target_fair_labels_str

    sen_centroids = np.unique(data.sensitive_feat, axis=0)
    sen_centroids = torch.from_numpy(sen_centroids).to(args.device)

    with torch.no_grad():
        model.eval()

        if eval_train:
            train_nll_loss = 0
            train_c_loss = 0
            train_total_loss = 0

            train_indiv_prob = []
            train_label = []

            calibrated_prob = []
            if eval_fairness:
                train_feat_z = []
                train_sensitive = data.sensitive_feat[subset_idx]
            with tqdm(
                    range(int(len(subset_idx) / float(data.batch_size)) + 1),
                    desc=f'Evaluate on {subset} set') as t:

                for i in t:
                    start = i * data.batch_size
                    end = min(data.batch_size * (i + 1), len(subset_idx))
                    idx = subset_idx[start:end]

                    input_feat = torch.from_numpy(
                        data.input_feat[idx]).to(args.device)

                    input_label = torch.from_numpy(data.labels[idx])
                    input_label = deepcopy(input_label).float().to(args.device)

                    label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                        input_label, input_feat)
                    total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, _ = compute_loss(
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
                        sensitive_feat = torch.from_numpy(
                            data.sensitive_feat[idx]).to(args.device)
                        sen_belong = torch.all(
                            torch.eq(sensitive_feat.unsqueeze(1), sen_centroids), dim=2)

                        cal_prob = calibrate_p(
                            indiv_prob.unsqueeze(-1), threshold_)
                        cal_prob = cal_prob.transpose(1, 2)[sen_belong]
                        calibrated_prob.append(cal_prob.cpu().data.numpy())

                train_indiv_prob = np.array(train_indiv_prob)
                train_label = np.array(train_label)

                nll_loss = train_nll_loss / len(subset_idx)
                c_loss = train_c_loss / len(subset_idx)
                total_loss = train_total_loss / len(subset_idx)

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
                    train_feat_z = np.concatenate(calibrated_prob)
                    assert train_feat_z.shape[0] == len(subset_idx) and \
                        train_feat_z.shape[1] == data.labels.shape[1]

                    sensitive_centroid = np.unique(train_sensitive, axis=0)
                    idxs = np.arange(len(subset_idx))

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
                                        np.sum(np.power(unfair_feat_z_sen - feat_z_weighted, 2)))

                    mean_diffs = np.mean(mean_diffs)
                    # mean_diffs = np.max(mean_diffs) / \
                    #     (np.min(mean_diffs) + 1e-6)

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

        return best_val_metrics


def evaluate_target_labels(args, logger=Logger()):
    np.random.seed(args.seed)
    _, _, labels, _, _, _ = load_data(args.dataset, args.mode, True)
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_labels = label_type[idx: idx + 1].astype(int)

    return evaluate_over_labels(target_fair_labels, args, logger)


def evaluate_over_labels(target_fair_labels, args, logger=Logger()):

    np.random.seed(args.seed)
    nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx, test_idx = load_data(
        args.dataset, args.mode, True, 'onehot')

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx,
        valid_idx=valid_idx, batch_size=args.batch_size, test_idx=test_idx, label_clusters=None,
        sensitive_feat=sensitive_feat)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    label_dist_metric_paths = []
    for param_setting in [meth for meth in IMPLEMENTED_METHODS if meth != 'unfair']:
        param_setting += f'_{args.target_label_idx}'
        if args.mask_target_label:
            param_setting += '_masked'

        label_dist_files = search_files(
            os.path.join(args.model_dir, param_setting), postfix='.npy')

        if len(label_dist_files):
            label_dist_metric_paths += [os.path.join(
                args.model_dir, param_setting, label_dist_file) for
                label_dist_file in label_dist_files]
    logger.logging('\n' * 5)
    logger.logging(f"""Fairness definitions: {label_dist_metric_paths}""")

    threshold_paths = []
    for param_setting in [meth for meth in IMPLEMENTED_METHODS if meth != 'unfair']:
        param_setting += f'_{args.target_label_idx}'
        if args.mask_target_label:
            param_setting += '_masked'

        threshold_files = search_files(os.path.join(
            args.model_dir,  param_setting), postfix=f'-{args.fair_coeff:.2f}_{args.seed:04d}.pkl')
        if len(threshold_files):
            threshold_paths += [os.path.join(
                args.model_dir, param_setting, threshold_file) for
                threshold_file in threshold_files]

    logger.logging('\n' * 5)
    logger.logging(f"""Fair thresholds: {threshold_paths}""")

    fair_results = {}
    perform_result = {}
    for dist_metric in label_dist_metric_paths:
        logger.logging(f'Evaluate fairness definition: {dist_metric}...')
        logger.logging('\n' * 3)
        label_dist = pickle.load(open(dist_metric, 'rb'))

        dist_metric = dist_metric.replace(
            '.npy', '').split('/')[-1].split('-')[1]
        fair_results[dist_metric] = {}

        for threshold_path in threshold_paths:
            print(f'Fair threshold: {threshold_path}')
            model = load_trained_mpvae_unfair(args)

            threshold_ = pickle.load(open(threshold_path, 'rb'))
            threshold_ = torch.from_numpy(threshold_).to(args.device)
            threshold = threshold_
            if args.learn_logit:
                threshold = sigmoid(threshold_)

            results = []
            for subset in ['train', 'valid', 'test']:
                results.append(evaluate_fair_through_postprocess(
                    model, data, target_fair_labels, label_dist, threshold, args, subset=subset, logger=logger))

            threshold_trained = threshold_path.split('/')[-1].split('-')[1]

            fair_results[dist_metric][
                threshold_trained] = [result['fair_mean_diff'] for result in results]

            if threshold_trained not in perform_result:
                perform_result[threshold_trained] = {}

            print(args.perform_metric)
            for perform_metric in args.perform_metric:
                perform_result[threshold_trained][perform_metric] = [
                    subset[perform_metric] for subset in results]

    return fair_results, perform_result
