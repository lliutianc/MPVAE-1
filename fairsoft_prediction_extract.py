from distutils.command.build import build
from multiprocessing.sharedctypes import Value
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
from utils import search_files, build_path
from logger import Logger
from fairsoft_trial import IMPLEMENTED_METHODS

rood_dir = os.path.dirname(__file__)


def extract_prediction_mpvae(model, data, target_fair_labels, args, subset='train', eval_fairness=True, logger=Logger()):
    if subset == 'train':
        subset_idx = data.train_idx
    elif subset == 'valid':
        subset_idx = data.valid_idx
    else:
        subset_idx = data.test_idx

    with torch.no_grad():
        model.eval()

        sensitive_idx = []
        y_probs = []
        y_reals = []
        latent_sample = []
        is_target_label = []

        sen_centroids = np.unique(data.sensitive_feat, axis=0)
        sen_centroids = torch.from_numpy(sen_centroids).to(args.device)

        with tqdm(
                range(int(len(subset_idx) / float(data.batch_size)) + 1),
                desc=f'Evaluate on {subset} set') as t:

            for i in t:
                start = i * data.batch_size
                end = min(data.batch_size * (i + 1), len(subset_idx))
                idx = subset_idx[start:end]

                input_feat = torch.from_numpy(
                    data.input_feat[idx]).to(args.device)

                input_label = data.labels[idx]
                input_label_ = torch.from_numpy(
                    input_label).float().to(args.device)

                label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = model(
                    input_label_, input_feat)

                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, indiv_prob, _ = compute_loss(
                    input_label, label_out, label_mu, label_logvar, feat_out, feat_mu,
                    feat_logvar, model.r_sqrt_sigma, args)

                y_probs.append(indiv_prob.cpu().data.numpy())
                y_reals.append(input_label)

                is_target_label_ = []
                for target_fair_label in target_fair_labels:
                    is_target_label_.append(np.all(
                        np.equal(input_label, target_fair_label), axis=1))
                is_target_label_ = np.any(np.array(is_target_label_), axis=0)
                if len(is_target_label_) != len(feat_out):
                    raise ValueError(f'Incorrect shape of is_target_label,'
                                     f'expected length: {len(feat_out)}, observed length: {len(is_target_label_)}')
                is_target_label.append(is_target_label_)

                sensitive_feat = torch.from_numpy(
                    data.sensitive_feat[idx]).to(args.device)
                sen_belong = torch.all(
                    torch.eq(sensitive_feat.unsqueeze(1), sen_centroids), dim=2)
                if len(sen_belong) != len(feat_out):
                    raise ValueError(f'Incorrect shape of sen_belong,'
                                     f'expected length: {len(feat_out)}, observed length: {len(sen_belong)}')
                sensitive_idx.append(sen_belong.cpu().data.numpy())
            
            y_probs = np.array(y_probs)
            y_reals = np.array(y_reals)

        return y_probs, y_reals, sensitive_idx, is_target_label


def extract_over_labels(target_fair_labels, args, logger=Logger()):

    np.random.seed(args.seed)
    nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx, test_idx = load_data(
        args.dataset, args.mode, True, 'onehot')

    data = types.SimpleNamespace(
        input_feat=nonsensitive_feat, labels=labels, train_idx=train_idx,
        valid_idx=valid_idx,  test_idx=test_idx, batch_size=args.batch_size, label_clusters=None,
        sensitive_feat=sensitive_feat)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    model_paths = []
    result_paths = []
    for model_prior in IMPLEMENTED_METHODS:
        if model_prior != 'unfair':
            model_prior += f'_{args.target_label_idx}'
        if args.mask_target_label:
            model_prior += '_masked'
        model_files = search_files(os.path.join(
            args.model_dir,  model_prior), postfix=f'-{args.fair_coeff:.2f}_{args.seed:04d}.pkl')
        if len(model_files):
            model_paths += [os.path.join(
                args.model_dir, model_prior, model_file) for
                model_file in model_files]
        if len(model_files):
            build_path(os.path.join(
                args.model_dir, 'probability', model_prior))
            result_paths += [os.path.join(
                args.model_dir, 'probability', model_prior, model_file) for
                model_file in model_files]
    logger.logging('\n' * 5)
    logger.logging(f"""Fair Models: {model_paths}""")

    for result_path, model_stat in zip(result_paths, model_paths):
        model = VAE(args).to(args.device)
        model.load_state_dict(torch.load(model_stat))
        results = {}
        for subset in ['train', 'valid', 'test']:
            y_probs, y_reals, sensitive_idx, is_target_label = extract_prediction_mpvae(
                model, data, target_fair_labels, args, subset=subset, logger=logger)
            results[subset] = {'y_probs': y_probs,
                               'y_reals': y_reals,
                               'sensitive_idx': sensitive_idx,
                               'is_target_label': is_target_label}

        pickle.dump(results, open(result_path, 'wb'))


def extract_target_labels(args, logger=Logger()):
    np.random.seed(args.seed)
    _, _, labels, _, _, _ = load_data(args.dataset, args.mode, True)
    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    idx = args.target_label_idx  # idx choices: 0, 10, 20, 50
    target_fair_labels = label_type[idx: idx + 1].astype(int)

    return extract_over_labels(target_fair_labels, args, logger)


if __name__ == '__main__':
    from faircluster_train import parser

    parser.add_argument('-dist_gamma', type=float, default=1.0)
    parser.add_argument('-target_label', type=str, default=None)
    parser.add_argument('-target_label_idx', type=int, default=0)
    parser.add_argument('-mask_target_label', type=int, default=0)
    parser.add_argument('-perform_metric', type=str, default='HA',
                        choices=['ACC', 'HA', 'ebF1', 'maF1', 'miF1'])
    args = parser.parse_args()

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    args.model_dir = f'fair_through_distance/model/{args.dataset}'
    logger = Logger(os.path.join(
        args.model_dir, f'evalution-{args.target_label_idx}.txt'))
    extract_target_labels(args, logger)

# python fairsoft_prediction_extract.py -dataset adult -latent_dim 8 -cuda 3 -target_label_idx 0
