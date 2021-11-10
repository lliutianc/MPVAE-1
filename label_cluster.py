import os, types

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import torch
import torch.nn as nn
from torch import optim

from embed import CBOW, CBOWData
from train import train_mpvae_one_epoch
from data import preprocess, load_data
from model import VAE


def construct_labels_embed(data, args):
    one_epoch_iter = np.ceil(len(data.train_idx) / args.batch_size)
    train_idx = np.arange(int(len(data.labels) * .7))

    if args.labels_embed_method == 'mpvae':
        # train a prior mpvae
        prior_vae = VAE(args).to(args.device)
        prior_vae.train()

        optimizer = optim.Adam(prior_vae.parameters(),
                               lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        prior_vae_checkpoint_path = os.path.join(args.model_dir, f'prior_vae')
        if args.resume and os.path.exists(prior_vae_checkpoint_path):
            print('load trained prior mpvae...')
            prior_vae.load_state_dict(torch.load(prior_vae_checkpoint_path))
        else:
            print('train a new prior mpvae...')
            for _ in range(args.max_epoch // 3):
                train_mpvae_one_epoch(
                    data, prior_vae, optimizer, scheduler,
                    penalize_unfair=False, eval_after_one_epoch=True, args=args)
            torch.save(prior_vae.cpu().state_dict(), prior_vae_checkpoint_path)

        prior_vae = prior_vae.to(args.device)
        with torch.no_grad():
            prior_vae.eval()
            idxs = np.arange(int(len(data.input_feat)))
            labels_mu, labels_logvar = [], []
            for i in range(int(len(idxs) / float(data.batch_size)) + 1):
                start = i * data.batch_size
                end = min(data.batch_size * (i + 1), len(idxs))

                input_feat = data.input_feat[idxs[start:end]]
                input_feat = torch.from_numpy(input_feat).to(args.device)
                input_label = data.labels[idxs[start:end]]
                input_label = torch.from_numpy(input_label).to(args.device)
                label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar = prior_vae(
                    input_label, input_feat)
                labels_mu.append(label_mu.cpu().data.numpy())
                labels_logvar.append(label_logvar.cpu().data.numpy())

        labels_mu = np.concatenate(labels_mu)
        labels_logvar = np.concatenate(labels_logvar)
        labels_embed = labels_mu

    elif args.labels_embed_method == 'cbow':
        cbow_data = CBOWData(data.labels[train_idx], args.device)
        prior_cbow = CBOW(
            data.labels.shape[1], args.latent_dim, 64).to(args.device)
        prior_cbow.train()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(prior_cbow.parameters(),
                               lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, one_epoch_iter * (args.max_epoch / args.lr_decay_times), args.lr_decay_ratio)

        prior_cbow_checkpoint_path = os.path.join(
            args.model_dir, f'prior_cbow')
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
                        grad_norm = nn.utils.clip_grad_norm_(
                            prior_cbow.parameters(), 100)
                        optimizer.step()
                        scheduler.step()
                        smooth_loss += loss.item()
                        t.set_postfix({'running loss': smooth_loss / (i + 1)})
            torch.save(prior_cbow.cpu().state_dict(),
                       prior_cbow_checkpoint_path)

        prior_cbow = prior_cbow.to(args.device)
        with torch.no_grad():
            prior_cbow.eval()
            labels_embed = []
            for idx in np.arange(int(len(data.input_feat))):
                input_label = data.labels[1]
                input_label = torch.from_numpy(input_label).to(args.device)

                idx = torch.arange(data.labels.shape[1], device=args.device)
                label_embed = prior_cbow.get_embedding(idx[input_label == 1])
                labels_embed.append(label_embed.cpu().data.numpy())

            labels_embed = np.vstack(labels_embed)

    elif args.labels_embed_method == 'none':
        labels_embed = np.copy(data.labels)
    else:
        raise NotImplementedError(
            f'Unsupported label embedding method: {args.labels_embed_method}')

    return labels_embed


def instance_based_cluster(labels_embed, cluster_method, args, **kwargs):
    # TODO: do we have soft clustering algorithm? For example, can we use gumble softmax?

    # TODO: how to properly cluster labels based on JSD or KL average distance?
    #  Take KL for instance, afer merging two points, the new cluster is a Gaussian mixture,
    #  do we still have closed form formula to update new distance?
    train_idx = np.arange(int(len(labels_embed) * .7))
    if cluster_method == 'hierarchical':
        distance_threshold = args.labels_cluster_distance_threshold
        succ_cluster = False
        for _ in range(10):
            cluster = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold).fit(labels_embed[train_idx])
            labels_cluster = cluster.labels_
            _, counts = np.unique(labels_cluster, return_counts=True)
            if counts.min() < args.labels_cluster_min_size:
                distance_threshold *= 2
            else:
                succ_cluster = True
                break
        if succ_cluster is False:
            raise UserWarning('Labels clustering not converged')
        labels_cluster = cluster.predict(labels_embed)

    elif cluster_method == 'kmeans':
        n_cluster = 16
        for _ in range(10):
            cluster = KMeans(n_clusters=n_cluster).fit(labels_embed[train_idx])
            labels_cluster = cluster.labels_
            _, counts = np.unique(labels_cluster, return_counts=True)
            if counts.min() < args.labels_cluster_min_size:
                n_cluster -= 2
            else:
                succ_cluster = True
                break
            if n_cluster <= 1:
                break
        if succ_cluster is False:
            raise UserWarning('Labels clustering not converged')
        labels_cluster = cluster.predict(labels_embed)

    elif cluster_method in ['kmodes', 'kprototypes']:
        n_cluster = 16
        for _ in range(10):
            cluster = KPrototypes(n_jobs=-1, n_clusters=n_cluster, init='Cao', random_state=0).fit(
                labels_embed[train_idx], categorical=kwargs.get('catecols'))
            labels_cluster = cluster.labels_
            _, counts = np.unique(labels_cluster, return_counts=True)
            if counts.min() < args.labels_cluster_min_size:
                n_cluster -= 2
            else:
                succ_cluster = True
                break
            if n_cluster <= 1:
                break
        if succ_cluster is False:
            raise UserWarning('Labels clustering not converged')
        labels_cluster = cluster.predict(
            labels_embed, categorical=kwargs.get('catecols'))

    else:
        raise NotImplementedError()

    print(
        f'{len(counts)} clusters: sizes (descending):{np.sort(counts)[::-1]}')

    assert labels_cluster.shape[0] == labels_embed.shape[0], \
        f'{labels_embed.shape}, {labels_cluster.shape}'

    return labels_cluster


def apriori_dist(labelset1, labelset2, apriori_rules):
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


def apriori_cluster(labels, args):
    labels[:, 0] = labels[:, 0].astype(str)
    
    encoder = TransactionEncoder()
    labels_df = encoder.fit_transform(labels)
    cols = encoder.columns_
    cols[0] = 'high_income'
    labels_df = pd.DataFrame(labels_df, columns=cols)
    labels_apri = apriori(
        labels_df, min_support=1 / len(labels_df), use_colnames=True, verbose=1)
    labels_rules = association_rules(labels_apri, min_threshold=0.)

    income_level = np.unique(labels[:, 0])
    occupation = np.unique(labels[:, 1])
    workclass = np.unique(labels[:, 2])

    labelsets = []
    for income in income_level:
        for occu in occupation[1:]:
            for work in workclass[1:]:
                labelsets.append(set([income, occu, work]))

    dist_matrix = np.asarray(
        [[apriori_cluster(p1, p2) for p2 in labelsets] for p1 in labelsets])
    dist_matrix[np.isinf(dist_matrix)] = dist_matrix[~np.isinf(dist_matrix)].max() * 10

    distance_threshold = 0.01
    for _ in range(10):
        cluster = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.05, 
            linkage='average', affinity='precomputed')
        labelcluster = cluster.fit_predict(dist_matrix)

        labelsets_cluster = {}
        for label, cluster in zip(labelsets, labelcluster):
            labelsets_cluster[frozenset(label)] = cluster

        labels_df = pd.DataFrame(labels)
        labels_cluster = labels_df.apply(
            lambda row: labelsets_cluster.get(frozenset(row), max(labelcluster) + 1), axis=1)

        _, counts = np.unique(labels_cluster, return_counts=True)
        if counts.min() < args.labels_cluster_min_size:
            distance_threshold *= 2
        else:
            succ_cluster = True
            break
    
    if succ_cluster is False:
        raise UserWarning('Labels clustering not converged')

    return labels_cluster.to_numpy()
    

def construct_label_clusters(args):
    """
    This function implements various ways of clustering labels. 

    `kmeans`, `hierarchical`, and `kprototypes` support learned embeddings,
    
    `kmodes` directly uses original labels, 
    
    `apriori` computes distance between label1 and label2 using following rule:
        dist(label1, label2) := |confidence(share -> dist_1) - confidence(share -> dist_2)|

    Args:
        args ([type]): args used in training.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: cluster index for each instance, has shape (n_sample, )
    """
    np.random.seed(4)

    nonsensitive_feat, sensitive_feat, labels = load_data(
        args.dataset, args.mode, True, None)
    train_cnt, valid_cnt = int(
        len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)

    data = types.SimpleNamespace(
        input_feat=preprocess(nonsensitive_feat, 'onehot'),
        labels=preprocess(labels, 'onehot'),
        train_idx=train_idx, valid_idx=valid_idx,
        batch_size=args.batch_size)
    args.feature_dim = data.input_feat.shape[1]
    args.label_dim = data.labels.shape[1]

    if args.labels_cluster_method == 'kmodes':
        _, catecols = preprocess(labels, 'categorical', True)
        label_clusters = instance_based_cluster(
            labels, args.labels_cluster_method, args, catecols=catecols)

    elif args.labels_cluster_method in ['kmeans', 'hierarchical']:
        labels_embed = construct_labels_embed(data, args)
        label_clusters = instance_based_cluster(
            labels_embed, args.labels_cluster_method, args)

    elif args.labels_cluster_method == 'kprototypes':
        labels_embed = construct_labels_embed(data, args)
        if args.labels_embed_method != 'none':
            labels_feat = np.hstack([labels_embed, nonsensitive_feat])
        else:
            labels_feat = np.hstack([labels, nonsensitive_feat])
        _, catecols = preprocess(labels_feat, 'categorical', True)
        label_clusters = instance_based_cluster(
            labels, args.labels_cluster_method, args, catecols=catecols)

    elif args.labels_cluster_method == 'apriori':
        if args.dataset != 'adult':
            raise NotImplementedError('Only support adult dataset')
        label_clusters = apriori_cluster(labels, args)

    else:
        raise NotImplementedError()

    assert label_clusters.shape == (nonsensitive_feat.shape[0], )
    return label_clusters.astype(np.int32)
