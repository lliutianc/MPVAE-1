import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from data import load_data, preprocess


def indication_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels_oh_str = labels_oh.astype(str)
    label_oh_sets = np.unique(labels_oh_str, axis=0)

    dist_dict = {}
    for lab1 in label_oh_sets:
        lab1 = ''.join(lab1)
        dist_dict[lab1] = {}
        for lab2 in label_oh_sets:
            lab2 = ''.join(lab2)
            dist_dict[lab1][lab2] = 1.0 if lab1 == lab2 else 0.

    return dist_dict


def constant_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels_oh_str = labels_oh.astype(str)
    label_oh_sets = np.unique(labels_oh_str, axis=0)

    dist_dict = {}
    for lab1 in label_oh_sets:
        lab1 = ''.join(lab1)
        dist_dict[lab1] = {}
        for lab2 in label_oh_sets:
            lab2 = ''.join(lab2)
            dist_dict[lab1][lab2] = 1.0

    return dist_dict


def jaccard_similarity(args):
    if args.dist_gamma is None:
        return _jaccard_similarity(args)
    else:
        return _jaccard_nonlinear_similarity(args)


def str_jac_similarity(str1, str2):
    if len(str1) != len(str2):
        raise ValueError('Incompatible string pairs: have different lengths.')
    same = 0
    total = 0
    for idx, char in enumerate(str1):
        if char == '1':
            same += (char == str2[idx])
        if char == '1' or str2[idx] == 1:
            total += 1

    return same / total


def _jaccard_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels_oh_str = labels_oh.astype(str)
    label_oh_sets = np.unique(labels_oh_str, axis=0)

    dist_dict = {}
    for lab1 in label_oh_sets:
        lab1 = ''.join(lab1)
        dist_dict[lab1] = {}
        for lab2 in label_oh_sets:
            lab2 = ''.joiin(lab2)
            dist_dict[lab1][lab2] = str_jac_similarity(lab1, lab2)

    return dist_dict


def _jaccard_nonlinear_similarity(args, minimum_clip=0., maximum_clip=1.):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels_oh_str = labels_oh.astype(str)
    label_oh_sets = np.unique(labels_oh_str, axis=0)

    dist_dict = {}
    for lab1 in label_oh_sets:
        lab1 = ''.join(lab1)
        dist_dict[lab1] = {}
        for lab2 in label_oh_sets:
            lab2 = ''.joiin(lab2)
            sim = str_jac_similarity(lab1, lab2)
            weight = np.exp(args.dist_gamma * (sim - 1))
            dist_dict[lab1][lab2] = np.clip(
                weight, minimum_clip, maximum_clip)

    return dist_dict



def hamming_similarity(args):
    if args.dist_gamma is None:
        return _hamming_similarity(args)
    else:
        return _hamming_nonlinear_similarity(args)


def str_ham_similarity(str1, str2):
    if len(str1) != len(str2):
        raise ValueError('Incompatible string pairs: have different lengths.')
    same = 0
    for idx, char in enumerate(str1):
        same += (char == str2[idx])

    return same / len(str1)


def _hamming_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels_oh_str = labels_oh.astype(str)
    label_oh_sets = np.unique(labels_oh_str, axis=0)

    dist_dict = {}
    for lab1 in label_oh_sets:
        lab1 = ''.join(lab1)
        dist_dict[lab1] = {}
        for lab2 in label_oh_sets:
            lab2 = ''.joiin(lab2)
            dist_dict[lab1][lab2] = str_ham_similarity()(lab1, lab2)

    return dist_dict

def _hamming_nonlinear_similarity(args, minimum_clip=0., maximum_clip=1.):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels_oh_str = labels_oh.astype(str)
    label_oh_sets = np.unique(labels_oh_str, axis=0)

    dist_dict = {}
    for lab1 in label_oh_sets:
        lab1 = ''.join(lab1)
        dist_dict[lab1] = {}
        for lab2 in label_oh_sets:
            lab2 = ''.joiin(lab2)
            sim = str_ham_similarity(lab1, lab2)
            weight = np.exp(args.dist_gamma * (sim - 1))
            dist_dict[lab1][lab2] = np.clip(
                weight, minimum_clip, maximum_clip)

    return dist_dict
