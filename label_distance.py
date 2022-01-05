import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from data import load_data, preprocess


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


def apriori_similarity(args, gamma=1., minimum_clip=1e-6, maximum_clip=1.):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels = labels.astype(str)

    labels_oh_str = np.concatenate([labels_oh.astype(str), labels], axis=1)
    labels_oh_str = np.unique(labels_oh_str, axis=0)

    labels_express = {}
    for label in labels_oh_str:
        label_oh = label[:-3]
        label_str = label[-3:]
        labels_express[frozenset(label_str)] = label_oh.astype(int)

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
                    dist = apriori_pair_dist(p1, p2, labels_rules)
                    weight = np.exp(-gamma * dist)
                    dist_dict[lab1][lab2] = np.clip(weight, minimum_clip, maximum_clip)

    return dist_dict


def indication_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels = labels.astype(str)

    labels_oh_str = np.concatenate([labels_oh.astype(str), labels], axis=1)
    labels_oh_str = np.unique(labels_oh_str, axis=0)

    labels_express = {}
    for label in labels_oh_str:
        label_oh = label[:-3]
        label_str = label[-3:]
        labels_express[frozenset(label_str)] = label_oh.astype(int)

    income_level = np.unique(labels[:, 0])
    occupation = np.unique(labels[:, 1])
    workclass = np.unique(labels[:, 2])
    labelsets = []
    for income in income_level:
        for occu in occupation:
            for work in workclass:
                labelsets.append(set([income, occu, work]))

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
                    dist_dict[lab1][lab2] = 1.0 if lab1 == lab2 else 1e-6

    return dist_dict


def constant_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels = labels.astype(str)

    labels_oh_str = np.concatenate([labels_oh.astype(str), labels], axis=1)
    labels_oh_str = np.unique(labels_oh_str, axis=0)

    labels_express = {}
    for label in labels_oh_str:
        label_oh = label[:-3]
        label_str = label[-3:]
        labels_express[frozenset(label_str)] = label_oh.astype(int)

    income_level = np.unique(labels[:, 0])
    occupation = np.unique(labels[:, 1])
    workclass = np.unique(labels[:, 2])
    labelsets = []
    for income in income_level:
        for occu in occupation:
            for work in workclass:
                labelsets.append(set([income, occu, work]))

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
                    dist_dict[lab1][lab2] = 1.0

    return dist_dict


def str_ham_similarity(str1, str2):
    if len(str1) != len(str2):
        raise ValueError('Incompatible string pairs: have different lengths.')
    same = 0
    active = 0
    for idx, char in enumerate(str1):
        if char == '1':
            same += (char == str2[idx])
            active += (char == '1')
    
    return same / active


def hamming_similarity(args):
    np.random.seed(4)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels = labels.astype(str)

    labels_oh_str = np.concatenate([labels_oh.astype(str), labels], axis=1)
    labels_oh_str = np.unique(labels_oh_str, axis=0)

    labels_express = {}
    for label in labels_oh_str:
        label_oh = label[:-3]
        label_str = label[-3:]
        labels_express[frozenset(label_str)] = label_oh.astype(int)

    income_level = np.unique(labels[:, 0])
    occupation = np.unique(labels[:, 1])
    workclass = np.unique(labels[:, 2])
    labelsets = []
    for income in income_level:
        for occu in occupation:
            for work in workclass:
                labelsets.append(set([income, occu, work]))

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
                    dist_dict[lab1][lab2] = np.clip(str_ham_similarity(lab1, lab2), 1e-6, 1.)

    return dist_dict
