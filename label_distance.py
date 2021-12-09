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


def apriori_distance(args):
    np.random.seed(4)
    _, _, labels = load_data(
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
                    dist_dict[lab1][lab2] = apriori_pair_dist(
                        p1, p2, labels_rules)

    return dist_dict


def indication_distance(args):
    np.random.seed(4)
    _, _, labels = load_data(
        args.dataset, args.mode, True, None)
    labels_oh = preprocess(labels, 'onehot').astype(int)
    labels = labels.astype(str)

    labels_oh_str = np.concatenate([labels_oh.astype(str), labels], axis=1)
    labels_oh_str = np.unique(labels_oh_str, axis=0)
    
    label_type, count = np.unique(labels_oh_str, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    target_fair_labels = label_type[:1].astype(int)
    print(target_fair_labels, count[count_sort_idx])

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
                    dist_dict[lab1][lab2] = 0. if lab1 == lab2 else np.inf

    return dist_dict
