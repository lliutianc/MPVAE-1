import os
from typing import cast

import numpy as np
import pandas as pd

from utils import allexists

DATASETPATH = 'dataset/'


# def load_adult(subset):
#     _header = [
#         'age', 'workclass', 'fnlwgt', 'education', 'education_num',
#         'marital_stat', 'occupation', 'relationship', 'race', 'sex',
#         'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
#         'income_level'
#     ]
#     if subset != 'test':
#         dfpath = DATASETPATH + 'adult/adult.data'
#     else:
#         dfpath = DATASETPATH + 'adult/adult.test'
#     df = pd.read_csv(dfpath, header=None)
#     df.columns = _header
#     df['income_level'] = (df['income_level'] == ' <=50K')

#     label_cols = ['income_level', 'occupation', 'workclass']
#     labels = df[label_cols]
#     feat = df.drop(label_cols, axis=1)
#     # The exact computation of fnlwgt is unclear, while it is likely to be highly correlated to sensitive features
#     feat = feat.drop('fnlwgt', axis=1)
#     # feat['fnlwgt'] = feat['fnlwgt'] / feat['fnlwgt'].sum()
#     feat['capital_gain'] = (feat['capital_gain'] - feat['capital_gain'].min()) / \
#                            (feat['capital_gain'].max() -
#                             feat['capital_gain'].min())
#     feat['capital_loss'] = (feat['capital_loss'] - feat['capital_loss'].min()) / \
#                            (feat['capital_loss'].max() -
#                             feat['capital_loss'].min())

#     # labels['occupation'] = labels['occupation'].apply(lambda x: x in [' Handlers-cleaners',' Craft-repair', ' Transport-moving', ' Farming-fishing']).astype(int)
#     # labels['workclass'] = labels['workclass'].apply(lambda x: x in [' Self-emp-not-inc', ' Private', ' Self-emp-inc']).astype(int)

#     sensitive = ['race', 'sex']
#     return feat, labels, sensitive


# def load_credit(subset):
#     _header = [
#         'stat_exist_check_account', 'duration', 'cred_hist', 'purpose', 'cred_amt', 'sav_bonds', 'present_employment',
#         'install_rate', 'gender_marriage', 'guarantor', 'residence_dura', 'property', 'age', 'other_install', 'housing',
#         'cred_num', 'job', 'people_provide_maintain', 'has_telephone', 'is_foreign', 'is_good'
#     ]
#     dfpath = DATASETPATH + 'credit/german.data'
#     df = pd.read_csv(dfpath, header=None, sep=' ')
#     df.columns = _header
#     df['is_good'] = (df['is_good'] == 1)

#     label_cols = ['job', 'present_employment', 'is_good']
#     labels = df[label_cols]
#     feat = df.drop(label_cols, axis=1)

#     feat['duration'] = (feat['duration'] - feat['duration'].min()) / \
#                         (feat['duration'].max() -
#                         feat['duration'].min())
#     feat['cred_amt'] = (feat['cred_amt'] - feat['cred_amt'].min()) / \
#                         (feat['cred_amt'].max() -
#                         feat['cred_amt'].min())
#     feat['age'] = np.ceil(feat['age'] / 20).astype(int).astype(str)

#     train_cnt = int(len(feat) * .9)
#     print(train_cnt)
#     print(feat.shape)
#     if subset != 'test':
#         feat = feat[:train_cnt]
#         labels = labels[:train_cnt]
#     else:
#         feat = feat[train_cnt:]
#         labels = labels[train_cnt:]

#     sensitive = ['gender_marriage', 'age']
#     return feat, labels, sensitive


def load_adult():
    _header = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_stat', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_level'
    ]

    dfpath = DATASETPATH + 'adult/adult.data'
    df = pd.read_csv(dfpath, header=None)
    df.columns = _header
    df['income_level'] = (df['income_level'] == ' <=50K')

    label_cols = ['income_level', 'occupation', 'workclass']
    labels = df[label_cols]
    feat = df.drop(label_cols, axis=1)
    # The exact computation of fnlwgt is unclear, while it is likely to be highly correlated to sensitive features
    feat = feat.drop('fnlwgt', axis=1)
    # feat['fnlwgt'] = feat['fnlwgt'] / feat['fnlwgt'].sum()
    feat['capital_gain'] = (feat['capital_gain'] - feat['capital_gain'].min()) / \
                           (feat['capital_gain'].max() -
                            feat['capital_gain'].min())
    feat['capital_loss'] = (feat['capital_loss'] - feat['capital_loss'].min()) / \
                           (feat['capital_loss'].max() -
                            feat['capital_loss'].min())

    # labels['occupation'] = labels['occupation'].apply(lambda x: x in [' Handlers-cleaners',' Craft-repair', ' Transport-moving', ' Farming-fishing']).astype(int)
    # labels['workclass'] = labels['workclass'].apply(lambda x: x in [' Self-emp-not-inc', ' Private', ' Self-emp-inc']).astype(int)

    sensitive = ['race', 'sex']
    return feat, labels, sensitive


def load_credit():
    _header = [
        'stat_exist_check_account', 'duration', 'cred_hist', 'purpose', 'cred_amt', 'sav_bonds', 'present_employment',
        'install_rate', 'gender_marriage', 'guarantor', 'residence_dura', 'property', 'age', 'other_install', 'housing',
        'cred_num', 'job', 'people_provide_maintain', 'has_telephone', 'is_foreign', 'is_good'
    ]
    dfpath = DATASETPATH + 'credit/german.data'
    df = pd.read_csv(dfpath, header=None, sep=' ')
    df.columns = _header
    df['is_good'] = (df['is_good'] == 1)

    label_cols = ['job', 'present_employment', 'is_good']
    labels = df[label_cols]
    feat = df.drop(label_cols, axis=1)

    feat['duration'] = (feat['duration'] - feat['duration'].min()) / \
        (feat['duration'].max() -
         feat['duration'].min())
    feat['cred_amt'] = (feat['cred_amt'] - feat['cred_amt'].min()) / \
        (feat['cred_amt'].max() -
         feat['cred_amt'].min())
    feat['age'] = np.ceil(feat['age'] / 20).astype(int).astype(str)

    sensitive = ['gender_marriage', 'age']
    return feat, labels, sensitive


def _load_donor_label(subset=None):
    label_cols = [
        'projectid',
        'fully_funded',
        'at_least_1_teacher_referred_donor',
        # 'great_chat',
        'at_least_1_green_donation',
        # 'three_or_more_non_teacher_referred_donors',
        # 'one_non_teacher_referred_donor_giving_100_plus',
        'donation_from_thoughtful_donor',
    ]
    df = pd.read_csv(DATASETPATH + 'donor/outcomes.csv')
    labels = df[label_cols]
    if subset:
        labels = pd.merge(labels, subset, how='inner', on='projectid')

    for binary_feat in [
        'fully_funded',
        'at_least_1_teacher_referred_donor',
        # 'great_chat',
        'at_least_1_green_donation',
        # 'three_or_more_non_teacher_referred_donors',
        # 'one_non_teacher_referred_donor_giving_100_plus',
        'donation_from_thoughtful_donor'
    ]:
        labels[binary_feat] = (labels[binary_feat] == 't')

    return labels


def _load_test_projectid():
    return pd.read_csv(DATASETPATH + 'donor/sampleSubmission.csv')[['projectid']]


def _load_donor_resource_cost(subset=None):
    resource = pd.read_csv(DATASETPATH + 'donor/resources.csv')
    if subset:
        resource = pd.merge(resource, subset, how='inner', on='projectid')
    resource = resource[0 <= resource['item_unit_price'] <= 1e-5]
    resource['item_cost'] = resource['item_unit_price'] * \
        resource['item_quantity']
    resource = resource.groupby('projectid').agg(
        {'item_cost': np.sum}).reset_index()
    resource = resource.rename(columns={'item_cost': 'resource_cost'})

    return resource


def _load_donor_projects(subset=None):
    feat_cols = [
        'projectid',
        # 'school_city',
        'school_state',
        'school_charter',
        'school_magnet',
        'school_year_round',
        'school_nlns',
        'school_kipp',
        'school_charter_ready_promise',
        'teacher_prefix',
        'teacher_teach_for_america',
        'teacher_ny_teaching_fellow',
        'primary_focus_subject',
        'primary_focus_area',
        'secondary_focus_subject',
        'secondary_focus_area',
        'resource_type',
        'poverty_level',
        'grade_level',
        'fulfillment_labor_materials',
        'total_price_excluding_optional_support',
        'total_price_including_optional_support',
        'students_reached',
        'eligible_double_your_impact_match',
        'eligible_almost_home_match'
    ]

    feat = pd.read_csv(DATASETPATH + 'donor/projects.csv')[feat_cols]
    if subset:
        feat = pd.merge(feat, subset, how='inner', on='projectid')

    for binary_feat in [
        'school_charter',
        'school_magnet',
        'school_year_round',
        'school_nlns',
        'school_kipp',
        'school_charter_ready_promise',
        'teacher_teach_for_america',
        'teacher_ny_teaching_fellow',
        'eligible_double_your_impact_match',
        'eligible_almost_home_match',
    ]:
        feat[binary_feat] = (feat[binary_feat] == 't')

    sensitive = ['poverty_level', 'teacher_prefix']
    return feat, sensitive


def load_donor(subset=None):
    if subset == 'test':
        subset = _load_test_projectid()
    else:
        subset = None

    labels = _load_donor_label(subset)
    feat, sensitive = _load_donor_projects(subset)
    feat = pd.merge(
        feat, labels[['projectid']], how='inner', on='projectid')
    labels = pd.merge(
        labels, feat[['projectid']], how='inner', on='projectid')

    # resource_cost  = _load_donor_resource_cost(subset)
    # feat = pd.merge(feat, resource_cost, how='inner', on='projectid')

    feat = feat.drop('projectid', axis=1)
    labels = labels.drop('projectid', axis=1)
    feat = feat.fillna(0)
    labels = labels.fillna(0)

    for normalize in [
        'total_price_excluding_optional_support',
        'total_price_including_optional_support',
        'students_reached'
    ]:
        feat[normalize] = (feat[normalize] - feat[normalize].min()) / \
                          (feat[normalize].max() - feat[normalize].min())

    return feat, labels, sensitive


def preprocess(df, categorical_encode, show_categorcial_idx=False):
    df = np.array(df)
    if categorical_encode not in ['onehot', 'categorical', None]:
        raise ValueError('Unrecognized categorical_encode')

    npy_cast = []
    npy_str_idx = []
    for i in range(df.shape[1]):
        col = df[:, i]
        if isinstance(col[0], str):
            if categorical_encode:
                npy_str_idx.append(i)
                if categorical_encode == 'onehot':
                    col = onehot(col)
                else:
                    col = categroical(col)
        else:
            col = col.astype(np.float32)

        if len(col.shape) == 1:
            col = col[:, np.newaxis]
        npy_cast.append(col)
    npy = np.concatenate(npy_cast, axis=1)

    if not show_categorcial_idx:
        return npy
    else:
        return npy, npy_str_idx


def onehot(col):
    col = col.astype(str)
    unique_val = np.array(list(set(col)))
    unique_val.sort()
    return np.char.equal(unique_val[np.newaxis, :], col[:, np.newaxis]).astype(np.int32)


def categroical(col):
    return np.argmax(onehot(col), 1).astype(np.int32)


def cast_to_float(df):
    if df.dtype == np.float64:
        df = df.astype(np.float32)

    return df


def load_data(dataset, mode, separate_sensitive=False, categorical_encode='onehot'):
    if categorical_encode not in ['onehot', 'categorical', None]:
        raise ValueError('Unrecognized categorical_encode')

    if dataset not in ['adult', 'donor', 'credit']:
        raise NotImplementedError()

    datapath = DATASETPATH + dataset
    sensitive_featfile = os.path.join(
        datapath, f'sensitive_{categorical_encode}.npy')
    nonsensitive_featfile = os.path.join(
        datapath, f'nonsensitive_{categorical_encode}.npy')
    labelfile = os.path.join(
        datapath, f'label_{categorical_encode}.npy')

    if not allexists(sensitive_featfile, nonsensitive_featfile, labelfile):
        print(f'prepare dataset: {dataset}...')
        if dataset == 'adult':
            feat, labels, sensitive = load_adult()
        elif dataset == 'donor':
            feat, labels, sensitive = load_donor()
        elif dataset == 'credit':
            feat, labels, sensitive = load_credit()
        else:
            raise NotImplementedError()

        sensitive_feat = feat[sensitive]
        nonsensitive_feat = feat.drop(sensitive, axis=1)

        nonsensitive_feat = preprocess(
            nonsensitive_feat, categorical_encode)
        sensitive_feat = preprocess(
            sensitive_feat, categorical_encode)
        labels = preprocess(
            labels, categorical_encode)

        np.save(open(sensitive_featfile, 'wb'), sensitive_feat)
        np.save(open(nonsensitive_featfile, 'wb'), nonsensitive_feat)
        np.save(open(labelfile, 'wb'), labels)
    else:
        print(f'load existing dataset: {dataset}...')

        sensitive_feat = np.load(
            open(sensitive_featfile, 'rb'), allow_pickle=True)
        nonsensitive_feat = np.load(
            open(nonsensitive_featfile, 'rb'), allow_pickle=True)
        labels = np.load(open(labelfile, 'rb'), allow_pickle=True)

    sensitive_feat = cast_to_float(sensitive_feat)
    nonsensitive_feat = cast_to_float(nonsensitive_feat)
    labels = cast_to_float(labels)

    train_val_cnt = int(0.9 * len(nonsensitive_feat))

    train_cnt, valid_cnt = int(
        len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)
    train_idx = np.arange(train_cnt)
    valid_idx = np.arange(train_cnt, valid_cnt + train_cnt)
    test_idx = np.setdiff1d(
        np.arange(len(nonsensitive_feat)),
        np.concatenate([train_idx, valid_idx]))

    if separate_sensitive:
        return nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx, test_idx
    else:
        return np.concatenate([nonsensitive_feat, sensitive_feat], axis=1), labels, train_idx, valid_idx, test_idx


def load_data_masked(dataset, mode, separate_sensitive=False, categorical_encode='onehot', masked_label=None, unmasked_sensitive=None):
    if dataset not in ['adult', 'donor', 'credit']:
        raise NotImplementedError(
            f'cannot masked dataset {dataset}...')
    if not separate_sensitive:
        raise ValueError(
            'can only create masked dataset when `separate_sensitive=True`...')
    nonsensitive_feat, sensitive_feat, labels, _, _, _ = load_data(
        dataset, mode, separate_sensitive, categorical_encode)

    if masked_label is None:
        label, cnt = np.unique(labels, axis=0, return_counts=True)
        count_sort_idx = np.argsort(-cnt)
        label = label[count_sort_idx]
        masked_label = label[0]
    if unmasked_sensitive is None:
        sen, cnt = np.unique(sensitive_feat, axis=0, return_counts=True)
        count_sort_idx = np.argsort(-cnt)
        sen = sen[count_sort_idx]
        unmasked_sensitive = sen[0]

    sensitive_mask = (~np.all(
        np.equal(unmasked_sensitive, sensitive_feat), axis=1))
    label_mask = np.all(np.equal(masked_label, labels), axis=1)

    unmasked_idx = np.arange(len(nonsensitive_feat))[
        ~np.all([sensitive_mask, label_mask], axis=0)]
    masked_idx = np.arange(len(nonsensitive_feat))[
        np.all([sensitive_mask, label_mask], axis=0)]

    train_cnt, valid_cnt = int(
        len(nonsensitive_feat) * 0.7), int(len(nonsensitive_feat) * .2)

    train_idx = unmasked_idx[:train_cnt]

    valid_idx = np.concatenate([
        unmasked_idx[train_cnt: (train_cnt + valid_cnt)],
        masked_idx], axis=0)

    test_idx = np.setdiff1d(
        np.arange(len(nonsensitive_feat)),
        np.concatenate([train_idx, valid_idx])
    )

    if separate_sensitive:
        return nonsensitive_feat, sensitive_feat, labels, train_idx, valid_idx, test_idx
    else:
        return np.concatenate([nonsensitive_feat, sensitive_feat], axis=1), labels, train_idx, valid_idx, test_idx
