import os

import numpy as np
import pandas as pd

from utils import allexists

DATASETPATH = 'dataset/'


def load_adult(subset):
    _header = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
        'marital_stat', 'occupation', 'relationship', 'race', 'sex', 
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 
        'income_level'
    ]
    if subset == 'train':
        dfpath = DATASETPATH + 'adult/adult.data'
    else:
        dfpath = DATASETPATH + 'adult/adult.test'
    df = pd.read_csv(dfpath, header=None)
    df.columns = _header
    df['income_level'] = (df['income_level'] == ' <=50K')
    
    label_cols = ['income_level', 'occupation', 'workclass']
    labels = df[label_cols]
    feat = df.drop(label_cols, axis=1)
    sensitive = ['race', 'sex']
    return feat, labels, sensitive


def _load_donor_label(subset=None):
    label_cols = [
        'projectid',
        'fully_funded',
        'at_least_1_teacher_referred_donor',
        'great_chat',
        'at_least_1_green_donation',
        'three_or_more_non_teacher_referred_donors',
        'one_non_teacher_referred_donor_giving_100_plus',
        'donation_from_thoughtful_donor',
    ]
    df = pd.read_csv(DATASETPATH + 'donor/outcomes.csv')
    labels = df[label_cols]
    if subset:
        labels = pd.merge(labels, subset, how='inner', on='projectid')    
    return labels


def _load_test_projectid():
    return pd.read_csv(DATASETPATH + 'donor/sampleSubmission.csv')[['projectid']]


def _load_donor_resource_cost(subset=None):
    resource = pd.read_csv(DATASETPATH + 'donor/resources.csv')
    if subset:
        resource = pd.merge(resource, subset, how='inner', on='projectid')
    resource['item_cost'] = resource['item_unit_price'] * resource['item_quantity']
    resource = resource.groupby('projectid').agg({'item_cost': np.sum}).reset_index()
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
        'teacher_ny_teaching_fellow'
        ]:
        feat[binary_feat] = (feat[binary_feat] == 't')

    sensitive = ['poverty_level', 'teacher_prefix']
    return feat, sensitive
    
    
def load_donor(subset):
    if subset == 'test': 
        subset = _load_test_projectid()
    else:
        subset = None
    
    labels = _load_donor_label(subset)
    feat, sensitive = _load_donor_projects(subset)
    resource_cost = _load_donor_resource_cost(subset)
    feat = pd.merge(feat, resource_cost, how='inner', on='projectid')
    feat = feat.drop('projectid', axis=1)
    return feat, labels, sensitive


def cast_to_numpy(df, to_onehot=True):
    npy = np.array(df)
    if to_onehot:
        npy_oh = []
        for i in range(npy.shape[1]):
            col = npy[:, i]
            if isinstance(col[0], str):
                col = onehot(col)
            if len(col.shape) == 1:
                col = col[:, np.newaxis]
            npy_oh.append(col)
        npy = np.concatenate(npy_oh, axis=1)

    return npy.astype(np.float32)


def onehot(col):
    unique_val = np.array(list(set(col)))
    return np.equal(unique_val[np.newaxis, :], col[:, np.newaxis]).astype(np.int32)


def load_data(dataset, mode):
    if dataset not in ['adult', 'donor']:
        raise NotImplementedError()

    datapath = DATASETPATH + dataset
    featfile = os.path.join(datapath, 'feat_oh.npy')
    labelfile = os.path.join(datapath, 'label_oh.npy')

    if not allexists(featfile, labelfile):
        print('prepare dataset...')
        if dataset == 'adult':
            feat, labels, _ = load_adult(mode)
        elif dataset == 'donor':
            feat, labels, sensitive = load_donor(mode)

        feat = cast_to_numpy(feat)
        labels = cast_to_numpy(labels)
        np.save(open(featfile, 'wb'), feat)
        np.save(open(labelfile, 'wb'), labels)

    else:
        print('load existing datasets...')
        feat = np.load(open(featfile, 'rb'))
        labels = np.load(open(labelfile, 'rb'))

    return feat, labels