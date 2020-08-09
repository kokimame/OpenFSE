import os
import json
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from src.utils.utils import import_dataset_from_pt


def create_folds(features, labels, ids):
    folds = [defaultdict(list) for _ in range(5)]
    for feature, label, filename in tqdm(zip(features, labels, ids), total=len(features)):
        metadata = filename.split('-')
        try:
            fold_idx = int(metadata[0]) - 1
            class_idx = int(metadata[-1])
            folds[fold_idx]['X'].append(feature.tolist())
            folds[fold_idx]['y'].append(class_idx)
        except:
            pass
    return folds

def return_other_fold_indexes(test_fold_idx):
    return [i for i in range(5) if i != test_fold_idx]

def train(folds):
    scores = []
    for fold_idx, test_fold in enumerate(folds):
        other_fold_indexes = return_other_fold_indexes(fold_idx)
        X = np.array(list(chain(*[folds[idx]['X'] for idx in other_fold_indexes])))
        y = np.array(list(chain(*[folds[idx]['y'] for idx in other_fold_indexes])))
        X_test = np.array(test_fold['X'])
        y_test = np.array(test_fold['y'])

        if len(X_test.shape) > 2:
            X = X.mean(axis=1)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=NUMBER_OF_EPOCHS)
        clf.fit(X, y)

        if len(X_test.shape) > 2:
            X_test = X_test.mean(axis=1)
        X_test = scaler.transform(X_test)
        score = clf.score(X_test, y_test)
        scores.append(score)
        print(f'Fold #{fold_idx + 1}: Score {score}')

    print(f'Average Score  {np.mean(scores)}\n')

    return np.mean(scores)


def discriminative_tasks():

    for folds, setting_name in [
        (l3_folds, 'L3'),  # Baseline
        (mfcc_folds, 'MFCC'),  # Baseline
        (ma_folds, 'MA'),         # With margin adapter
        (woma_folds, 'WOMA'),   # Without margin adapter
    ]:
        print(' -------------------------------- ')
        print(f'Start: Training classifier for {setting_name}')
        print(' -------------------------------- \n')
        time.sleep(1)
        mean_score = train(folds)


def clustering_tasks():
    print(' ----------------------------------------------- ')
    print('Start: Clustering tasks using Adjusted Rand Score')
    print(' ----------------------------------------------- \n')

    result_lookup = {}
    for features, labels, setting_name in [
        (l3_features, l3_labels, 'L3'),  # Baseline
        (mfcc_features, mfcc_labels, 'MFCC'),  # Baseline
        (ma_features, ma_labels, 'MA'),         # With margin adapter
        (woma_features, woma_labels, 'WOMA'),   # Without margin adapter
    ]:
        kmeans = KMeans(n_clusters=50, random_state=0).fit(features)
        ars = adjusted_rand_score(kmeans.labels_, labels)
        result_lookup[setting_name] = ars
        print(f'Adjusted Rand Score ({setting_name}): {ars}')

    plt.bar(np.arange(len(result_lookup)), result_lookup.values(), align='center', alpha=0.5)
    plt.xticks(np.arange(len(result_lookup)), result_lookup.keys())
    plt.ylabel('Adjusted Rand Score')
    plt.show()


NUMBER_OF_EPOCHS = 300

# -------------- Specifying paths and loading datasets ----------------
FSE_MA_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_top100_ma_8k.pt'
FSE_WOMA_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_top100_woma_8k.pt'
L3_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_l3.pt'
MFCC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_mfcc.pt'

print(' ---------------------------- ')
print('Start: Loading models and data')
print(' ---------------------------- \n')
# -- L3
l3_features, l3_labels, l3_ids = import_dataset_from_pt('{}'.format(L3_PATH), chunks=1)
l3_folds = create_folds(l3_features, l3_labels, l3_ids)
print(f'L3 data has been loaded!\t\t\t | Data Size: {len(l3_features)}')

# -- MA
ma_features, ma_labels, ma_ids = import_dataset_from_pt('{}'.format(FSE_MA_PATH), chunks=1)
ma_folds = create_folds(ma_features, ma_labels, ma_ids)
print(f'MA data has been loaded!\t\t\t | Data Size: {len(ma_features)}')

# -- WOMA
woma_features, woma_labels, woma_ids = import_dataset_from_pt('{}'.format(FSE_WOMA_PATH), chunks=1)
woma_folds = create_folds(woma_features, woma_labels, woma_ids)
print(f'WOMA data has been loaded!\t\t\t | Data Size: {len(woma_features)}')

# -- MFCC
mfcc_features, mfcc_labels, mfcc_ids = import_dataset_from_pt('{}'.format(MFCC_PATH), chunks=1)
mfcc_folds = create_folds(mfcc_features, mfcc_labels, mfcc_ids)
print(f'MFCC data has been loaded!\t\t\t | Data Size: {len(mfcc_features)}')

print()
time.sleep(1)

discriminative_tasks()
# clustering_tasks()