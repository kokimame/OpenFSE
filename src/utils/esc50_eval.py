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

from tqdm import tqdm, trange
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score

from src.utils.utils import import_dataset_from_pt
from models.model_vgg_dropout import VGGModelDropout


class Classifier(nn.Module):
    def __init__(self, input_shape, version=None):
        super(Classifier, self).__init__()
        self.version = version
        self.lin_bn = nn.BatchNorm1d(input_shape, affine=False)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 50)
        self.fc1m = nn.Linear(input_shape, 128)

    def forward(self, x):
        if self.version == 'dropout':
            x = self.forward_dropout(x)
        elif self.version == 'mini':
            x = self.forward_mini(x)
        elif self.version == 'plain' or self.version is None:
            x = self.forward_plain(x)
        return x

    def forward_dropout(self, x):
        x = self.lin_bn(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x

    def forward_plain(self, x):
        x = self.lin_bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward_mini(self, x):
        x = self.lin_bn(x)
        x = F.relu(self.fc1m(x))
        x = F.relu(self.fc3(x))
        return x


class ESC50Dataset(Dataset):
    def __init__(self, data, labels, model_name=None):
        if model_name is not None and model_name != 'openl3':
            with open('../data/model_defaults.json') as f:
                d = json.load(f)
            openfse = VGGModelDropout(emb_size=d['emb_size'])
            openfse.load_state_dict(torch.load(f'{model_name}'))
            self.data = openfse(torch.cat(data).unsqueeze(dim=1)).detach()
        elif model_name == 'openl3':
            self.data = data
        else:
            self.data = torch.stack([torch.squeeze(d) for d in data])
        labels_list = list(set(labels))
        self.labels = torch.LongTensor([
            labels_list.index(label) for label in labels
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def train(classifier, train_loader, val_loader):
    train_loss, val_loss = [], []
    train_xticks, val_xticks = [], []
    val_acc = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01)

    for epoch in range(NUMBER_OF_EPOCHS):
        train_pbar = tqdm(train_loader)
        train_avg_loss = []
        for inputs, targets in train_pbar:
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_avg_loss.append(loss.item())
            train_pbar.set_description(f'Epoch #{epoch:5d} | Train Avg. Loss: {np.mean(train_avg_loss):.2f}')
        train_loss.append(np.mean(train_avg_loss))
        train_xticks.append(epoch)

        if epoch % 10 == 0:
            val_pbar = tqdm(val_loader)
            val_avg_loss, val_avg_acc = [], []

            for inputs, targets in val_pbar:
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = accuracy_score(targets.tolist(), predictions.tolist()) * 100
                val_avg_acc.append(accuracy)
                val_avg_loss.append(loss.item())
                val_pbar.set_description(f'*** Avg. Val Loss: {np.mean(val_avg_loss):.2f} | Accuracy: {accuracy:.2f}%')

            val_loss.append(np.mean(val_avg_loss))
            val_acc.append(np.mean(val_avg_acc))
            val_xticks.append(epoch)

    return train_loss, val_loss, train_xticks, val_xticks, val_acc

def discriminative_tasks(classifier_version):
    # --------------------------------------------------------------------
    # Classification on ESC50 dataset using different types of input data
    # --------------------------------------------------------------------
    # Classifier for embedding input (OpenFSE)
    classifier_emb = Classifier(input_shape=EMBEDDING_SHAPE, version=classifier_version)
    # Classifier for MFCC input
    classifier_mfcc = Classifier(input_shape=MFCC_SHAPE, version=classifier_version)

    result_lookup = {'loss' : {}, 'acc' : {}}
    for classifier, train_loader, val_loader, setting_name in [
        (classifier_emb, train_l3_loader, val_l3_loader, 'L3'),  # Baseline
        (classifier_mfcc, train_mfcc_loader, val_mfcc_loader, 'MFCC'),  # Baseline
        (classifier_emb, train_ma_loader, val_ma_loader, 'MA'),         # With margin adapter
        (classifier_emb, train_woma_loader, val_woma_loader, 'WOMA'),   # Without margin adapter
    ]:
        print(' -------------------------------- ')
        print(f'Start: Training classifier for {setting_name}')
        print(' -------------------------------- \n')
        time.sleep(1)
        tl, vl, txs, vxs, va = train(classifier, train_loader, val_loader)
        result_lookup['loss'][setting_name] = [tl, txs, vl, vxs]
        result_lookup['acc'][setting_name] = [vxs, va]

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 2, 1)
    color_lookup = {'MA' : 'C0', 'WOMA' : 'C1', 'MFCC' : 'C2', 'L3' : 'C3'}
    for setting_name, (tl, txs, vl, vxs) in result_lookup['loss'].items():
        plt.plot(txs, tl, label=f'{setting_name} - Train', color=color_lookup[setting_name])
        plt.plot(vxs, vl, label=f'{setting_name} - Val', linestyle='-.', color=color_lookup[setting_name])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(np.arange(0, NUMBER_OF_EPOCHS + 1, NUMBER_OF_EPOCHS//20), rotation=45)

    plt.subplot(1, 2, 2)
    for setting_name, (vxs, va) in result_lookup['acc'].items():
        plt.plot(vxs, va, label=f'{setting_name} - Acc.', color=color_lookup[setting_name])
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.xticks(np.arange(0, NUMBER_OF_EPOCHS + 1, NUMBER_OF_EPOCHS//20), rotation=45)

    plt.show()


def clustering_tasks():
    print(' ----------------------------------------------- ')
    print('Start: Clustering tasks using Adjusted Rand Score')
    print(' ----------------------------------------------- \n')

    result_lookup = {}
    for train_set, val_set, setting_name in [
        (train_l3_set, val_l3_set, 'L3'),  # Baseline
        (train_mfcc_set, val_mfcc_set, 'MFCC'),  # Baseline
        (train_ma_set, val_ma_set, 'MA'),         # With margin adapter
        (train_woma_set, val_woma_set, 'WOMA'),   # Without margin adapter
    ]:
        kmeans = KMeans(n_clusters=50, random_state=0).fit(torch.cat([train_set.data, val_set.data]))
        ars = adjusted_rand_score(kmeans.labels_, torch.cat([train_set.labels, val_set.labels]))
        result_lookup[setting_name] = ars
        print(f'Adjusted Rand Score ({setting_name}): {ars}')

    plt.bar(np.arange(len(result_lookup)), result_lookup.values(), align='center', alpha=0.5)
    plt.xticks(np.arange(len(result_lookup)), result_lookup.keys())
    plt.ylabel('Adjusted Rand Score')
    plt.show()


EMBEDDING_SHAPE = 512
MFCC_SHAPE = 120
NUMBER_OF_EPOCHS = 100
CLASSIFIER_VERSION = 'plain'
# -------------- Specifying paths and loading datasets ----------------
MODEL_MA_PATH = '../saved_models/model_multi_top500_2020-07-31_10:58:32.673148.pt'
MODEL_WOMA_PATH = '../saved_models/unique5_12k_semihard_margin2_lr001.pt'
ESC50_DIR = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50'
ESC50_SPEC = f'{ESC50_DIR}/spec'
ESC50_AUDIO = f'{ESC50_DIR}/audio'


TRAIN_SPEC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_train_1.pt'
VAL_SPEC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_val.pt'
TRAIN_MFCC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_mfcc_train_1.pt'
VAL_MFCC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_mfcc_val.pt'
TRAIN_L3_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_l3_train_1.pt'
VAL_L3_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_l3_val.pt'

print(' ---------------------------- ')
print('Start: Loading models and data')
print(' ---------------------------- \n')

# -- L3
train_l3_data, train_l3_labels, _ = import_dataset_from_pt('{}'.format(TRAIN_L3_PATH), chunks=1)
train_l3_set = ESC50Dataset(train_l3_data, train_l3_labels)
train_l3_loader = torch.utils.data.DataLoader(train_l3_set, batch_size=4, shuffle=True)
print(f'Train (L3) data has been loaded!\t\t\t | Data Size: {len(train_l3_data)}')

val_l3_data, val_l3_labels, _ = import_dataset_from_pt('{}'.format(VAL_L3_PATH), chunks=1)
val_l3_set = ESC50Dataset(val_l3_data, val_l3_labels)
val_l3_loader = torch.utils.data.DataLoader(val_l3_set, batch_size=len(val_l3_set), shuffle=True)
print(f'Validation (L3) data has been loaded!\t\t | Data Size: {len(val_l3_data)}')

# -- MA
train_spec_data, train_spec_labels, _ = import_dataset_from_pt('{}'.format(TRAIN_SPEC_PATH), chunks=1)
val_spec_data, val_spec_labels, _ = import_dataset_from_pt('{}'.format(VAL_SPEC_PATH), chunks=1)

train_ma_set = ESC50Dataset(train_spec_data, train_spec_labels, model_name=MODEL_MA_PATH)
train_ma_loader = torch.utils.data.DataLoader(train_ma_set, batch_size=4, shuffle=True)
print(f'Train (MA) data has been loaded!\t\t\t | Data Size: {len(train_spec_data)}')
val_ma_set = ESC50Dataset(val_spec_data, val_spec_labels, model_name=MODEL_MA_PATH)
val_ma_loader = torch.utils.data.DataLoader(val_ma_set, batch_size=len(val_ma_set), shuffle=True)
print(f'Validation (MA) data has been loaded!\t\t | Data Size: {len(val_spec_data)}')

# -- WOMA
train_woma_set = ESC50Dataset(train_spec_data, train_spec_labels, model_name=MODEL_WOMA_PATH)
train_woma_loader = torch.utils.data.DataLoader(train_woma_set, batch_size=4, shuffle=True)
print(f'Train (WOMA) data has been loaded!\t\t\t | Data Size: {len(train_spec_data)}')
val_woma_set = ESC50Dataset(val_spec_data, val_spec_labels, model_name=MODEL_WOMA_PATH)
val_woma_loader = torch.utils.data.DataLoader(val_woma_set, batch_size=len(val_woma_set), shuffle=True)
print(f'Validation (WOMA) data has been loaded!\t\t | Data Size: {len(val_spec_data)}')

# -- MFCC
train_mfcc_data, train_mfcc_labels, _ = import_dataset_from_pt('{}'.format(TRAIN_MFCC_PATH), chunks=1)
train_mfcc_set = ESC50Dataset(train_mfcc_data, train_mfcc_labels)
train_mfcc_loader = torch.utils.data.DataLoader(train_mfcc_set, batch_size=4, shuffle=True)
print(f'Train (MFCC) data has been loaded!\t\t\t | Data Size: {len(train_mfcc_data)}')

val_mfcc_data, val_mfcc_labels, _ = import_dataset_from_pt('{}'.format(VAL_MFCC_PATH), chunks=1)
val_mfcc_set = ESC50Dataset(val_mfcc_data, val_mfcc_labels)
val_mfcc_loader = torch.utils.data.DataLoader(val_mfcc_set, batch_size=len(val_mfcc_set), shuffle=True)
print(f'Validation (MFCC) data has been loaded!\t\t | Data Size: {len(val_mfcc_data)}')
print()
time.sleep(1)


clustering_tasks()
discriminative_tasks(CLASSIFIER_VERSION)