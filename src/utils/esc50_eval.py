import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from tqdm import tqdm, trange
from src.utils.utils import import_dataset_from_pt
from models.model_vgg_dropout import VGGModelDropout
from sklearn.metrics import accuracy_score


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
    def __init__(self, data, labels, model_path=None):
        if model_path is not None:
            with open('../data/model_defaults.json') as f:
                d = json.load(f)
            openfse = VGGModelDropout(emb_size=d['emb_size'])
            openfse.load_state_dict(torch.load(f'{model_path}'))
            self.data = openfse(torch.cat(data).unsqueeze(dim=1)).detach()
        else:
            self.data = [torch.squeeze(d) for d in data]
        labels_list = list(set(labels))
        self.labels = torch.LongTensor([
            labels_list.index(label) for label in labels
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


# -------------- Specifying paths and loading datasets ----------------

# MODEL_PATH = '../saved_models/multi_top100_ma_8k.pt'
MODEL_MA_PATH = '../saved_models/model_multi_top500_2020-07-31_10:58:32.673148.pt'
MODEL_WOMA_PATH = '../saved_models/model_multi_top500_2020-07-31_10:58:32.673148.pt'
ESC50_DIR = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50'
ESC50_SPEC = f'{ESC50_DIR}/spec'

TRAIN_SPEC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_train_1.pt'
VAL_SPEC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_val.pt'
TRAIN_MFCC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_mfcc_train_1.pt'
VAL_MFCC_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_mfcc_val.pt'

train_spec_data, train_spec_labels, _ = import_dataset_from_pt('{}'.format(TRAIN_SPEC_PATH), chunks=1)
val_spec_data, val_spec_labels, _ = import_dataset_from_pt('{}'.format(VAL_SPEC_PATH), chunks=1)

train_ma_set = ESC50Dataset(train_spec_data, train_spec_labels, model_path=MODEL_MA_PATH)
train_ma_loader = torch.utils.data.DataLoader(train_ma_set, batch_size=4, shuffle=True)
print(f'Train (MA) data has been loaded! | Data Size: {len(train_spec_data)}')
val_ma_set = ESC50Dataset(val_spec_data, val_spec_labels, model_path=MODEL_MA_PATH)
val_ma_loader = torch.utils.data.DataLoader(val_ma_set, batch_size=len(val_ma_set), shuffle=True)
print(f'Validation (MA) data has been loaded! | Data Size: {len(val_spec_data)}')

train_woma_set = ESC50Dataset(train_spec_data, train_spec_labels, model_path=MODEL_WOMA_PATH)
train_woma_loader = torch.utils.data.DataLoader(train_woma_set, batch_size=4, shuffle=True)
print(f'Train (WOMA) data has been loaded! | Data Size: {len(train_spec_data)}')
val_woma_set = ESC50Dataset(val_spec_data, val_spec_labels, model_path=MODEL_WOMA_PATH)
val_woma_loader = torch.utils.data.DataLoader(val_woma_set, batch_size=len(val_woma_set), shuffle=True)
print(f'Validation (WOMA) data has been loaded! | Data Size: {len(val_spec_data)}')

train_mfcc_data, train_mfcc_labels, _ = import_dataset_from_pt('{}'.format(TRAIN_MFCC_PATH), chunks=1)
train_mfcc_set = ESC50Dataset(train_mfcc_data, train_mfcc_labels)
train_mfcc_loader = torch.utils.data.DataLoader(train_mfcc_set, batch_size=4, shuffle=True)
print(f'Train (MFCC) data has been loaded! | Data Size: {len(train_mfcc_data)}')

val_mfcc_data, val_mfcc_labels, _ = import_dataset_from_pt('{}'.format(VAL_MFCC_PATH), chunks=1)
val_mfcc_set = ESC50Dataset(val_mfcc_data, val_mfcc_labels)
val_mfcc_loader = torch.utils.data.DataLoader(val_mfcc_set, batch_size=len(val_mfcc_set), shuffle=True)
print(f'Validation (MFCC) data has been loaded! | Data Size: {len(val_mfcc_data)}')
print()
time.sleep(1)

# -----------------------------------------------------------------------
# ------------------- Start training -------------------------
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

# ------------------- Plotting figures ---------------------------------
def plot_figrues():
    classifier_emb = Classifier(input_shape=EMBEDDING_SHAPE, version='plain')
    classifier_mfcc = Classifier(input_shape=MFCC_SHAPE, version='plain')

    result_lookup = {'loss' : {}, 'acc' : {}}
    for classifier, train_loader, val_loader, setting_name in [
        (classifier_mfcc, train_mfcc_loader, val_mfcc_loader, 'MFCC'),
        (classifier_emb, train_ma_loader, val_ma_loader, 'MA'),
        (classifier_emb, train_woma_loader, val_woma_loader, 'WOMA'),
    ]:
        time.sleep(1)
        print(f'Start training for {setting_name}')
        tl, vl, txs, vxs, va = train(classifier, train_loader, val_loader)
        result_lookup['loss'][setting_name] = [tl, txs, vl, vxs]
        result_lookup['acc'][setting_name] = [vxs, va]

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 2, 1)
    color_lookup = {'MA' : 'C0', 'WOMA' : 'C1', 'MFCC' : 'C2'}
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

EMBEDDING_SHAPE = 512
MFCC_SHAPE = 120
NUMBER_OF_EPOCHS = 300

plot_figrues()