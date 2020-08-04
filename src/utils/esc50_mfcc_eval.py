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
        self.lin_bn = nn.BatchNorm1d(120, affine=False)
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
    def __init__(self, data, labels):
        self.data = data
        labels_list = list(set(labels))
        self.labels = torch.LongTensor([
            labels_list.index(label) for label in labels
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


with open('../data/model_defaults.json') as f:
    d = json.load(f)

MODEL_PATH = '../saved_models/multi_top100_ma_8k.pt'
ESC50_DIR = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50'
ESC50_SPEC = f'{ESC50_DIR}/spec'
TRAIN_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_mfcc_train_1.pt'
VAL_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_mfcc_val.pt'
INPUT_SHAPE = 120

train_data, train_labels, train_ids = import_dataset_from_pt('{}'.format(TRAIN_PATH), chunks=1)
train_set = ESC50Dataset(train_data, train_labels)
print(f'Train data has been loaded! | Data Size: {len(train_data)}')

val_data, val_labels, val_ids = import_dataset_from_pt('{}'.format(VAL_PATH), chunks=1)
val_set = ESC50Dataset(val_data, val_labels)
print(f'Validation data has been loaded! | Data Size: {len(val_data)}')
print()
time.sleep(1)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)

classifier = Classifier(input_shape=INPUT_SHAPE, version='mini')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.01)

train_loss, val_loss = [], []
train_xticks, val_xticks = [], []
train_acc, val_acc = [], []
NUMBER_OF_EPOCHS = 200

for epoch in range(NUMBER_OF_EPOCHS):
    train_pbar = tqdm(train_loader)
    train_avg_loss = []
    for inputs, targets in train_pbar:

        inputs = torch.squeeze(inputs)
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
            inputs = torch.squeeze(inputs)
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(targets.tolist(), predictions.tolist())
            val_avg_acc.append(accuracy)
            val_avg_loss.append(loss.item())
            val_pbar.set_description(f'*** Avg. Val Loss: {np.mean(val_avg_loss):.2f} | Accuracy: {accuracy:.2f}')

        val_loss.append(np.mean(val_avg_loss))
        val_acc.append(np.mean(val_avg_acc))
        val_xticks.append(epoch)

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(train_xticks, train_loss, label='Train Loss')
plt.plot(val_xticks, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(np.arange(0, NUMBER_OF_EPOCHS + 1, NUMBER_OF_EPOCHS//20))

plt.subplot(1, 2, 2)
plt.plot(val_xticks, val_acc, label='Val Acc.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(np.arange(0, NUMBER_OF_EPOCHS + 1, NUMBER_OF_EPOCHS//20))

plt.show()