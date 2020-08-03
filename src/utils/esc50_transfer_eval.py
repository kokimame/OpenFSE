import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
from utils import import_dataset_from_pt
from models.model_vgg_dropout import VGGModelDropout

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.openfse = VGGModelDropout(emb_size=d['emb_size'])
        self.openfse.load_state_dict(torch.load(f'{MODEL_PATH}'))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 50)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.openfse(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


with open('../data/model_defaults.json') as f:
    d = json.load(f)

MODEL_PATH = '../saved_models/model_multi_top100_2020-07-30_20:39:31.592502.pt'
ESC50_DIR = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50-master'
ESC50_SPEC = f'{ESC50_DIR}/spec'
TRAIN_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_train_1.pt'
VAL_PATH = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_all_val.pt'

classifier = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001)

train_data, train_labels, train_ids = import_dataset_from_pt('{}'.format(TRAIN_PATH), chunks=1)
train_set = MyDataset(train_data, train_labels)
print(f'Train data has been loaded! Length: {len(train_data)}')

val_data, val_labels, val_ids = import_dataset_from_pt('{}'.format(VAL_PATH), chunks=1)
val_set = MyDataset(val_data, val_labels)
print('Validation data has been loaded!')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4)

for epoch in range(100):
    avg_loss = []
    for items, labels in tqdm(train_loader):
        outputs = classifier(items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
    print(f'Epoch Avg. Loss: {np.mean(avg_loss)}')
