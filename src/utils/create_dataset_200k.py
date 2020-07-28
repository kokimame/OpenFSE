import numpy as np
import torch
import os
import glob
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

TRAIN_SPLIT = 0.8
CHUNK_WIDTH = 128
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
DATADIR = f'{ROOTDIR}/spec_200k'
DATASET_NAME = f'mix_200k'

paths = glob.glob(os.path.join(DATADIR, '*.npy'))
# Looking up paths by the label to which the sound belongs
path_lookup = {}
for path in paths:
    label = Path(path).stem
    paths = path_lookup.get(label, [])
    paths.append(path)
    path_lookup[label] = paths
path_lookup = OrderedDict(sorted(path_lookup.items(), key=lambda x: -len(x[1])))

# Setup dataset
train_data, train_labels = [], []
val_data, val_labels = [], []
data_list = list(path_lookup.items())
min_sounds_per_label = min(len(paths) for label, paths in data_list)
print(f'Sounds per Label: {min_sounds_per_label}')

with tqdm(total=len(data_list)) as t:
    for label, paths in data_list:
        t.set_description(desc=f'Size (Train: {len(train_data):8d}|Validation: {len(val_data):8d})')

        label, paths = label[:min_sounds_per_label], paths[:min_sounds_per_label]
        # Both train and test dataset contain each label at the same ratio
        last_train_index = int(len(paths) * TRAIN_SPLIT)
        for i in range(len(paths)):
            spec = np.load(paths[i])
            if spec.shape[1] != CHUNK_WIDTH:
                continue
            tensor = torch.from_numpy(spec).double()
            if i <= last_train_index:
                train_data.append(tensor.unsqueeze(0))
                train_labels.append(label)
            else:
                val_data.append(tensor.unsqueeze(0))
                val_labels.append(label)
        t.update()

# Create data file based on the datasets
for run_type, data, labels in [('train', train_data, train_labels), ('val', val_data, val_labels)]:
    dataset_dict = {'data': data, 'labels': labels}
    postfix = '_1' if run_type == 'train' else ''
    print(f'# of specs in {run_type} dataset: {len(data)}')
    print(f'# of labels in {run_type} dataset: {len(set(labels))}')
    torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{run_type}{postfix}.pt'))

    # Create annotation file
    if run_type == 'val':
        ytrue = []
        for i in tqdm(range(len(labels)), desc=f'Creating annotation file for {run_type}'):
            main_label = labels[i]  # label of the ith sound
            sub_ytrue = []
            for j in range(len(labels)):
                if labels[j] == main_label and i != j:  # checking whether the ith and jth song has the same label
                    sub_ytrue.append(1)
                else:
                    sub_ytrue.append(0)
            ytrue.append(sub_ytrue)
        ytrue = torch.Tensor(ytrue)
        torch.save(ytrue, os.path.join(ROOTDIR, f'{DATASET_NAME}_{run_type}_ytrue.pt'))
