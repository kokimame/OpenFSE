import numpy as np
import torch
import os
import glob
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import json
import random
import time

ONTROLOGY = '../data/ontology.json'

ontology_lookup = {}
with open(ONTROLOGY, 'r') as f:
    label_json = json.load(f)
for entry in label_json:
    label_id = entry['id'].replace('/', '_')
    assert label_id not in ontology_lookup.keys()
    ontology_lookup[label_id] = entry

TRAIN_SPLIT = 0.8
CHUNK_WIDTH = 128
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
SPECDIR = f'{ROOTDIR}/spec_tagged_mcuts'
# DATASET_NAME = f'tag_top{USE_TOPK_LABELS}'
DATASET_NAME = f'unique5_random'

spec_paths = glob.glob(os.path.join(SPECDIR, '*', '*.npy'))
# Looking up paths by the label to which the sound belongs
path_lookup = {}
for spec_path in spec_paths:
    label = Path(Path(spec_path).parent).stem
    spec_paths = path_lookup.get(label, [])
    spec_paths.append(spec_path)
    path_lookup[label] = spec_paths
path_lookup = OrderedDict(sorted(path_lookup.items(), key=lambda x: -len(x[1])))

# Setup dataset
label_to_use = []
for label, spec_paths in path_lookup.items():
    class_name = ontology_lookup[label]['name']
    if class_name.lower() in ['bark', 'guitar', 'car', 'speech', 'thunderstorm']:
        print(class_name, ' --> ', len(spec_paths))
        label_to_use.append(label)

data_list = [[label, path_lookup[label]] for label in label_to_use]
min_sounds_per_label = min(len(paths) for label, paths in data_list)
print(f'Sounds per Label: {min_sounds_per_label}')

time.sleep(0.01)

train_data, train_labels = [], []
val_data, val_labels = [], []
with tqdm(total=len(data_list)) as t:
    for label, spec_paths in data_list:
        random.shuffle(spec_paths)
        spec_paths = spec_paths[:min_sounds_per_label]
        # Both train and test dataset contain each label at the same ratio
        last_train_index = int(len(spec_paths) * TRAIN_SPLIT)
        for i in range(len(spec_paths)):
            spec = np.load(spec_paths[i])
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
        t.set_description(desc=f'Dataset Size (Train: {len(train_data):5d}|Validation: {len(val_data):5d})')

time.sleep(0.01)

# Create data file based on the datasets
for run_type, data, labels in [('train', train_data, train_labels), ('val', val_data, val_labels)]:
    dataset_dict = {'data': data, 'labels': labels}
    postfix = '_1' if run_type == 'train' else ''
    print(f'# of specs in {run_type} dataset: {len(data)}')
    print(f'# of labels in {run_type} dataset: {len(set(labels))}')
    torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{run_type}{postfix}.pt'))
    time.sleep(0.01)

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
        torch.save(ytrue, os.path.join(ROOTDIR, f'ytrue_{run_type}_{DATASET_NAME}.pt'))
