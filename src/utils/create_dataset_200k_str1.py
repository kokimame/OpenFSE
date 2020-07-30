# 200K dataset with Strategy 1
# Use multiple tags directly, not via embeddings

import numpy as np
import torch
import os
import glob
from pathlib import Path
from collections import OrderedDict, Counter
from tqdm import tqdm
import csv

USE_TOPK = 100
TRAIN_SPLIT = 0.8
CHUNK_WIDTH = 128
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
DATADIR = f'{ROOTDIR}/spec_200k'
DATASET_NAME = f'multi_top{USE_TOPK}'

#### Read tags ####
with open('../data/tags_ppc_top_500.csv', 'r') as f:
    rows = []
    rows.extend(csv.reader(f))

unique_tags = set()
total_tags = []
id_tag_lookup = {}
for row in rows[1:]:
    sound_id, tags = row[0], row[1:]
    total_tags.extend(tags)
    id_tag_lookup[sound_id] = tags
    unique_tags |= set(tags)

tag_counter = Counter(total_tags)
print(f'Least common common tags: {tag_counter.most_common()[-1]}')
###################

paths = glob.glob(os.path.join(DATADIR, '*.npy'))
# Looking up paths by the label to which the sound belongs
path_lookup = {}
id_category_lookup = {}
for path in paths:
    sound_id, numbering = Path(path).stem.split('_')
    tags = id_tag_lookup[sound_id]
    for tag in tags:
        paths = path_lookup.get(tag, [])
        paths.append(path)
        path_lookup[tag] = paths
path_lookup = OrderedDict(sorted(path_lookup.items(), key=lambda x: -len(x[1]))[:USE_TOPK])

# Setup dataset
train_data, train_labels, train_ids = [], [], []
val_data, val_labels, val_ids = [], [], []
data_list = list(path_lookup.items())
min_sounds_per_label = 30 #min(len(paths) for label, paths in data_list)
print(f'Sounds per Label: {min_sounds_per_label}')

with tqdm(total=len(data_list)) as t:
    for label, paths in data_list:
        t.set_description(desc=f'Size (Train: {len(train_data):8d}|Validation: {len(val_data):8d})')

        paths = paths[:min_sounds_per_label]
        # Both train and test dataset contain each label at the same ratio
        last_train_index = int(len(paths) * TRAIN_SPLIT)
        for i in range(len(paths)):
            sound_id, _ =  Path(paths[i]).stem.split('_')
            sound_id = int(sound_id)
            spec = np.load(paths[i])
            if spec.shape[1] != CHUNK_WIDTH:
                continue
            tensor = torch.from_numpy(spec)
            if i <= last_train_index:
                train_data.append(tensor.unsqueeze(0))
                train_labels.append(label)
                train_ids.append(sound_id)
            else:
                val_data.append(tensor.unsqueeze(0))
                val_labels.append(label)
                val_ids.append(sound_id)
        t.update()

# Create data file based on the datasets
for run_type, data, labels, sound_ids in [('train', train_data, train_labels, train_ids),
                                          ('val', val_data, val_labels, val_ids)]:
    dataset_dict = {'data': data, 'labels': labels, 'sound_ids': sound_ids}
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
