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

ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50'
DATADIR = f'{ROOTDIR}/mfcc'
DATASET_NAME = f'esc50_all'
CHUNK_WIDTH = -1
TRAIN_SPLIT = 0.8
#### Read tags ####
with open(f'{ROOTDIR}/meta/esc50.csv', 'r') as f:
    rows = []
    for row in csv.DictReader(f):
        row['category'] = row['category'].replace('_', ' ')
        rows.append(row)

unique_tags = set()
total_tags = []
filename_tag_lookup = {}
for row in rows:
    tag, filename = row['category'], row['filename'].replace('.wav', '')
    total_tags.append(tag)
    filename_tag_lookup[filename] = tag
    unique_tags |= set(tag)

tag_counter = Counter(total_tags)
print(f'Least common common tags: {tag_counter.most_common()[:-1]}')
###################

paths = glob.glob(os.path.join(DATADIR, '*.npy'))
# Looking up paths by the label to which the sound belongs
path_lookup = {}
id_category_lookup = {}
for path in paths:
    filename, numbering = Path(path).stem.split('_')
    tag = filename_tag_lookup[filename]
    paths = path_lookup.get(tag, [])
    paths.append(path)
    path_lookup[tag] = paths
path_lookup = OrderedDict(sorted(path_lookup.items(), key=lambda x: -len(x[1])))

# Setup dataset
train_data, train_labels, train_ids = [], [], []
val_data, val_labels, val_ids = [], [], []
data_list = list(path_lookup.items())
min_sounds_per_label = min(len(paths) for label, paths in data_list)
print(f'Sounds per Label: {min_sounds_per_label}')

mfcc_list, label_list, id_list = [], [], []
with tqdm(total=len(data_list)) as t:
    for label, paths in data_list:
        t.set_description(desc=f'Total Size {len(mfcc_list):8d})')

        for i in range(len(paths)):
            filename, _ =  Path(paths[i]).stem.split('_')
            mfcc = np.load(paths[i])
            tensor = torch.from_numpy(mfcc).unsqueeze(0)
            mfcc_list.append(tensor)
            label_list.append(label)
            id_list.append(filename)
        t.update()


# Create data file based on the datasets
for data_type, features, labels, sound_ids in [(f'mfcc', mfcc_list, label_list, id_list)]:
    dataset_dict = {'data': features, 'labels': label_list, 'sound_ids': sound_ids}
    print(f'# of specs in {data_type} dataset: {len(features)}')
    print(f'# of labels in {data_type} dataset: {len(set(labels))}')
    torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{data_type}.pt'))