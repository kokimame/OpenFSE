import openl3
import soundfile as sf
import os
import glob
import torch
import json
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm
from collections import Counter, OrderedDict

TRAIN_SPLIT = 0.8
CHUNK_WIDTH = 128
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50'
DATADIR = f'{ROOTDIR}/audio'
DATASET_NAME = f'esc50_l3'


#### Read tags ####
with open(f'{ROOTDIR}/meta/esc50.csv', 'r') as f:
    rows = []
    for row in csv.DictReader(f):
        row['category'] = row['category'].replace('_', ' ')
        rows.append(row)

tags_set = set()
total_tags = []
filename_tag_lookup = {}
for row in rows:
    label, filename = row['category'], Path(row['filename']).stem
    total_tags.append(label)
    filename_tag_lookup[filename] = label
    tags_set.add(label)

tag_counter = Counter(total_tags)
print(f'Least common common tags: {tag_counter.most_common()[:-1]}')
###################

audio_paths = glob.glob(os.path.join(DATADIR, '*.wav'))
# Looking up paths by the label to which the sound belongs
path_lookup = {}

for path in audio_paths:
    filename = Path(path).stem
    label = filename_tag_lookup[filename]
    paths = path_lookup.get(label, [])
    paths.append(path)
    path_lookup[label] = paths
path_lookup = OrderedDict(sorted(path_lookup.items(), key=lambda x: -len(x[1])))

emb_lookup = {}
for label, paths in path_lookup.items():
    audio_list, sr_list = [], []
    for path in tqdm(paths, desc='Loading audio files ...'):
        audio, sr = sf.read(path)
        audio_list.append(audio)
        sr_list.append(sr)
    emb_list, _ = openl3.get_audio_embedding(audio_list, sr_list, content_type='env', embedding_size=512, hop_size=0.5)
    emb_lookup[label] = emb_list

# ----------------------------------
# Setup dataset
# ----------------------------------
train_data, train_labels, train_ids = [], [], []
val_data, val_labels, val_ids = [], [], []
data_list = list(emb_lookup.items())
min_sounds_per_label = min(len(paths) for label, paths in data_list)
print(f'Sounds per Label: {min_sounds_per_label}')

with tqdm(total=len(data_list)) as t:
    for label, embeddings in data_list:
        t.set_description(desc=f'Size (Train: {len(train_data):8d}|Validation: {len(val_data):8d})')
        # Both train and test dataset contain each label at the same ratio
        last_train_index = int(len(embeddings) * TRAIN_SPLIT)
        for i in range(len(embeddings)):
            embedding = torch.Tensor(np.mean(embeddings[i], axis=0))
            if i <= last_train_index:
                train_data.append(embedding)
                train_labels.append(label)
                train_ids.append('nan')
            else:
                val_data.append(embedding)
                val_labels.append(label)
                val_ids.append('nan')
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
