# 200K dataset with Strategy 1
# Use multiple tags directly, not via embeddings

import numpy as np
import torch
import os
import glob
import csv
import json
from pathlib import Path
from collections import OrderedDict, Counter
from models.model_vgg_dropout import VGGModelDropout
from tqdm import tqdm

TRAIN_SPLIT = 0.8
CHUNK_WIDTH = 128
MODEL_TYPE = 'multi_top500_woma_3k'
MODEL_PATH = '../saved_models/model_multi_top500_2020-08-10_01:02:38.414963.pt'
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50'
DATADIR = f'{ROOTDIR}/spec'
DATASET_NAME = f'esc50_all'

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
data_list = list(path_lookup.items())
min_sounds_per_label = min(len(paths) for label, paths in data_list)
print(f'Sounds per Label: {min_sounds_per_label}')

# =========== Make spectrogram, labels etc
spec_list, label_list, id_list = [], [], []
with tqdm(total=len(data_list)) as t:
    for label, paths in data_list:
        t.set_description(desc=f'Total Size {len(spec_list):8d})')

        for i in range(len(paths)):
            filename, _ =  Path(paths[i]).stem.split('_')
            spec = np.load(paths[i])
            if spec.shape[1] != CHUNK_WIDTH:
                continue
            tensor = torch.from_numpy(spec).unsqueeze(0)
            spec_list.append(tensor)
            label_list.append(label)
            id_list.append(filename)
        t.update()
# ================= Convert spec to embedding
with open('../data/model_defaults.json') as f:
    d = json.load(f)
openfse = VGGModelDropout(emb_size=d['emb_size'])
openfse.load_state_dict(torch.load(f'{MODEL_PATH}'))
embedding_list = openfse(torch.cat(spec_list).unsqueeze(dim=1)).detach()



# Create data file based on the datasets
for run_type, data, labels, sound_ids in [(f'{MODEL_TYPE}', embedding_list, label_list, id_list)]:
    dataset_dict = {'data': data, 'labels': labels, 'sound_ids': sound_ids}
    print(f'# of specs in {run_type} dataset: {len(data)}')
    print(f'# of labels in {run_type} dataset: {len(set(labels))}')
    torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{run_type}.pt'))