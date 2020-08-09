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
DATASET_NAME = f'esc50_all'


#### Read tags ####
with open(f'{ROOTDIR}/meta/esc50.csv', 'r') as f:
    rows = []
    for row in csv.DictReader(f):
        row['category'] = row['category'].replace('_', ' ')
        rows.append(row)

###################

audio_paths = glob.glob(os.path.join(DATADIR, '*.wav'))
# Looking up paths by the label to which the sound belongs
path_lookup = {}
embedding_lookup = {}
audio_list, sr_list, file_list = [], [], []
for path in tqdm(audio_paths, desc='Loading audio files ...'):
    filename = Path(path).stem
    audio, sr = sf.read(path)
    audio_list.append(audio)
    sr_list.append(sr)
    file_list.append(filename)

# ----------------------------------
# Setup datasetutils/create_dataset_esc50_l3_v2.py:26
# ----------------------------------
feature_list, label_list, id_list = [], [], []
l3_outputs, _ = openl3.get_audio_embedding(audio_list, sr_list, content_type='env', embedding_size=512, hop_size=0.5)
for filename, output in tqdm(zip(file_list, l3_outputs)):
    feature = torch.Tensor(np.mean(output, axis=0))
    feature_list.append(feature)
    label_list.append(filename.split('-')[-1])
    id_list.append(filename)
feature_list = torch.stack(feature_list)

# Create data file based on the datasets
for data_type, features, labels, sound_ids in [(f'l3', feature_list, label_list, id_list)]:
    dataset_dict = {'data': features, 'labels': label_list, 'sound_ids': sound_ids}
    print(f'# of specs in {data_type} dataset: {len(features)}')
    print(f'# of labels in {data_type} dataset: {len(set(labels))}')
    torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{data_type}.pt'))