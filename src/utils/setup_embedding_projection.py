import torch
import os
import glob
import json
import csv
import numpy as np
import random
from pathlib import Path
from models.model_vgg import VGGModel
from models.model_vgg_v2 import VGGModelV2
from src.utils import generate_spec_v3
from tqdm import tqdm

TOTAL_EMBEDDINGS = 2000
ONTROLOGY = '../../json/ontology.json'
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
USED_DATASET_PATH = f'{ROOTDIR}/tag_top100_val.pt'
AUDIODIR = f'{ROOTDIR}/audio_all'
OUTPUTDIR = f'{ROOTDIR}/projector'

used_dataset = torch.load(USED_DATASET_PATH)
used_labels = set(used_dataset['labels'])

with open(ONTROLOGY, 'r') as f:
    label_json = json.load(f)

ontology_lookup = {}
for entry in label_json:
    label_id = entry['id'].replace('/', '_')
    assert label_id not in ontology_lookup.keys()
    ontology_lookup[label_id] = entry

print(', '.join([ontology_lookup[label]['name'] for label in used_labels]))

paths = glob.glob(os.path.join(AUDIODIR, '*', '*.mp3'))
paths = [path for path in paths if Path(Path(path).parent).stem in used_labels]
random.shuffle(paths)
label_counts = {}
emb_tsv = []
label_tsv = []
audio_tsv = []

for path in tqdm(paths):
    # Use parent directory as a label
    label = Path(Path(path).parent).stem

    label_count = label_counts.get(label, 0)
    if label_count >= TOTAL_EMBEDDINGS / len(used_labels):
        continue

    spec_chunks, audio_nums = generate_spec_v3.projection_setup(path, f'{OUTPUTDIR}/audio/{label}', f'{OUTPUTDIR}/spec/{label}')
    for spec, audio_num in zip(spec_chunks, audio_nums):
        label_tsv.append([ontology_lookup[label]['name']])
        audio_tsv.append([f'{label}/{audio_num}'])

    label_counts[label] = label_count + len(spec_chunks)

with open(f'{OUTPUTDIR}/label.tsv', 'w') as f:
    for label in label_tsv:
        csv.writer(f, delimiter='\t').writerow(label)
with open(f'{OUTPUTDIR}/audio.tsv', 'w') as f:
    for audio_path in audio_tsv:
        csv.writer(f, delimiter='\t').writerow(audio_path)