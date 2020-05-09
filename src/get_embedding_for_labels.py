import torch
import os
import glob
import json
import csv
import numpy as np
import random
from pathlib import Path
from models.vgg_model import VGGModel
from models.vgg_model_v2 import VGGModelV2

SAVED_MODEL = 'saved_models/model_tag_top10_2020-05-07_13:13:45.914868.pt'
ONTROLOGY = '../json/ontology.json'
ROOTDIR = f'{os.environ["HOME"]}/Project/Master_Files'
USED_DATASET_PATH = '/home/kokimame/Project/Master_Files/tag_top10_val.pt'
DATADIR = f'{ROOTDIR}/spec_tagged_mcuts'

used_dataset = torch.load(USED_DATASET_PATH)
used_labels = set(used_dataset['labels'])

with open(ONTROLOGY, 'r') as f:
    label_json = json.load(f)

ontology_lookup = {}
for entry in label_json:
    label_id = entry['id'].replace('/', '_')
    assert label_id not in ontology_lookup.keys()
    ontology_lookup[label_id] = entry

for label in used_labels:
    print(ontology_lookup[label]['name'], end=', ')
exit()
paths = glob.glob(os.path.join(DATADIR, '*', '*.npy'))
random.shuffle(paths)
model = VGGModelV2(emb_size=256)
model.load_state_dict(torch.load(SAVED_MODEL))
model.eval()
label_counts = {}
emb_tsv = []
label_tsv = []
for path in paths:
    # Use parent directory as a label
    label = Path(Path(path).parent).stem
    if label not in used_labels:
        continue
    label_count = label_counts.get(label, 0) + 1
    if label_count >= 1000 / len(used_labels):
        continue

    spec = np.load(path)
    tensor = torch.from_numpy(spec).double()
    spec = tensor.unsqueeze(0).unsqueeze(0)

    emb = model(spec).squeeze()

    emb_tsv.append(emb.tolist())
    label_tsv.append([ontology_lookup[label]['name']])
    label_counts[label] = label_count

with open('data/emb.tsv', 'w') as f:
    for emb in emb_tsv:
        csv.writer(f, delimiter='\t').writerow(emb)
with open('data/label.tsv', 'w') as f:
    for label in label_tsv:
        csv.writer(f, delimiter='\t').writerow(label)
