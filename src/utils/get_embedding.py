import torch
import os
import glob
import json
import csv
import numpy as np
import random
from models.model_vgg import VGGModel

SAVED_MODEL = 'saved_models/model_tag_train_2020-04-28_03:09:29.309522.pt'
ONTROLOGY = '../json/ontology.json'
ROOTDIR = f'{os.environ["HOME"]}/Project/Master_Files'
DATADIR = f'{ROOTDIR}/spec_tagged'

with open(ONTROLOGY, 'r') as f:
    label_json = json.load(f)

ontology_lookup = {}
for entry in label_json:
    label_id = entry['id'].replace('/', '_')
    assert label_id not in ontology_lookup.keys()
    ontology_lookup[label_id] = entry


files = glob.glob(os.path.join(DATADIR, '*', '*.npy'))
random.shuffle(files)
model = VGGModel(emb_size=256)
model.load_state_dict(torch.load(SAVED_MODEL))
model.eval()

emb_tsv = []
label_tsv = []
for file in files[:1000]:
    spec = np.load(file)
    tensor = torch.from_numpy(spec)
    # Use parent directory as a label
    spec = tensor.unsqueeze(0).unsqueeze(0)
    label = file.split('/')[-2]
    emb = model(spec).squeeze()

    emb_tsv.append(emb.tolist())
    label_tsv.append([ontology_lookup[label]['name']])

    # print(f'emb shape {emb.shape}  <- spec shape {spec.shape}')


with open('data/emb.tsv', 'w') as f:
    for emb in emb_tsv:
        csv.writer(f, delimiter='\t').writerow(emb)
with open('data/label.tsv', 'w') as f:
    for label in label_tsv:
        csv.writer(f, delimiter='\t').writerow(label)
