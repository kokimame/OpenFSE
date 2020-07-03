import soundfile as sf
import os
import glob
import json
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm

import torch
from models.model_vgg import VGGModel
from models.model_vgg_v2 import VGGModelV2
from models.model_vgg_dropout import VGGModelDropout
from src.utils import generate_spec_v3


SAVED_MODEL = '../saved_models/model_unique5_2020-07-03_01:19:36.119977.pt'
ONTROLOGY = '../data/ontology.json'
ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
OUTPUTDIR = f'{ROOTDIR}/unique5'

AUDIO_CHUNKS = f'{OUTPUTDIR}/audio'

ontology_lookup = {}
with open(ONTROLOGY, 'r') as f:
    label_json = json.load(f)
for entry in label_json:
    label_id = entry['id'].replace('/', '_')
    assert label_id not in ontology_lookup.keys()
    ontology_lookup[label_id] = entry

paths = glob.glob(os.path.join(AUDIO_CHUNKS, '*', '*.wav'))
model = VGGModelDropout(emb_size=512)
model.load_state_dict(torch.load(SAVED_MODEL))
model.eval()

audio_tsv = []
label_tsv = []
emb_tsv = []

for path in tqdm(paths):
    label = Path(Path(path).parent).stem
    filename = Path(path).name

    spec_chunks = generate_spec_v3.spec_to_chunk(path)
    for spec in spec_chunks:
        tensor = torch.from_numpy(spec).double()
        spec = tensor.unsqueeze(0).unsqueeze(0)
        emb = model(spec).squeeze().tolist()

        label_tsv.append([ontology_lookup[label]['name']])
        audio_tsv.append([f'{label}/{filename}'])
        emb_tsv.append(emb)

with open(f'{OUTPUTDIR}/emb.tsv', 'w') as f:
    for emb in emb_tsv:
        csv.writer(f, delimiter='\t').writerow(emb)
with open(f'{OUTPUTDIR}/label.tsv', 'w') as f:
    for label in label_tsv:
        csv.writer(f, delimiter='\t').writerow(label)
with open(f'{OUTPUTDIR}/audio.tsv', 'w') as f:
    for audio_path in audio_tsv:
        csv.writer(f, delimiter='\t').writerow(audio_path)
