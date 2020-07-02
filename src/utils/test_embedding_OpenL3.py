import openl3
import soundfile as sf
import os
import glob
import json
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm

ROOTDIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files'
OUTPUTDIR = f'{ROOTDIR}/unique5'
AUDIO_CHUNKS = f'{OUTPUTDIR}/audio'
ONTROLOGY = '../data/ontology.json'

with open(ONTROLOGY, 'r') as f:
    label_json = json.load(f)
ontology_lookup = {}
for entry in label_json:
    label_id = entry['id'].replace('/', '_')
    assert label_id not in ontology_lookup.keys()
    ontology_lookup[label_id] = entry


paths = glob.glob(os.path.join(AUDIO_CHUNKS, '*', '*.wav'))

audio_tsv = []
label_tsv = []
emb_tsv = []

audio_list, sr_list = [], []
for path in tqdm(paths, desc='Loading audio files ...'):
    audio, sr = sf.read(path)
    audio_list.append(audio)
    sr_list.append(sr)
    filename = Path(path).name
    label = Path(Path(path).parent).stem
    label_tsv.append([ontology_lookup[label]['name']])
    audio_tsv.append([f'{label}/{filename}'])

emb_list, _ = openl3.get_audio_embedding(audio_list, sr_list, content_type='env', hop_size=0.5)
for emb in tqdm(emb_list, desc="Creating embeddings ..."):
    emb = np.mean(emb, axis=0).tolist()
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
