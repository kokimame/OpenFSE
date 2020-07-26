import numpy as np
import csv
import spacy

from tqdm import tqdm
from collections import Counter

with open('tags_ppc_top_500.csv', 'r') as f:
    rows = []
    rows.extend(csv.reader(f))

unique_tags = set()
total_tags = []
original_tags = set()
for row in rows[1:]:
    original_tags.add(' '.join(row[1:]))
    for tag in row[1:]:
        unique_tags.add(tag)
        total_tags.append(tag)
    if len(original_tags) > 5000:
        break

print(f'Top Most common tags: {Counter(total_tags).most_common(5)}')
print(f'Unique tags: {len(unique_tags)}')


nlp = spacy.load('en_core_web_md')
tag_vector_lookup = {}

for tag in tqdm(original_tags, desc='Computing Word2Vec'):
    vectors = np.asarray([word.vector for word in nlp(tag)])
    vector = np.sum(vectors, axis=0)
    tag_vector_lookup[tag] = vector #/ (np.linalg.norm(vector) + 0.0001)

with open('tag_emb.tsv', 'w') as femb:
    with open('tag_meta.tsv', 'w') as fmeta:
        for tag, emb in tag_vector_lookup.items():
            csv.writer(fmeta, delimiter='\t').writerow([tag])
            csv.writer(femb, delimiter='\t').writerow(emb)

for tag, emb in tag_vector_lookup.items():
    dists = []
    for tag2, emb2 in tag_vector_lookup.items():
        if tag == tag2: continue
        dists.append((tag2, np.linalg.norm(emb - emb2)))
    dists.sort(key=lambda x: x[1])
    print(f'{tag} ---> {dists[:5]}')