import os
import json
import numpy as np
import spacy
import torch
from tqdm import tqdm
from itertools import permutations

class MarginAdapter:
    def __init__(self, label_list, description_file=None):
        nlp = spacy.load('en_core_web_md')

        description_lookup = {}
        if description_file:
            assert description_file.endswith('.json')
            assert os.path.exists(description_file)
            with open(description_file, 'r') as f:
                description_lookup = json.load(f)

        self.label_to_vector = {}

        label_sequence = [label for sublist in label_list for label in sublist]
        for label in tqdm(label_sequence, desc='Computing word2vec for labels'):
            if label in description_lookup:
                description = description_lookup[label]
            else:
                description = label
            if label not in self.label_to_vector:
                vectors = np.asarray([word.vector for word in nlp(description)])
                mean_vector = np.mean(vectors, axis=0)
                self.label_to_vector[label] = mean_vector

        self.pairwise_dists = {}
        for l1, l2 in permutations(self.label_to_vector.keys(), 2):
            dist = np.linalg.norm(
                self.label_to_vector[l1] - self.label_to_vector[l2]
            )
            assert (l1, l2) not in self.pairwise_dists
            self.pairwise_dists[(l1, l2)] = dist
        self.average_dist = np.mean(list(self.pairwise_dists.values()))


    def adapt(self, base_margin, labels, sel_pos, sel_neg):
        margin_list = []
        for pos, neg in zip(sel_pos, sel_neg):
            pos_label = labels[pos]
            neg_label = labels[neg]
            try:
                dist = self.pairwise_dists[(pos_label, neg_label)]

                if dist > self.average_dist:
                    margin_list.append([base_margin + 1])
                else:
                    margin_list.append([base_margin - 1])
            except:
                margin_list.append([0])

        adapted_margin = torch.tensor(margin_list)
        return adapted_margin
