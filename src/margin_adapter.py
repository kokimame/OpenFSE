import os
import json
import numpy as np
import spacy
import torch
from tqdm import tqdm
from itertools import permutations

class MarginAdapter:

    def l2norm(self, vector):
        return vector / (np.linalg.norm(vector) + np.finfo(float).eps)


    def __init__(self, label_list, base_margin, description_file=None):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.base_margin = base_margin

        self.initialize_lookup(label_list, description_file)

        self.pairwise_dists = {}
        for l1, l2 in permutations(self.label_to_vector.keys(), 2):
            dist = np.linalg.norm(
                self.label_to_vector[l1] - self.label_to_vector[l2]
            )
            assert (l1, l2) not in self.pairwise_dists
            self.pairwise_dists[(l1, l2)] = dist
        self.average_dist = np.mean(list(self.pairwise_dists.values()))

        # Similar to pairwise distance
        # Either of them will be discarded
        self.d_semantic = {}
        for l1, l2 in permutations(self.label_to_vector.keys(), 2):
            dist = np.linalg.norm(
                self.label_to_vector[l1] - self.label_to_vector[l2]
            )
            assert (l1, l2) not in self.d_semantic
            # d_semantic (t_a, t_n) = || g(t_a) - g(t_n) || ^ 2 / (4 - beta)
            self.d_semantic[(l1, l2)] = dist ** 2 / (4 - self.base_margin)

    def initialize_lookup(self, label_list, description_file):
        nlp = spacy.load('en_core_web_md')
        description_lookup = {}
        if description_file:
            assert description_file.endswith('.json'), description_file
            assert os.path.exists(description_file), description_file
            with open(description_file, 'r') as f:
                description_lookup = json.load(f)

        self.label_to_vector = {}
        # Flatten list of list to list
        label_sequence = [label for sublist in label_list for label in sublist]
        for label in tqdm(label_sequence, desc='Computing word2vec for labels'):
            if label in description_lookup:
                description = description_lookup[label]
            else:
                description = label
            if label not in self.label_to_vector:
                vectors = np.asarray([word.vector for word in nlp(description)])
                sum_vector = np.sum(vectors, axis=0)

                # g(tag) = sum(word vectors) / || sum(word vectors) ||
                self.label_to_vector[label] = self.l2norm(sum_vector)

    def adapt(self, labels, sel_pos, sel_neg):
        margin_list = []
        for pos, neg in zip(sel_pos, sel_neg):
            pos_label = labels[pos]
            neg_label = labels[neg]
            dist = self.pairwise_dists[(pos_label, neg_label)]

            if dist > self.average_dist:
                margin_list.append([self.base_margin + 1])
            else:
                margin_list.append([self.base_margin - 1])

        adapted_margin = torch.tensor(margin_list).to(self.device)
        return adapted_margin

    def adapt2(self, labels, sel_pos, sel_neg):
        # Adaptive margin implementation based on
        # "A weakly supervised adaptive triplet loss for deep metric learning"
        margin_list = []
        for i_pos, i_neg in zip(sel_pos, sel_neg):
            pos_label = labels[i_pos]
            neg_label = labels[i_neg]
            semantic_similarity = self.d_semantic[(pos_label, neg_label)]
            margin_list.append([self.base_margin + semantic_similarity])
        adapted_margin = torch.tensor(margin_list).to(self.device)
        return adapted_margin
