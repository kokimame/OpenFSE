import numpy as np
import torch
import csv
from torch.utils.data import DataLoader

class TrainLoader(DataLoader):
    def __init__(self, train_set, *args, **kwargs):
        super().__init__(train_set, *args, **kwargs)
        self.hard_indices = []

    def save_hard_indices(self, indices):
        self.hard_indices.extend(indices)
        with open('hard_indices.csv', 'w') as f:
            for row in self.hard_indices:
                csv.writer(f).writerow(row)

