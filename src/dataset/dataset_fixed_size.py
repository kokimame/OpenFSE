import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import cs_augment


class DatasetFixed(Dataset):
    """
    Dataset object returns 4 sounds for a given label.
    Given features are pre-processed to have a same particular shape.
    """

    def __init__(self, data, labels, h=128, w=128, data_aug=1):
        """
        Initialization function for the MOVEDataset object
        :param data: features
        :param labels: labels of features (should be in the same order as features)
        :param h: height of features (number of bins, e.g. 12 or 23)
        :param w: width of features (number of frames in the temporal dimension)
        :param data_aug: whether to apply data augmentation to each sound (1 or 0)
        """
        self.data = data  # spectrogram features
        self.labels = np.array(labels)  # labels of the features

        self.seed = 42  # random seed
        self.h = h  # height of a feature
        self.w = w  # width of a feature
        self.data_aug = data_aug  # whether to apply data augmentation to each sound

        self.labels_set = set(self.labels)  # the set of labels

        # dictionary to store which indexes belong to which label
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        self.clique_list = []  # list to store all cliques

        # adding some cliques multiple times depending on their size
        for label in self.label_to_indices.keys():
            if self.label_to_indices[label].size < 2:
                pass
            elif self.label_to_indices[label].size < 6:
                self.clique_list.extend([label] * 1)
            elif self.label_to_indices[label].size < 10:
                self.clique_list.extend([label] * 2)
            elif self.label_to_indices[label].size < 14:
                self.clique_list.extend([label] * 3)
            else:
                self.clique_list.extend([label] * 4)

    def __getitem__(self, index):
        """
        getitem function for the Dataset object
        :param index: index of the clique picked by the dataloader
        :return: 4 sounds and their labels from the picked clique
        """
        label = self.clique_list[index]  # getting the clique chosen by the dataloader

        assert self.label_to_indices[label].size > 3
        # selecting 4 sounds from the given clique
        idx1, idx2, idx3, idx4 = np.random.choice(self.label_to_indices[label], 4, replace=False)
        item1, item2, item3, item4 = self.data[idx1], self.data[idx2], self.data[idx3], self.data[idx4]
        indices = [idx1, idx2, idx3, idx4]
        items_i = [item1, item2, item3, item4]  # list for storing selected sounds

        items = []

        # pre-processing each sound separately
        for item in items_i:
            if self.data_aug == 1:  # applying data augmentation to the sound
                item = cs_augment(item)
            # if the sound is longer than the required width, choose a random start point to crop
            if item.shape[2] >= self.w:
                p_index = [i for i in range(0, item.shape[2] - self.w + 1)]
                if len(p_index) != 0:
                    start = np.random.choice(p_index)
                    temp_item = item[:, :, start:start + self.w]
                    items.append(temp_item)
            else:  # if the sound is shorter than the required width, zero-pad the end
                items.append(torch.cat((item, torch.zeros([1, self.h, self.w - item.shape[2]]).double()), 2))

        return torch.stack(items, 0), (label, indices)

    def __len__(self):
        """
        Size of the Dataset object
        :return: length of the clique list containing all the cliques (multiple cliques included for larger ones)
        """
        return len(self.clique_list)
