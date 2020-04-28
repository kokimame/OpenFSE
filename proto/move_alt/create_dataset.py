import numpy as np
import torch
import os
import glob
from tqdm import tqdm

DATASET_SIZE = 9000
TRAIN_SPLIT = 0.8
ROOTDIR = f'{os.environ["HOME"]}/Project/Master_Files'
DATADIR = f'{ROOTDIR}/spec_tagged'
DATASET_NAME = 'tag'
last_train_index = int(DATASET_SIZE * TRAIN_SPLIT)

files = glob.glob(os.path.join(DATADIR, '*', '*.npy'))
# Create data file
for desc, data_range in [('train', range(last_train_index)), ('val', range(last_train_index, DATASET_SIZE))]:
    data, labels = [], []
    for i in tqdm(data_range, desc=f'Creating {desc} dataset'):
        spec = np.load(files[i])
        tensor = torch.from_numpy(spec)
        # Use parent directory as a label
        label = files[i].split('/')[-2]
        data.append(tensor.unsqueeze(0))
        labels.append(label)

    dataset_dict = {'data': data, 'labels': labels}
    postfix = '_1' if desc == 'train' else ''
    torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{desc}{postfix}.pt'))

    # Create annotation file
    ytrue = []
    for i in tqdm(range(len(labels)), desc=f'Creating annotation file for {desc}'):
        main_label = labels[i]  # label of the ith song
        sub_ytrue = []
        for j in range(len(labels)):
            if labels[j] == main_label and i != j:  # checking whether the ith and jth song has the same label
                sub_ytrue.append(1)
            else:
                sub_ytrue.append(0)
        ytrue.append(sub_ytrue)
    ytrue = torch.Tensor(ytrue)
    torch.save(ytrue, os.path.join(ROOTDIR, f'ytrue_{desc}_{DATASET_NAME}.pt'))
