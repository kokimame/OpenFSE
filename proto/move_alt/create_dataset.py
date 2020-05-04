import numpy as np
import torch
import os
import glob
from tqdm import tqdm

TRAIN_SPLIT = 0.8
ROOTDIR = f'{os.environ["HOME"]}/Project/Master_Files'
DATADIR = f'{ROOTDIR}/spec_tagged_v2'
DATASET_NAME = 'tag_dense'

files = glob.glob(os.path.join(DATADIR, '*', '*.npy'))
dataset_size = min(10000, len(files))
last_train_index = int(dataset_size * TRAIN_SPLIT)
# Create data file
for run_type, data_range in [('train', range(last_train_index)), ('val', range(last_train_index, dataset_size))]:
    data, labels = [], []
    for i in tqdm(data_range, desc=f'Creating {run_type} dataset'):
        spec = np.load(files[i])
        tensor = torch.from_numpy(spec)
        # Use parent directory as a label
        label = files[i].split('/')[-2]
        data.append(tensor.unsqueeze(0))
        labels.append(label)

    dataset_dict = {'data': data, 'labels': labels}
    postfix = '_1' if run_type == 'train' else ''
    print(f'Label Size: {len(set(labels))}')
    # torch.save(dataset_dict, os.path.join(ROOTDIR, f'{DATASET_NAME}_{run_type}{postfix}.pt'))

    # Create annotation file
    if run_type == 'val':
        ytrue = []
        for i in tqdm(range(len(labels)), desc=f'Creating annotation file for {run_type}'):
            main_label = labels[i]  # label of the ith sound
            sub_ytrue = []
            for j in range(len(labels)):
                if labels[j] == main_label and i != j:  # checking whether the ith and jth song has the same label
                    sub_ytrue.append(1)
                else:
                    sub_ytrue.append(0)
            ytrue.append(sub_ytrue)
        ytrue = torch.Tensor(ytrue)
        # torch.save(ytrue, os.path.join(ROOTDIR, f'ytrue_{run_type}_{DATASET_NAME}.pt'))
