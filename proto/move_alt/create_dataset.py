import numpy as np
import torch
import os
import glob
from tqdm import tqdm

root_dir = f'{os.environ["HOME"]}/Project/Master_Files'
data_dir = f'{root_dir}/spec_casia_v2'
files = glob.glob(os.path.join(data_dir, '*', '*.npy'))
# Create data file
data = []
labels = []
for i in tqdm(range(1000), desc='Creating dataset dict'):
    spec = np.load(files[i])
    tensor = torch.from_numpy(spec)
    # Use parent directory as a label
    label = files[i].split('/')[-2]

    data.append(tensor.unsqueeze(0))
    labels.append(label)

dataset_dict = {'data': data, 'labels': labels}
torch.save(dataset_dict, os.path.join(root_dir, 'benchmark_ta_1.pt'))

# Create annotation file
# labels = torch.load(os.path.join(data_dir, 'benchmark_.pt'))['labels']
ytrue = []

for i in tqdm(range(len(labels)), desc='Creating annotation file'):
    main_label = labels[i]  # label of the ith song
    sub_ytrue = []
    for j in range(len(labels)):
        if labels[j] == main_label and i != j:  # checking whether the ith and jth song has the same label
            sub_ytrue.append(1)
        else:
            sub_ytrue.append(0)
    ytrue.append(sub_ytrue)

ytrue = torch.Tensor(ytrue)
torch.save(ytrue, os.path.join(root_dir, 'ytrue_benchmark_ta.pt'))
