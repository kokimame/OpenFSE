import torch
import numpy as np

working_pt = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_l3_train_1.pt'
working_ytrue = '/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/esc50_l3_val_ytrue.pt'
# working_pt = '/home/kokimame/Project/Master_Files/benchmark_vgg_1.pt'
# working_ytrue = '/home/kokimame/Project/Master_Files/benchmark_vgg.pt'
# failed_pt = '/home/kokimame/Project/Master_Files/benchmark_ta_1.pt'
# failed_ytrue = '/home/kokimame/Project/Master_Files/benchmark_ta.pt'
working_data = torch.load(working_pt)
working_ytrue = torch.load(working_ytrue)
# failed_data = torch.load(failed_pt)
# failed_ytrue = torch.load(failed_ytrue)

print(f'Working Data ({working_data["data"][0].shape}, {working_data["labels"][0]}),'
      f'Length = ({len(working_data["data"])}, {len(working_data["labels"])})')
print(f'Working YTrue Length = ({len(working_ytrue)})')

# print(f'Failed Data ({failed_data["data"][0].shape}, {failed_data["labels"][0]}),'
#       f'Length = ({len(failed_data["data"])}, {len(failed_data["labels"])})')
# print(f'Failed YTrue Length = ({len(failed_ytrue)})')


