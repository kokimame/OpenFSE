from tinytag import TinyTag
from tqdm import trange
import glob


## Average duration: 5.47 sec:  99%|█████████▊| 169529/172107 [30:26<00:56, 45.66it/s]
audio_files = glob.glob('/media/kokimame/Work_A_1TB/Project/Master_Files/audio_200k/*.mp3')

total_duration = 0
total_count = 0

pbar = trange(len(audio_files))
for i in pbar:
    try:
        file = audio_files[i]
        total_duration += TinyTag.get(file).duration
        total_count += 1
        pbar.set_description(f'Average duration: {total_duration / (total_count):.2f} sec')
    except:
        pass



