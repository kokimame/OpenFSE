import os
import glob
import librosa
import warnings
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm import trange
from mutagen.mp3 import MP3

# Suppress soundfile warning on loading MP3
warnings.simplefilter('ignore')
DOWNLOAD_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/audio'
MFCC_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/ESC-50/mfcc'

# Check if the directories exist
if not os.path.exists(MFCC_DIR):
    os.makedirs(MFCC_DIR)

SAMPLING_FREQUENCY = 16000
NFTT = 1024
HOP_LENGTH = 512
CHUNK_PER_SOUND = 5
CHUNK_WIDTH = 128

def save_chucks(original_path, chunks):
    sound_id = Path(original_path).stem

    if not os.path.exists(f'{MFCC_DIR}'):
        os.makedirs(f'{MFCC_DIR}')
    assert len(glob.glob(f'{MFCC_DIR}/{sound_id}_*.npy')) == 0, f'{sound_id} already has specs.'

    for i, chunk in enumerate([chunks]):
        chunk_path = f'{MFCC_DIR}/{sound_id}_{i + 1:06d}.npy'
        np.save(chunk_path, chunk)

    return len(chunks)


def compute_mfcc(filename, sr=22000):
    # zero pad and compute log mel spec
    try:
        audio, sr = librosa.load(filename, sr=sr, res_type='kaiser_fast')
    except:
        audio, o_sr = sf.read(filename)
        audio = librosa.core.resample(audio, o_sr, sr)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc, width=5, mode='nearest')
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=5, mode='nearest')

    feature = np.concatenate((np.mean(mfcc, axis=1), np.var(mfcc, axis=1),
                              np.mean(mfcc_delta, axis=1), np.var(mfcc_delta, axis=1),
                              np.mean(mfcc_delta2, axis=1), np.var(mfcc_delta2, axis=1)))

    return feature

if __name__ == '__main__':
    audio_files = glob.glob(f'{DOWNLOAD_DIR}/*.wav')
    pbar = trange(len(audio_files))
    error_count = 0

    for i in pbar:
        path = audio_files[i]

        try:
            chunks = compute_mfcc(path, sr=SAMPLING_FREQUENCY)

            if len(chunks) == 0:
                raise Exception('Why zero?')

            save_chucks(path, chunks)
            pbar.set_description(f'{len(chunks)} added')
            # visualize_chunks(path, chunks)

        except Exception as e:
            error_count += 1
            print(f'Error #{error_count} out of {i + 1}: {e}')