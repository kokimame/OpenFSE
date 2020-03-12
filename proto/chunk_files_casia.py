import os
import glob
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import warnings
# Suppress soundfile warning on loading MP3
warnings.simplefilter("ignore")

DLTOP = f"{os.environ['HOME']}/Project/Master_Files/audio_v2"
SPECTOP = f"{os.environ['HOME']}/Project/Master_Files/spec_casia_v2"

# Check if the directories exist
for dir in [DLTOP, SPECTOP]:
    if not os.path.exists(dir):
        os.makedirs(dir)

CHUNK_PER_SOUND = 5
CHUNK_WIDTH = 128

def compute_spectrogram(path, sr=16000, n_mels=128):
    try:
        audio, sr = librosa.load(path, sr=sr)
    except:
        audio, o_sr = sf.read(path)
        audio = librosa.core.resample(audio, o_sr, sr)
    audio_rep = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=512, n_fft=1024, n_mels=n_mels, power=1.
    )
    audio_rep = np.log(audio_rep + np.finfo(np.float32).eps)
    return audio_rep

def pad_2d_horizontal(a2d, target_length):
    h, w = a2d.shape
    assert w <= target_length
    padding = target_length - w
    concat = np.concatenate((a2d, np.zeros([h, padding])), axis=1)
    return concat

# Currently, the maximum duration is 20 sec.
# Therefore, the maximum chunk size is 625 (20 * 16000 / 512).
# Then, set the chunk width to 650 / 5 = 130 ~ 128 with zero padding.
def spec_to_chunk(spec):
    output_chunks = []
    for chunk in np.array_split(spec, CHUNK_PER_SOUND, axis=1):
        padded_chunk = pad_2d_horizontal(chunk, CHUNK_WIDTH)
        output_chunks.append(padded_chunk)
    return output_chunks

def save_chucks(chunks, original_path):
    audio_id = Path(original_path).stem

    if not os.path.exists(f"{SPECTOP}/{audio_id}/"):
        os.makedirs(f"{SPECTOP}/{audio_id}/")

    for i, chunk in enumerate(chunks):
        chunk_path = f"{SPECTOP}/{audio_id}/" + "%03d" % (i + 1)
        np.save(chunk_path, chunk)

audio_paths = glob.glob(f"{DLTOP}/*.mp3")

for path in tqdm(audio_paths):
    spec = compute_spectrogram(path)
    chunks = spec_to_chunk(spec)
    save_chucks(chunks, path)

