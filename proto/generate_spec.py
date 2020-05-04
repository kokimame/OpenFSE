import os
import glob
import librosa
import warnings
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from mutagen.mp3 import MP3

# Suppress soundfile warning on loading MP3
warnings.simplefilter('ignore')

DOWNLOAD_DIR = f'{os.environ["HOME"]}/Project/Master_Files/audio_tagged'
SPEC_DIR = f'{os.environ["HOME"]}/Project/Master_Files/spec_tagged'

# Check if the directories exist
if not os.path.exists(SPEC_DIR):
    os.makedirs(SPEC_DIR)

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
    label = Path(Path(original_path).parent).stem

    if not os.path.exists(f'{SPEC_DIR}/{label}/'):
        os.makedirs(f'{SPEC_DIR}/{label}/')
    existing = len(glob.glob(f'{SPEC_DIR}/{label}/*.npy'))

    for i, chunk in enumerate(chunks):
        chunk_path = f'{SPEC_DIR}/{label}/{existing + i + 1:06d}.npy'
        np.save(chunk_path, chunk)

audio_paths = glob.glob(f'{DOWNLOAD_DIR}/*/*.mp3')

for path in tqdm(audio_paths[:10000]):
    if MP3(path).info.length <= 12:
        continue
    spec = compute_spectrogram(path)
    chunks = spec_to_chunk(spec)
    save_chucks(chunks, path)

