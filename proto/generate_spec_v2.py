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
DATA_MAX = 10000
DOWNLOAD_DIR = f'{os.environ["HOME"]}/Project/Master_Files/audio_tagged'
SPEC_DIR = f'{os.environ["HOME"]}/Project/Master_Files/spec_tagged_v2'

# Check if the directories exist
if not os.path.exists(SPEC_DIR):
    os.makedirs(SPEC_DIR)

SAMPLING_FREQUENCY = 16000
NFTT = 1024
HOP_LENGTH = 512
CHUNK_PER_SOUND = 5
CHUNK_WIDTH = 128

def get_melspec(spec, n_mels):
    # Power spectrum
    powerspec = np.abs(spec)**2
    melspec = librosa.feature.melspectrogram(S=powerspec, n_mels=n_mels)
    S = librosa.power_to_db(melspec, np.max)
    return S

def get_spectrum(path, n_fft, hop_length, sr=None):
    try:
        y, sr = librosa.load(path, sr=sr)
    except:
        y, o_sr = sf.read(path)
        y = librosa.core.resample(y, o_sr, sr)
    y_stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
    return y_stft

def compute_spectrogram(path, sr=SAMPLING_FREQUENCY, n_mels=128):
    y_stft = get_spectrum(path, n_fft=NFTT, sr=sr, hop_length=HOP_LENGTH)
    melspec = get_melspec(y_stft, n_mels=n_mels)
    # Convert melspec from -80 to 0dB to range [0,1]
    spec = (melspec + 80.0) / 80.0
    return spec

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

def visualize_chunks(path, chunks):
    for i, chunk in enumerate(chunks):
        plt.subplot(2, 3, i + 1)
        plt.imshow(chunk)
    plt.xlabel(path)
    plt.show()

audio_paths = glob.glob(f'{DOWNLOAD_DIR}/*/*.mp3')

for path in tqdm(audio_paths[:DATA_MAX]):
    if MP3(path).info.length <= 15:
        continue
    spec = compute_spectrogram(path)
    chunks = spec_to_chunk(spec)
    save_chucks(chunks, path)
