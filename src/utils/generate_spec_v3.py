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
DOWNLOAD_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/audio_all'
SPEC_DIR = f'/media/kokimame/Work_A_1TB/Project/Master_Files/spec_tagged_mcuts'

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

def spec_to_chunk(path):
    spec = compute_spectrogram(path)
    output_chunks = []
    for i in range(0, len(spec) // CHUNK_WIDTH):
        chunk = spec[:, i * CHUNK_WIDTH : (i + 1) * CHUNK_WIDTH]
        if chunk.shape[1] != CHUNK_WIDTH:
            continue
        output_chunks.append(chunk)
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

def audio_to_chunk(path, sr=SAMPLING_FREQUENCY):
    # (Feature size) = (sampling rate) x (duration) / (Hop size)
    # (Sample per chunk) = (Chunk width) x (Hop size)
    samples_per_chunk = CHUNK_WIDTH * HOP_LENGTH
    y, sr = librosa.load(path, sr=SAMPLING_FREQUENCY)
    audio_chunks = []

    for i in range(0, len(y) // samples_per_chunk):
        # Approximating waveforms of corresponding chunked spectrogram
        audio = y[i * samples_per_chunk : (i + 1) * samples_per_chunk]
        if len(audio) != samples_per_chunk:
            continue
        audio_chunks.append(audio)
    return audio_chunks

def projection_setup(path, audio_dir, spec_dir):
    # print(f'Setup for projection: Chunking {path}')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)
    spec_chunks = spec_to_chunk(path)
    audio_chunks = audio_to_chunk(path)
    audio_nums = []

    for spec_chunk, audio_chunk in zip(spec_chunks, audio_chunks):
        spec_existing = len(glob.glob(f'{spec_dir}/*.npy'))
        audio_existing = len(glob.glob(f'{audio_dir}/*.wav'))
        assert spec_existing == audio_existing, (path, audio_dir, spec_dir)
        spec_path = f'{spec_dir}/{spec_existing + 1:06d}.npy'
        audio_path = f'{audio_dir}/{audio_existing + 1:06d}.wav'
        audio_nums.append(f'{audio_existing + 1:06d}.wav')
        np.save(spec_path, spec_chunk)
        librosa.output.write_wav(audio_path, audio_chunk, SAMPLING_FREQUENCY)

    return spec_chunks, audio_nums

if __name__ == '__main__':
    audio_paths = glob.glob(f'{DOWNLOAD_DIR}/*/*.mp3')

    for path in tqdm(audio_paths):
        chunks = spec_to_chunk(path)
        save_chucks(chunks, path)
